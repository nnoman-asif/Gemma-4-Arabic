"""Stage 2 Direct Preference Optimization (DPO) alignment pipeline for Gemma-4-Arabic (E4B & 26B-A4B MoE).

Key architectural guarantees:
1. Formats preference rows into native TRL conversational format (`prompt`: `prompt_messages`,
   `chosen`: `chosen_only_messages`, `rejected`: `rejected_only_messages`, `chat_template_kwargs`)
   so `DPOTrainer` NEVER concatenates full-conversation strings or duplicates prompts.
2. On `Gemma-4-26B-A4B` (`is_moe=True`), aligns the training-time `chat_template` so non-thinking
   completed assistant turns include `<|channel>thought\\n<channel|>` matching `add_generation_prompt=True`,
   asserting `full_chosen_ids[:len(prompt_ids)] == prompt_ids` before training starts, and restores
   the unmodified upstream `chat_template.jinja` on save.
3. Anchors KL divergence cleanly to the frozen Stage 1 SFT `"ref"` adapter created by `DPOTrainer`
   (`peft>=0.21.3` supports `"ref"` alongside 3D `target_parameters` on `26B-A4B`), without
   destructive mid-training parameter overwrites.
4. Saves only the `"default"` policy adapter (cleaning up any duplicate `ref/` subdirectory and
   intermediate `checkpoint-*` folders).
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import logging
import math
import os
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional

def _enforce_deterministic_dpo_kernels() -> None:
    """Enforce bitwise-identical decoder-block, RMSNorm/MLP/RoPE, sliding-attention, LM-head,
    and MoE kernel execution across `_precompute_ref_logps` (`no_grad`, `GC=False`),
    `verify_step0_dpo_ref_alignment`, and `trainer.train()` (`enable_grad`, non-reentrant GC on E4B)
    before Unsloth initializes module and kernel patches.
    """
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["UNSLOTH_FUSED_LINEAR"] = "0"
    os.environ["UNSLOTH_GEMMA4_FLASH_SLIDING"] = "0"
    os.environ["UNSLOTH_BLOCK_COMPILE"] = "0"
    os.environ["UNSLOTH_MOE_LORA_FOLD"] = "0"
    os.environ["UNSLOTH_MOE_GC_KEEP_BASE"] = "0"
    os.environ["UNSLOTH_MOE_ROUTED_KERNEL"] = "0"
    os.environ["UNSLOTH_HD512_ATTENTION"] = "0"
    flash_sliding_mod = sys.modules.get("unsloth_zoo.temporary_patches.gemma4_flash_sliding")
    if flash_sliding_mod is not None:
        enabled_fn = getattr(flash_sliding_mod, "_enabled", None)
        if hasattr(enabled_fn, "cache_clear"):
            enabled_fn.cache_clear()


_enforce_deterministic_dpo_kernels()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import (
    atomic_write_json,
    configure_clean_logging,
    ensure_unsloth_builtins,
    find_latest_valid_checkpoint,
    is_adapter_stage_complete,
    load_gemma4_8gb_model_and_tokenizer,
    verify_moe_lora_integrity,
    verify_moe_peft_version,
    verify_saved_moe_adapter_config,
)

ensure_unsloth_builtins()

try:
    import unsloth  # type: ignore # noqa: F401
except Exception:
    pass

import torch
import torch.nn as nn
import yaml
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from scripts.build_dpo_dataset import build_synthetic_dpo_dry_run_pools
from src.chat_template import make_uniform_message
from src.constants import (
    BASE_MODEL_ID,
    DPO_BETA,
    DPO_LEARNING_RATE,
    DPO_MAX_PROMPT_LENGTH,
    DPO_MAX_SEQ_LENGTH,
    DPO_TRUST_REGION_ALPHA,
    LORA_ALPHA,
    LORA_RANK,
    load_dotenv_if_present,
)
from src.dpo_formatter import mix_dpo_preference_datasets
from src.model_loader import freeze_embeddings_and_towers
from src.vram_hooks import (
    get_vram_telemetry,
    log_vram_telemetry,
    restore_multimodal_towers,
)

load_dotenv_if_present()
logger = logging.getLogger(__name__)


def load_dpo_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load and validate the Stage 2 DPO YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"DPO config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _is_moe_model_cfg(model_cfg: Dict[str, Any]) -> bool:
    arch = str(model_cfg.get("architecture_type", "")).strip().lower()
    base_id = str(model_cfg.get("base_model_id", "")).strip().lower()
    return arch == "moe" or "26b" in base_id or "a4b" in base_id


@torch.no_grad()
def update_trust_region_reference_params(
    policy_params: Dict[str, torch.Tensor],
    ref_state: Dict[str, torch.Tensor],
    tr_alpha: float = DPO_TRUST_REGION_ALPHA,
) -> Dict[str, torch.Tensor]:
    """Pure mathematical EMA helper retained for unit-test compatibility:
    `theta_ref <- tr_alpha * theta_policy + (1.0 - tr_alpha) * theta_ref`.
    """
    if not (0.0 <= tr_alpha <= 1.0):
        raise ValueError(f"tr_alpha must be in [0.0, 1.0], got {tr_alpha}")

    updated: Dict[str, torch.Tensor] = {}
    for name, policy_tensor in policy_params.items():
        pol_cpu = policy_tensor.detach().to(device="cpu", dtype=torch.float32)
        if name not in ref_state:
            updated[name] = pol_cpu.clone()
        else:
            ref_cpu = ref_state[name].detach().to(device="cpu", dtype=torch.float32)
            updated[name] = tr_alpha * pol_cpu + (1.0 - tr_alpha) * ref_cpu
    return updated


class TrustRegionDPOCallback(TrainerCallback):
    """Legacy read-only EMA tracker (never overwrites active policy weights)."""

    def __init__(
        self,
        tr_alpha: float = DPO_TRUST_REGION_ALPHA,
        update_steps: int = 50,
    ) -> None:
        self.tr_alpha = float(tr_alpha)
        self.update_steps = max(1, int(update_steps))
        self.ref_adapter_state: Dict[str, torch.Tensor] = {}
        self.num_updates: int = 0


class DPORefSanityCallback(TrainerCallback):
    """Fail-fast guard verifying DPO reference log-probabilities, early-step loss alignment, and NaN/Inf safety.

    When KL is anchored to the frozen Stage 1 SFT adapter (`precompute_ref_log_probs=True`),
    initial DPO loss at step 0 is `ln(2) = 0.6931` and early logged steps sit near `0.45..0.85`
    (whereas misaligned reference log-probabilities push loss above `1.0..4.0+`).

    Note on Gemma 4 pre-clip `grad_norm` (`huggingface/transformers#45676`):
    `Gemma4TextAttention` uses `self.scaling = 1.0` with QK-RMSNorm at `head_dim = 256/512` and
    TRL `sigmoid` DPO sums log-probabilities across `200..2000+` completion tokens across ~494M
    active LoRA parameters, so healthy pre-clip `grad_norm` routinely sits in `30..10,000+`
    (safely normalized to `<= 1.0` by `clip_grad_norm_(max_grad_norm=1.0)` before `optimizer.step()`).
    """

    def __init__(
        self,
        max_initial_loss: float = 0.98,
        max_diverged_loss: float = 3.50,
        max_initial_grad_norm: float = 100_000.0,
        check_initial_loss: bool = True,
    ) -> None:
        self.max_initial_loss = float(max_initial_loss)
        self.max_diverged_loss = float(max_diverged_loss)
        self.max_initial_grad_norm = float(max_initial_grad_norm)
        self.check_initial_loss = bool(check_initial_loss)
        self.passed_initial_gate = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs or "loss" not in logs:
            return
        loss_val = float(logs["loss"])
        has_grad_norm = logs.get("grad_norm") is not None
        grad_norm_val = float(logs.get("grad_norm", 0.0) or 0.0)
        if not math.isfinite(loss_val) or not math.isfinite(grad_norm_val):
            raise RuntimeError(
                f"DPO NaN/Inf detected at step={state.global_step}: "
                f"loss={loss_val}, grad_norm={grad_norm_val}"
            )
        if loss_val > self.max_diverged_loss:
            raise RuntimeError(
                f"DPO runaway loss divergence detected at step {state.global_step}: "
                f"loss={loss_val:.4f} (> max_diverged_loss={self.max_diverged_loss}), "
                f"pre_clip_grad_norm={grad_norm_val:.4f}."
            )
        if has_grad_norm and grad_norm_val > self.max_initial_grad_norm:
            raise RuntimeError(
                f"DPO gradient sanity check FAILED at step {state.global_step}: "
                f"pre_clip_grad_norm={grad_norm_val:.4f} (> {self.max_initial_grad_norm}), "
                f"loss={loss_val:.4f}."
            )
        if self.check_initial_loss and state.global_step <= max(15, int(getattr(args, "logging_steps", 10)) * 2):
            if not self.passed_initial_gate and loss_val > self.max_initial_loss:
                raise RuntimeError(
                    f"DPO reference logprob sanity gate FAILED at step {state.global_step}: "
                    f"loss={loss_val:.4f} (expected ~0.6931, max={self.max_initial_loss}), "
                    f"pre_clip_grad_norm={grad_norm_val:.4f}. "
                    "Reference log-probabilities are misaligned with Stage 1 SFT policy!"
                )
            if has_grad_norm and state.global_step >= 1 and grad_norm_val <= 0.0:
                raise RuntimeError(
                    f"DPO gradient sanity check FAILED at step {state.global_step}: "
                    f"pre_clip_grad_norm={grad_norm_val:.4f} (expected > 0.0), "
                    f"loss={loss_val:.4f}."
                )
            if not self.passed_initial_gate:
                self.passed_initial_gate = True
                logger.info(
                    "[DPO Sanity Gate PASSED] step=%d | loss=%.4f (~ln2=0.6931) | pre_clip_grad_norm=%.4f (clipped to max_grad_norm=%.1f)",
                    state.global_step,
                    loss_val,
                    grad_norm_val,
                    float(getattr(args, "max_grad_norm", 1.0) or 1.0),
                )


def salt_dpo_dataset_fingerprint(
    dataset: Any,
    base_model_id: str,
    sft_adapter_dir: Optional[str],
    precompute_ref_batch_size: Optional[int],
    is_moe: bool,
    cache_dir: Optional[pathlib.Path] = None,
) -> Any:
    """Salt HuggingFace `datasets` fingerprint so `DPOTrainer` reference logprob `.map()` never
    reuses stale Arrow cache files across different Stage 1 SFT adapters, batch sizes, or MoE kernel flags,
    while persisting verified reference log-probabilities under `cache_dir` when provided.
    """
    adapter_sig = "none"
    if sft_adapter_dir:
        st_file = pathlib.Path(sft_adapter_dir) / "adapter_model.safetensors"
        cfg_file = pathlib.Path(sft_adapter_dir) / "adapter_config.json"
        if st_file.is_file():
            stat = st_file.stat()
            adapter_sig = f"{st_file.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        elif cfg_file.is_file():
            stat = cfg_file.stat()
            adapter_sig = f"{cfg_file.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        else:
            adapter_sig = str(sft_adapter_dir)

    payload = json.dumps(
        {
            "schema": "v6_bitexact_dpo_ref_alignment",
            "base_model_id": str(base_model_id),
            "adapter_sig": adapter_sig,
            "precompute_ref_batch_size": precompute_ref_batch_size,
            "is_moe": bool(is_moe),
            "compile_disable": os.environ.get("UNSLOTH_COMPILE_DISABLE", "0"),
            "fused_linear": os.environ.get("UNSLOTH_FUSED_LINEAR", "1"),
            "gemma4_flash_sliding": os.environ.get("UNSLOTH_GEMMA4_FLASH_SLIDING", "1"),
            "block_compile": os.environ.get("UNSLOTH_BLOCK_COMPILE", "0"),
            "moe_lora_fold": os.environ.get("UNSLOTH_MOE_LORA_FOLD", "0"),
            "moe_gc_keep_base": os.environ.get("UNSLOTH_MOE_GC_KEEP_BASE", "0"),
            "moe_routed_kernel": os.environ.get("UNSLOTH_MOE_ROUTED_KERNEL", "0"),
            "hd512_attention": os.environ.get("UNSLOTH_HD512_ATTENTION", "0"),
            "rows": len(dataset),
        },
        sort_keys=True,
    )
    fp = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    if cache_dir is not None:
        target_dir = pathlib.Path(cache_dir) / f"{'moe' if is_moe else 'dense'}_{fp[:16]}"
        target_dir.mkdir(parents=True, exist_ok=True)
        cache_arrow = target_dir / "salted_dpo_pairs.arrow"
        return dataset.map(
            lambda batch: batch,
            batched=True,
            new_fingerprint=fp,
            cache_file_name=str(cache_arrow),
            load_from_cache_file=True,
            keep_in_memory=False,
            desc="Salting DPO dataset fingerprint",
        )
    return dataset.map(
        lambda batch: batch,
        batched=True,
        new_fingerprint=fp,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="Salting DPO dataset fingerprint",
    )


def _maybe_inject_gemma4_mm_token_type_ids(model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Match Unsloth `_unsloth_pre_compute_loss` (`unsloth/models/_utils.py:6072-6083`) so
    `DPOTrainer.compute_ref_log_probs` constructs the exact same Gemma 4 causal/sliding masks as `compute_loss`.
    """
    if not isinstance(inputs, dict) or "mm_token_type_ids" in inputs or "input_ids" not in inputs:
        return inputs
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return inputs
    inner = model
    for attr in ("base_model", "model", "model"):
        inner = getattr(inner, attr, inner)
    if getattr(getattr(inner, "config", None), "model_type", "") == "gemma4":
        mod = sys.modules.get(type(inner).__module__)
        if mod is not None and hasattr(mod, "create_causal_mask_mapping"):
            patched = dict(inputs)
            patched["mm_token_type_ids"] = torch.zeros_like(input_ids)
            return patched
    return inputs


def _enter_unsloth_training_mode(
    model: Any,
    trainer: Any = None,
    freeze_moe_router: bool = True,
) -> None:
    """Restore full Unsloth training state (`for_training`) after `UnslothDPOTrainer.__init__`'s
    trailing `model.for_inference()` call (`UnslothDPOTrainer.py:3235`), ensuring `gradient_checkpointing`,
    `use_cache=False`, compiler stance, and router/tower freeze flags match `trainer.train()` verbatim.
    """
    args = getattr(trainer, "args", None)
    use_gc = getattr(
        model,
        "_unsloth_gradient_checkpointing",
        getattr(args, "gradient_checkpointing", True) if args is not None else True,
    )
    if hasattr(model, "for_training") and callable(model.for_training):
        try:
            model.for_training(use_gradient_checkpointing=use_gc)
        except TypeError:
            model.for_training()
    else:
        model.train()
    freeze_embeddings_and_towers(model, freeze_moe_router=freeze_moe_router)


@contextlib.contextmanager
def _dpo_compute_loss_autocast_context(trainer: Any):
    """Enter the exact mixed-precision autocast context that `Trainer.training_step` /
    `DPOTrainer.compute_ref_log_probs` use during real forward passes.
    """
    ctx_fn = getattr(trainer, "compute_loss_context_manager", None)
    if callable(ctx_fn):
        with ctx_fn():
            yield
        return
    accel = getattr(trainer, "accelerator", None)
    if accel is not None and hasattr(accel, "autocast") and callable(accel.autocast):
        with accel.autocast():
            yield
        return
    args = getattr(trainer, "args", None)
    if bool(getattr(args, "bf16", False)) and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    yield


@contextlib.contextmanager
def _temporarily_disable_gradient_checkpointing(model: Any, trainer: Any = None):
    """Mirror `DPOTrainer.compute_ref_log_probs`'s `disable_gradient_checkpointing(...)` context."""
    args = getattr(trainer, "args", None)
    gc_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) if args is not None else None
    try:
        from trl.models.utils import disable_gradient_checkpointing  # type: ignore

        with disable_gradient_checkpointing(model, gc_kwargs):
            yield
        return
    except Exception:
        pass
    flagged = [m for m in model.modules() if getattr(m, "gradient_checkpointing", False) is True]
    for m in flagged:
        m.gradient_checkpointing = False
    try:
        yield
    finally:
        for m in flagged:
            m.gradient_checkpointing = True


def verify_step0_dpo_ref_alignment(
    trainer: Any,
    model: Any,
    is_moe: bool = False,
    freeze_moe_router: bool = True,
    max_loss_drift_from_ln2: float = 0.05,
    max_eval_train_drift: float = 0.02,
    max_step0_grad_norm: float = 100_000.0,
) -> Dict[str, float]:
    """Run a `<5s` pre-flight check before `trainer.train()` verifying exact Step-0 alignment
    between precomputed `ref_*_logps`, the `no_grad` reference-mode forward, the exact `for_training`
    + `autocast` `training_step` forward, and backward gradients through 2D + 3D MoE LoRA weights.
    """
    train_ds = trainer.train_dataset
    col_names = set(getattr(train_ds, "column_names", []) or [])
    if bool(getattr(trainer.args, "precompute_ref_log_probs", False)):
        if "ref_chosen_logps" not in col_names or "ref_rejected_logps" not in col_names:
            raise RuntimeError(
                f"DPO precompute_ref_log_probs=True failed to populate ref_*_logps columns! Columns={sorted(col_names)}"
            )
        inspect_n = min(16, len(train_ds))
        for i in range(inspect_n):
            row = train_ds[i]
            rc = float(row["ref_chosen_logps"])
            rr = float(row["ref_rejected_logps"])
            if not math.isfinite(rc) or not math.isfinite(rr) or rc >= 0.0 or rr >= 0.0:
                raise RuntimeError(
                    f"Invalid precomputed reference log-probabilities at row {i}: "
                    f"ref_chosen_logps={rc}, ref_rejected_logps={rr}"
                )
            if "chosen_input_ids" in row and "rejected_input_ids" in row:
                c_len = max(1, len(row["chosen_input_ids"]))
                r_len = max(1, len(row["rejected_input_ids"]))
                if (-rc / c_len) > 12.0 or (-rr / r_len) > 12.0:
                    raise RuntimeError(
                        f"Pathological per-token reference NLL at row {i}: "
                        f"chosen_nll={-rc / c_len:.3f}, rejected_nll={-rr / r_len:.3f}"
                    )

    bsz = min(len(train_ds), max(1, int(getattr(trainer.args, "per_device_train_batch_size", 1))))
    raw_items = [train_ds[i] for i in range(bsz)]
    collated = trainer.data_collator(raw_items)

    def _clone_batch(batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in batch_dict.items()
        }

    batch_eval = trainer._prepare_inputs(_clone_batch(collated))
    batch_train = trainer._prepare_inputs(_clone_batch(collated))

    try:
        # 1. Undo `UnslothDPOTrainer.__init__`'s trailing `model.for_inference()` and verify the
        #    no-grad + disabled-GC + autocast forward (identical to `DPOTrainer.compute_ref_log_probs`).
        _enter_unsloth_training_mode(model, trainer=trainer, freeze_moe_router=freeze_moe_router)
        with (
            torch.no_grad(),
            _temporarily_disable_gradient_checkpointing(model, trainer=trainer),
            _dpo_compute_loss_autocast_context(trainer),
        ):
            loss_nograd_t = trainer.compute_loss(model, batch_eval)
            loss_nograd = float(loss_nograd_t.detach().float().item())

        # 2. Verify the exact `trainer.train()` Step-0 forward + backward (`for_training` + GC + autocast).
        _enter_unsloth_training_mode(model, trainer=trainer, freeze_moe_router=freeze_moe_router)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), _dpo_compute_loss_autocast_context(trainer):
            loss_grad_t = trainer.compute_loss(model, batch_train)
            loss_grad = float(loss_grad_t.detach().float().item())
            loss_grad_t.backward()

        total_sq = 0.0
        expert_sq = 0.0
        for pname, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gnorm = float(param.grad.detach().float().norm(2).item())
                if not math.isfinite(gnorm):
                    raise RuntimeError(f"Non-finite Step-0 gradient on '{pname}': {gnorm}")
                total_sq += gnorm * gnorm
                if "experts" in pname and "lora_" in pname:
                    expert_sq += gnorm * gnorm
        step0_grad_norm = math.sqrt(total_sq)
        expert_grad_norm = math.sqrt(expert_sq)
    finally:
        model.zero_grad(set_to_none=True)
        _enter_unsloth_training_mode(model, trainer=trainer, freeze_moe_router=freeze_moe_router)
        metrics_dict = getattr(trainer, "_metrics", None)
        if isinstance(metrics_dict, dict):
            for sub in metrics_dict.values():
                if hasattr(sub, "clear"):
                    sub.clear()
        if hasattr(trainer, "_total_train_tokens"):
            trainer._total_train_tokens = 0

    ln2 = math.log(2.0)
    if (
        not math.isfinite(loss_nograd)
        or not math.isfinite(loss_grad)
        or not math.isfinite(step0_grad_norm)
    ):
        raise RuntimeError(
            f"DPO Step-0 check produced non-finite values: loss_eval={loss_nograd}, "
            f"loss_train={loss_grad}, grad_norm={step0_grad_norm}"
        )
    if (
        abs(loss_nograd - ln2) > max_loss_drift_from_ln2
        or abs(loss_grad - ln2) > max_loss_drift_from_ln2
        or abs(loss_grad - loss_nograd) > max_eval_train_drift
    ):
        raise RuntimeError(
            f"DPO Step-0 reference alignment FAILED BEFORE training! "
            f"loss_eval={loss_nograd:.6f}, loss_train={loss_grad:.6f}, ln(2)={ln2:.6f}, "
            f"|train-eval|={abs(loss_grad - loss_nograd):.6f}. "
            "Policy forward pass does not match precomputed Stage 1 SFT reference log-probabilities!"
        )
    if step0_grad_norm > max_step0_grad_norm:
        raise RuntimeError(
            f"DPO Step-0 gradient norm explosion ({step0_grad_norm:.4f} > {max_step0_grad_norm}) before step 1!"
        )
    if is_moe and step0_grad_norm > 1e-6 and expert_grad_norm <= 0.0:
        raise RuntimeError(
            "DPO Step-0 MoE gradient check FAILED: zero gradient norm on 3D `experts` LoRA parameters!"
        )

    summary = {
        "loss_eval": round(loss_nograd, 6),
        "loss_train": round(loss_grad, 6),
        "eval_train_abs_diff": round(abs(loss_grad - loss_nograd), 6),
        "step0_grad_norm": round(step0_grad_norm, 4),
        "expert_grad_norm": round(expert_grad_norm, 4),
    }
    logger.info(
        "[DPO Step-0 Alignment Verified] loss_eval=%.6f | loss_train=%.6f | |diff|=%.6f (ln2=%.6f) | step0_grad_norm=%.4f | expert_grad_norm=%.4f",
        loss_nograd,
        loss_grad,
        abs(loss_grad - loss_nograd),
        ln2,
        step0_grad_norm,
        expert_grad_norm,
    )
    return summary


@contextlib.contextmanager
def preserve_stage1_adapter_for_ref_precompute(
    model: Any,
    trainer_box: Dict[str, Any],
    dpo_trainer_cls: Any = None,
):
    """Keep the untouched Stage 1 SFT `"default"` adapter active during `precompute_ref_log_probs=True`
    without duplicating `"ref"` LoRA adapters (which corrupt nested 3D `ParamWrapper` layers when
    `target_parameters=['experts.gate_up_proj', 'experts.down_proj']` on `Gemma-4-26B-A4B`), without
    allowing `use_adapter(model, None)` / `model.disable_adapter()` to revert $\\pi_{\\text{ref}}$ to the
    untuned base model, and while ensuring `compute_ref_log_probs` passes identical Gemma 4
    `mm_token_type_ids` masks as `DPOTrainer.compute_loss`.
    """
    orig_disable = getattr(model, "disable_adapter", None)
    orig_add_adapter = getattr(model, "add_adapter", None)
    orig_get_parameter = getattr(model, "get_parameter", None)
    orig_compute_ref = (
        getattr(dpo_trainer_cls, "compute_ref_log_probs", None)
        if dpo_trainer_cls is not None
        else None
    )

    @contextlib.contextmanager
    def _guarded_disable_adapter():
        tr = trainer_box.get("trainer")
        step = int(getattr(getattr(tr, "state", None), "global_step", 0) or 0)
        if step == 0:
            # Keep Stage 1 SFT "default" adapter active during reference logprob precomputation
            yield
        elif orig_disable is not None:
            with orig_disable():
                yield
        else:
            yield

    def _guarded_add_adapter(adapter_name: str, peft_config: Any, *args: Any, **kwargs: Any) -> Any:
        if adapter_name == "ref":
            logger.info(
                "Skipping duplicate 'ref' PEFT adapter allocation during precompute_ref_log_probs=True "
                "(computing exact Stage 1 SFT reference log-probs directly from active 'default' adapter)."
            )
            return None
        if orig_add_adapter is not None:
            return orig_add_adapter(adapter_name, peft_config, *args, **kwargs)
        return None

    def _guarded_get_parameter(target: str) -> Any:
        if orig_get_parameter is None:
            raise AttributeError("Model does not implement get_parameter")
        try:
            return orig_get_parameter(target)
        except AttributeError:
            if ".ref." in target:
                return orig_get_parameter(target.replace(".ref.", ".default."))
            raise

    def _aligned_compute_ref_log_probs(self_trainer: Any, ref_or_policy_model: Any, inputs: Dict[str, Any]):
        aligned_inputs = _maybe_inject_gemma4_mm_token_type_ids(ref_or_policy_model, inputs)
        return orig_compute_ref(self_trainer, ref_or_policy_model, aligned_inputs)

    if orig_disable is not None:
        model.disable_adapter = _guarded_disable_adapter
    if orig_add_adapter is not None:
        model.add_adapter = _guarded_add_adapter
    if orig_get_parameter is not None:
        model.get_parameter = _guarded_get_parameter
    if orig_compute_ref is not None and dpo_trainer_cls is not None:
        dpo_trainer_cls.compute_ref_log_probs = _aligned_compute_ref_log_probs
    try:
        yield
    finally:
        if orig_disable is not None:
            model.disable_adapter = orig_disable
        if orig_add_adapter is not None:
            model.add_adapter = orig_add_adapter
        if orig_get_parameter is not None:
            model.get_parameter = orig_get_parameter
        if orig_compute_ref is not None and dpo_trainer_cls is not None:
            dpo_trainer_cls.compute_ref_log_probs = orig_compute_ref


def align_26b_chat_template_for_dpo(chat_template: Optional[str]) -> Optional[str]:
    """Align `26B-A4B-it` Jinja template during DPO tokenization so `enable_thinking=False`
    completed turns emit `<|channel>thought\\n<channel|>` identically to `add_generation_prompt=True`.
    """
    if not chat_template:
        return chat_template
    target_snippet = (
        "{%- if thinking_text and thinking_gate -%}\n"
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        "    {%- endif -%}"
    )
    replacement_snippet = (
        "{%- if thinking_text and thinking_gate -%}\n"
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        "    {%- elif role == 'model' and not enable_thinking and loop.index0 > ns_turn.last_user_idx -%}\n"
        "        {{- '<|channel>thought\\n<channel|>' -}}\n"
        "    {%- endif -%}"
    )
    if target_snippet in chat_template and "elif role == 'model' and not enable_thinking" not in chat_template:
        return chat_template.replace(target_snippet, replacement_snippet, 1)
    return chat_template


def prepare_conversational_dpo_dataset(dataset: Any) -> Any:
    """Convert normalized DPO records into pure TRL conversational `{prompt, chosen, rejected, chat_template_kwargs}`."""
    from datasets import Dataset

    rows: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        r = dataset[idx]
        p_msgs = r.get("prompt_messages")
        c_only = r.get("chosen_only_messages")
        r_only = r.get("rejected_only_messages")
        enable_thinking = bool(r.get("enable_thinking", False))

        if not (isinstance(p_msgs, list) and isinstance(c_only, list) and isinstance(r_only, list)):
            # Reconstruct from `chosen_messages` / `rejected_messages` if needed
            c_full = r.get("chosen_messages") or []
            r_full = r.get("rejected_messages") or []
            if len(c_full) >= 2 and len(r_full) >= 2:
                p_msgs = [make_uniform_message(m["role"], m["content"]) for m in c_full[:-1]]
                c_only = [make_uniform_message("assistant", c_full[-1]["content"], c_full[-1].get("reasoning_content", ""))]
                r_only = [make_uniform_message("assistant", r_full[-1]["content"], r_full[-1].get("reasoning_content", ""))]
            else:
                continue

        rows.append(
            {
                "prompt": p_msgs,
                "chosen": c_only,
                "rejected": r_only,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
        )

    if not rows:
        raise RuntimeError("prepare_conversational_dpo_dataset produced 0 valid preference pairs.")
    return Dataset.from_list(rows)


def verify_dpo_prefix_alignment(sample: Dict[str, Any], tokenizer: Any) -> None:
    """Assert `prompt_ids` is an exact token prefix of both `prompt + chosen` and `prompt + rejected`."""
    tpl_kwargs = dict(sample.get("chat_template_kwargs") or {})
    prompt_str = tokenizer.apply_chat_template(
        sample["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        **tpl_kwargs,
    )
    chosen_str = tokenizer.apply_chat_template(
        sample["prompt"] + sample["chosen"],
        tokenize=False,
        add_generation_prompt=False,
        **tpl_kwargs,
    )
    rejected_str = tokenizer.apply_chat_template(
        sample["prompt"] + sample["rejected"],
        tokenize=False,
        add_generation_prompt=False,
        **tpl_kwargs,
    )
    p_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    c_ids = tokenizer.encode(chosen_str, add_special_tokens=False)
    r_ids = tokenizer.encode(rejected_str, add_special_tokens=False)

    if c_ids[: len(p_ids)] != p_ids or r_ids[: len(p_ids)] != p_ids:
        raise RuntimeError(
            "DPO prompt-prefix invariant violated! `prompt_ids` is NOT an exact prefix of "
            f"`prompt + chosen` / `prompt + rejected`.\nprompt_tail={prompt_str[-80:]!r}\nchosen_head={chosen_str[:160]!r}"
        )
    if len(c_ids) <= len(p_ids) or len(r_ids) <= len(p_ids):
        raise RuntimeError("DPO completion token length is empty after prompt prefix!")


def _cleanup_dpo_output_dir(output_dir: pathlib.Path) -> None:
    """Remove duplicate `ref/` adapter subfolder and intermediate `checkpoint-*` folders after verified save."""
    if not is_adapter_stage_complete(output_dir, "dpo_metrics.json"):
        return
    ref_subdir = output_dir / "ref"
    if ref_subdir.is_dir():
        shutil.rmtree(ref_subdir, ignore_errors=True)
        logger.info("Removed temporary DPO reference adapter copy at %s.", ref_subdir)
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            shutil.rmtree(child, ignore_errors=True)


def install_dpo_default_adapter_resume_hook(
    trainer: Any,
    freeze_moe_router: bool = True,
) -> None:
    """Fix `transformers>=5.x` multi-adapter PEFT checkpoint resume when `checkpoint-N/ref/` exists.

    In `DPOTrainer` (`ref_model=None`), `PeftModel.save_pretrained` writes `"default"` at
    `checkpoint-N/adapter_model.safetensors` and `"ref"` inside `checkpoint-N/ref/`. On resume,
    `transformers.Trainer._load_from_checkpoint` sees `adapter_subdirs == ["ref"]` and skips the
    `else:` branch that loads the root `"default"` policy weights. This hook wraps
    `trainer._load_from_checkpoint` so `"default"` policy weights, active adapter selection, and
    grad flags are deterministically restored before optimizer state loading.
    """
    orig_load = getattr(trainer, "_load_from_checkpoint", None)
    if orig_load is None:
        return

    def _wrapped_load_from_checkpoint(resume_from_checkpoint: Any, *args: Any, **kwargs: Any) -> Any:
        result = orig_load(resume_from_checkpoint, *args, **kwargs)
        if not resume_from_checkpoint:
            return result
        ckpt_path = pathlib.Path(str(resume_from_checkpoint))
        target_model = kwargs.get("model")
        if target_model is None and args:
            target_model = args[0]
        if target_model is None:
            target_model = getattr(trainer, "model", None)
        if (
            target_model is not None
            and (ckpt_path / "adapter_config.json").is_file()
            and hasattr(target_model, "peft_config")
            and "default" in getattr(target_model, "peft_config", {})
        ):
            from peft.utils import load_peft_weights, set_peft_model_state_dict

            default_weights = load_peft_weights(str(ckpt_path), device="cpu")
            if not default_weights:
                raise RuntimeError(
                    f"DPO checkpoint resume failed: root policy adapter at {ckpt_path} loaded 0 tensors!"
                )
            set_peft_model_state_dict(
                target_model,
                default_weights,
                adapter_name="default",
            )
            if hasattr(target_model, "set_adapter"):
                target_model.set_adapter("default")
            for pname, param in target_model.named_parameters():
                if ".ref." in pname:
                    param.requires_grad_(False)
                elif ".default." in pname and "lora_" in pname:
                    param.requires_grad_(True)
            freeze_embeddings_and_towers(target_model, freeze_moe_router=freeze_moe_router)
            logger.info(
                "Restored trained 'default' DPO policy adapter (%d tensors) from %s.",
                len(default_weights),
                ckpt_path,
            )
        return result

    trainer._load_from_checkpoint = _wrapped_load_from_checkpoint


def build_dpo_training_arguments(
    cfg: Dict[str, Any],
    output_dir: str,
    smoke_test: bool = False,
    max_steps_override: Optional[int] = None,
    has_eval_dataset: bool = False,
) -> Any:
    """Build TRL `DPOConfig` (or `TrainingArguments` fallback) from YAML configuration."""
    train_cfg = cfg.get("training", {})
    dpo_cfg = cfg.get("dpo", {})
    vram_cfg = cfg.get("vram_optimization", {})
    lora_cfg = cfg.get("lora", {})
    use_gc = bool(lora_cfg.get("use_gradient_checkpointing", True))

    max_steps = (
        max_steps_override
        if max_steps_override is not None
        else (2 if smoke_test else int(train_cfg.get("max_steps", -1)))
    )

    use_bf16 = bool(train_cfg.get("bf16", True)) and torch.cuda.is_available()
    use_tf32 = bool(train_cfg.get("tf32", True)) and torch.cuda.is_available()
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    optim_name = str(
        train_cfg.get("optim", "adamw_bnb_8bit")
        if torch.cuda.is_available()
        else "adamw_torch"
    )

    eval_strategy = (
        str(train_cfg.get("eval_strategy", "no"))
        if (has_eval_dataset and not smoke_test)
        else "no"
    )

    common_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": (
            1 if smoke_test else int(train_cfg.get("gradient_accumulation_steps", 8))
        ),
        "optim": optim_name,
        "learning_rate": float(train_cfg.get("learning_rate", DPO_LEARNING_RATE)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1.0)),
        "max_steps": max_steps,
        "bf16": use_bf16,
        "tf32": use_tf32,
        "fp16": False,
        "gradient_checkpointing": use_gc,
        "dataloader_pin_memory": bool(train_cfg.get("dataloader_pin_memory", True)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "logging_first_step": True,
        "logging_steps": 1 if smoke_test else int(train_cfg.get("logging_steps", 10)),
        "eval_strategy": eval_strategy,
        "save_strategy": "no" if smoke_test else str(train_cfg.get("save_strategy", "steps")),
        "save_steps": int(train_cfg.get("save_steps", 200)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 1)),
        "seed": int(train_cfg.get("seed", 3407)),
        "log_level": "warning",
        "report_to": "none",
    }
    if eval_strategy != "no":
        common_kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 200))

    warmup_ratio_val = float(train_cfg.get("warmup_ratio", 0.10))

    try:
        from trl import DPOConfig

        sig = inspect.signature(DPOConfig.__init__)
        if "warmup_ratio" in sig.parameters:
            common_kwargs["warmup_ratio"] = warmup_ratio_val
        elif "warmup_steps" in sig.parameters:
            common_kwargs["warmup_steps"] = warmup_ratio_val
        if "beta" in sig.parameters:
            common_kwargs["beta"] = float(dpo_cfg.get("beta", DPO_BETA))
        if "loss_type" in sig.parameters:
            common_kwargs["loss_type"] = str(dpo_cfg.get("loss_type", "sigmoid"))
        if "max_length" in sig.parameters:
            common_kwargs["max_length"] = int(
                dpo_cfg.get("max_length", DPO_MAX_SEQ_LENGTH)
            )
        if "max_prompt_length" in sig.parameters:
            common_kwargs["max_prompt_length"] = int(
                dpo_cfg.get("max_prompt_length", DPO_MAX_PROMPT_LENGTH)
            )
        if "max_completion_length" in sig.parameters and dpo_cfg.get("max_completion_length") is not None:
            common_kwargs["max_completion_length"] = int(dpo_cfg["max_completion_length"])
        if "precompute_ref_log_probs" in sig.parameters:
            common_kwargs["precompute_ref_log_probs"] = bool(
                vram_cfg.get("precompute_ref_log_probs", True)
            )
        if "precompute_ref_batch_size" in sig.parameters and vram_cfg.get("precompute_ref_batch_size") is not None:
            common_kwargs["precompute_ref_batch_size"] = int(vram_cfg["precompute_ref_batch_size"])
        if "router_aux_loss_coef" in sig.parameters and bool(vram_cfg.get("freeze_moe_router", True)):
            common_kwargs["router_aux_loss_coef"] = 0.0
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            common_kwargs = {k: v for k, v in common_kwargs.items() if k in sig.parameters}
        return DPOConfig(**common_kwargs)
    except ImportError:
        ta_sig = inspect.signature(TrainingArguments.__init__)
        if "warmup_ratio" in ta_sig.parameters:
            common_kwargs["warmup_ratio"] = warmup_ratio_val
        elif "warmup_steps" in ta_sig.parameters:
            common_kwargs["warmup_steps"] = warmup_ratio_val
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ta_sig.parameters.values()):
            common_kwargs = {k: v for k, v in common_kwargs.items() if k in ta_sig.parameters}
        return TrainingArguments(**common_kwargs)


def load_or_build_dpo_dataset(
    dataset_dir: pathlib.Path,
    smoke_test: bool = False,
) -> Any:
    """Load the prepared Stage 2 DPO dataset from disk or synthesize smoke-test pairs."""
    from datasets import Dataset, load_from_disk

    if dataset_dir.exists() and not smoke_test:
        logger.info("Loading Stage 2 DPO dataset from disk: %s", dataset_dir)
        return load_from_disk(str(dataset_dir))

    if dataset_dir.exists() and smoke_test:
        ds = load_from_disk(str(dataset_dir))
        return ds.select(range(min(len(ds), 5)))

    if not smoke_test:
        raise FileNotFoundError(
            f"Prepared DPO dataset not found at {dataset_dir}. "
            "Run `python scripts/build_dpo_dataset.py --pull-from-hub` first or pass `--smoke-test`."
        )

    logger.info("Building in-memory preference dataset for --smoke-test mode.")
    pools = build_synthetic_dpo_dry_run_pools()
    mixed_pairs, _ = mix_dpo_preference_datasets(
        saudi_dpo_samples=pools["saudi"],
        arcee_dpo_samples=pools["arcee"],
        english_dpo_samples=pools.get("english", []),
        arabic_thinking_samples=pools.get("arabic_thinking", []),
        english_thinking_samples=pools.get("english_thinking", []),
    )
    return Dataset.from_list(mixed_pairs)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Stage 2 DPO training."""
    parser = argparse.ArgumentParser(
        description="Stage 2 Direct Preference Optimization (DPO) alignment pipeline for Gemma-4-Arabic."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "dpo_config.yaml"),
        help="Path to Stage 2 DPO YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the built Stage 2 DPO dataset directory.",
    )
    parser.add_argument(
        "--sft-adapter-dir",
        type=str,
        default=None,
        help="Path to Stage 1 SFT adapter checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save Stage 2 DPO checkpoints.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 2-step (max_steps=2) smoke test to verify DPO configuration.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional explicit override for max training steps.",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Validate config, arguments, and conversational preference dataset without loading weights.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for Stage 2 DPO training."""
    _enforce_deterministic_dpo_kernels()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    configure_clean_logging()
    args = parse_args(argv)

    cfg = load_dpo_config(pathlib.Path(args.config))
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    dpo_cfg = cfg.get("dpo", {})
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("datasets", {})
    vram_cfg = cfg.get("vram_optimization", {})

    output_dir = str(args.output_dir or train_cfg.get("output_dir", "outputs/dpo_stage2"))
    dataset_dir = pathlib.Path(
        args.dataset_dir or ds_cfg.get("output_dataset_dir", "data/dpo_mixed")
    )
    sft_adapter_dir = str(
        args.sft_adapter_dir or model_cfg.get("sft_adapter_dir", "outputs/sft_stage1")
    )
    is_moe = _is_moe_model_cfg(model_cfg)
    tr_alpha = float(dpo_cfg.get("tr_alpha", DPO_TRUST_REGION_ALPHA))

    raw_dataset = load_or_build_dpo_dataset(
        dataset_dir=dataset_dir,
        smoke_test=args.smoke_test or args.dry_run_only,
    )
    full_dataset = prepare_conversational_dpo_dataset(raw_dataset)

    eval_dataset = None
    train_dataset = full_dataset
    if (
        str(train_cfg.get("eval_strategy", "no")) != "no"
        and not args.smoke_test
        and not args.dry_run_only
        and len(full_dataset) >= 50
    ):
        split = full_dataset.train_test_split(
            test_size=0.02, seed=int(train_cfg.get("seed", 3407))
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    dpo_args = build_dpo_training_arguments(
        cfg=cfg,
        output_dir=output_dir,
        smoke_test=args.smoke_test,
        max_steps_override=args.max_steps,
        has_eval_dataset=(eval_dataset is not None),
    )
    configure_clean_logging()

    if args.dry_run_only:
        ema_check = update_trust_region_reference_params(
            policy_params={"lora_A": torch.tensor([10.0])},
            ref_state={"lora_A": torch.tensor([0.0])},
            tr_alpha=tr_alpha,
        )
        summary = {
            "status": "dry_run_verified",
            "base_model_id": model_cfg.get("base_model_id", BASE_MODEL_ID),
            "is_moe": is_moe,
            "sft_adapter_dir": sft_adapter_dir,
            "beta": float(dpo_cfg.get("beta", DPO_BETA)),
            "tr_alpha": tr_alpha,
            "ema_test_value": round(float(ema_check["lora_A"].item()), 4),
            "learning_rate": float(train_cfg.get("learning_rate", DPO_LEARNING_RATE)),
            "max_length": int(dpo_cfg.get("max_length", DPO_MAX_SEQ_LENGTH)),
            "offload_ple_to_cpu": bool(vram_cfg.get("offload_ple_to_cpu", True)),
            "offload_multimodal_towers_to_cpu": bool(
                vram_cfg.get("offload_multimodal_towers_to_cpu", True)
            ),
            "freeze_moe_router": bool(vram_cfg.get("freeze_moe_router", True)),
            "max_steps": dpo_args.max_steps,
            "dataset_pairs": len(train_dataset),
            "output_dir": output_dir,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    out_dir_path = pathlib.Path(output_dir)
    if not args.smoke_test and is_adapter_stage_complete(
        out_dir_path, "dpo_metrics.json", cleanup_partial_root=True
    ):
        logger.info("Stage 2 DPO final adapter already verified at %s; skipping training.", output_dir)
        return 0

    if not args.smoke_test and not (pathlib.Path(sft_adapter_dir) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Stage 1 SFT adapter not found at '{sft_adapter_dir}'! Stage 2 DPO requires the trained Stage 1 SFT adapter."
        )

    if is_moe and not args.smoke_test and torch.cuda.is_available():
        verify_moe_peft_version()

    try:
        from unsloth import PatchDPOTrainer  # type: ignore

        PatchDPOTrainer()
    except Exception:
        pass

    from trl import DPOTrainer

    adapter_ckpt = sft_adapter_dir if pathlib.Path(sft_adapter_dir).exists() else None
    model, tokenizer, offload_state = load_gemma4_8gb_model_and_tokenizer(
        model_name=str(model_cfg.get("base_model_id", BASE_MODEL_ID)),
        max_seq_length=int(model_cfg.get("max_seq_length", DPO_MAX_SEQ_LENGTH)),
        dtype=str(model_cfg.get("dtype", "bfloat16")),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        attach_lora=(adapter_ckpt is None),
        lora_r=int(lora_cfg.get("r", LORA_RANK)),
        lora_alpha=int(lora_cfg.get("lora_alpha", LORA_ALPHA)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        use_rslora=bool(lora_cfg.get("use_rslora", False)),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=int(lora_cfg.get("random_state", 3407)),
        adapter_checkpoint=adapter_ckpt,
        is_trainable=True,
        target_modules=lora_cfg.get("target_modules"),
        exclude_modules=lora_cfg.get("exclude_modules"),
        offload_ple_to_cpu=bool(vram_cfg.get("offload_ple_to_cpu", True)),
        offload_multimodal_towers_to_cpu=bool(
            vram_cfg.get("offload_multimodal_towers_to_cpu", True)
        ),
        freeze_moe_router=bool(vram_cfg.get("freeze_moe_router", True)),
    )
    verify_moe_lora_integrity(
        model,
        is_moe=is_moe,
        smoke_test=args.smoke_test,
        expect_trained_adapter=(not args.smoke_test and adapter_ckpt is not None),
        adapter_dir=adapter_ckpt,
        require_frozen_router=bool(vram_cfg.get("freeze_moe_router", True)),
    )

    base_model_id_str = str(model_cfg.get("base_model_id", BASE_MODEL_ID))
    precomp_bs = getattr(dpo_args, "precompute_ref_batch_size", None)
    ref_cache_root = (
        None
        if args.smoke_test
        else (out_dir_path.resolve().parent / ".dpo_ref_cache")
    )
    train_dataset = salt_dpo_dataset_fingerprint(
        train_dataset,
        base_model_id=base_model_id_str,
        sft_adapter_dir=adapter_ckpt,
        precompute_ref_batch_size=precomp_bs,
        is_moe=is_moe,
        cache_dir=ref_cache_root,
    )
    if eval_dataset is not None:
        eval_dataset = salt_dpo_dataset_fingerprint(
            eval_dataset,
            base_model_id=base_model_id_str,
            sft_adapter_dir=adapter_ckpt,
            precompute_ref_batch_size=precomp_bs,
            is_moe=is_moe,
            cache_dir=(ref_cache_root / "eval") if ref_cache_root is not None else None,
        )

    saved_gemma4_proc = getattr(tokenizer, "_gemma4_processor", None)
    original_chat_template = getattr(tokenizer, "chat_template", None)
    if is_moe and original_chat_template:
        tokenizer.chat_template = align_26b_chat_template_for_dpo(original_chat_template)
        if saved_gemma4_proc is not None and hasattr(saved_gemma4_proc, "chat_template"):
            saved_gemma4_proc.chat_template = tokenizer.chat_template

    # Hard pre-train verification that `prompt_ids` is an exact prefix of `prompt + chosen/rejected`
    # across BOTH non-thinking (`enable_thinking=False`) and thinking (`enable_thinking=True`) pairs.
    checked_modes = set()
    for check_idx in range(min(len(train_dataset), 200)):
        row_sample = train_dataset[check_idx]
        mode_flag = bool((row_sample.get("chat_template_kwargs") or {}).get("enable_thinking", False))
        if mode_flag not in checked_modes:
            verify_dpo_prefix_alignment(row_sample, tokenizer)
            checked_modes.add(mode_flag)
            if len(checked_modes) >= 2:
                break

    last_ckpt = None
    if not args.smoke_test and out_dir_path.exists():
        last_ckpt = find_latest_valid_checkpoint(out_dir_path)
        if last_ckpt is not None:
            logger.info("Resuming Stage 2 DPO from latest verified checkpoint: %s", last_ckpt)

    dpo_callbacks: List[TrainerCallback] = []
    if not args.smoke_test:
        dpo_callbacks.append(
            DPORefSanityCallback(
                max_initial_loss=0.98,
                max_initial_grad_norm=100_000.0,
                check_initial_loss=(last_ckpt is None),
            )
        )

    # Detach non-hashable VLM processor attachment during `DPOTrainer._prepare_dataset` so
    # HuggingFace `datasets` `Hasher` produces an identical fingerprint across restarts/resumes.
    if hasattr(tokenizer, "_gemma4_processor"):
        try:
            delattr(tokenizer, "_gemma4_processor")
        except AttributeError:
            pass

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
    }
    if dpo_callbacks:
        trainer_kwargs["callbacks"] = dpo_callbacks
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer_sig = inspect.signature(DPOTrainer.__init__)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_sig.parameters.values()
    )
    if "processing_class" in trainer_sig.parameters or has_var_kw:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    freeze_router_flag = bool(vram_cfg.get("freeze_moe_router", True))
    trainer_box: Dict[str, Any] = {}
    precompute_guard = (
        preserve_stage1_adapter_for_ref_precompute(model, trainer_box, dpo_trainer_cls=DPOTrainer)
        if bool(getattr(dpo_args, "precompute_ref_log_probs", False))
        else contextlib.nullcontext()
    )
    try:
        with precompute_guard:
            trainer = DPOTrainer(**trainer_kwargs)
            trainer_box["trainer"] = trainer
            if bool(getattr(dpo_args, "precompute_ref_log_probs", False)) and getattr(
                trainer, "_precomputed_train_ref_log_probs", True
            ) is False:
                # Force TRL 1.15.x lazy precomputation to execute now while `model` holds untouched Stage 1 SFT weights
                trainer.get_train_dataloader()
    finally:
        if saved_gemma4_proc is not None:
            setattr(tokenizer, "_gemma4_processor", saved_gemma4_proc)

    install_dpo_default_adapter_resume_hook(
        trainer,
        freeze_moe_router=freeze_router_flag,
    )
    if not args.smoke_test and last_ckpt is None:
        verify_step0_dpo_ref_alignment(
            trainer,
            model,
            is_moe=is_moe,
            freeze_moe_router=freeze_router_flag,
        )
    log_vram_telemetry(model, stage="pre_dpo_train")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    log_vram_telemetry(model, stage="post_dpo_train")

    # Restore unmodified canonical upstream chat_template before saving
    if original_chat_template:
        tokenizer.chat_template = original_chat_template
        proc = getattr(tokenizer, "_gemma4_processor", None)
        if proc is not None and hasattr(proc, "chat_template"):
            proc.chat_template = original_chat_template

    if not args.smoke_test:
        if offload_state.ple_module_names or offload_state.multimodal_module_names:
            restore_multimodal_towers(model, state=offload_state, target_device="cpu")
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(output_dir, selected_adapters=["default"])
            except TypeError:
                trainer.save_model(output_dir)
        else:
            trainer.save_model(output_dir)
        verify_saved_moe_adapter_config(out_path, is_moe=is_moe)
        proc_or_tok = getattr(tokenizer, "_gemma4_processor", tokenizer)
        proc_or_tok.save_pretrained(output_dir)

        metrics_payload = {
            "metrics": dict(train_result.metrics),
            "trainable_params": int(get_vram_telemetry(model).get("trainable_params", 0)),
            "log_history": list(getattr(trainer.state, "log_history", [])),
        }
        atomic_write_json(out_path / "dpo_metrics.json", metrics_payload)
        _cleanup_dpo_output_dir(out_path)

    logger.info("Stage 2 DPO completed: %s", train_result.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
