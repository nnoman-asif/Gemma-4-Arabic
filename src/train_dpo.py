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
import inspect
import json
import logging
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional

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
    verify_moe_peft_version,
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
        if "precompute_ref_log_probs" in sig.parameters:
            common_kwargs["precompute_ref_log_probs"] = bool(
                vram_cfg.get("precompute_ref_log_probs", False)
            )
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

    original_chat_template = getattr(tokenizer, "chat_template", None)
    if is_moe and original_chat_template:
        tokenizer.chat_template = align_26b_chat_template_for_dpo(original_chat_template)
        proc = getattr(tokenizer, "_gemma4_processor", None)
        if proc is not None and hasattr(proc, "chat_template"):
            proc.chat_template = tokenizer.chat_template

    # Hard pre-train verification that `prompt_ids` is an exact prefix of `prompt + chosen/rejected`
    verify_dpo_prefix_alignment(train_dataset[0], tokenizer)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer_sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)
    install_dpo_default_adapter_resume_hook(
        trainer,
        freeze_moe_router=bool(vram_cfg.get("freeze_moe_router", True)),
    )
    log_vram_telemetry(model, stage="pre_dpo_train")

    last_ckpt = None
    if not args.smoke_test and out_dir_path.exists():
        last_ckpt = find_latest_valid_checkpoint(out_dir_path)
        if last_ckpt is not None:
            logger.info("Resuming Stage 2 DPO from latest verified checkpoint: %s", last_ckpt)

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
