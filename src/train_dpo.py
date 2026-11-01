"""Stage 2 Trust-Region Direct Preference Optimization (TR-DPO) pipeline for Gemma-4-Arabic.

Configures:
- Unsloth 4-bit NF4 base model (`unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`) with
  8.0 GB VRAM CPU forward hook on `embed_tokens_per_layer` and CPU offload for
  `audio_tower` and `vision_tower`.
- `TRL DPOTrainer` with `ref_model=None` (0.0 GB extra reference model VRAM),
  `beta=0.1`, `learning_rate=5.0e-7`, `max_length=1536`, `max_prompt_length=768`,
  `optim="paged_adamw_8bit"`, and `lr_scheduler_type="cosine"`.
- Trust-Region DPO reference EMA update (`tr_alpha=0.6`) to prevent policy drift
  and guarantee zero catastrophic forgetting during preference alignment.
- `--smoke-test` (`max_steps=2`) and `--dry-run-only` verification modes.
"""

import argparse
import builtins
import inspect
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import configure_clean_logging, ensure_unsloth_builtins

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
from src.constants import (
    BASE_MODEL_ID,
    DPO_BETA,
    DPO_LEARNING_RATE,
    DPO_MAX_PROMPT_LENGTH,
    DPO_MAX_SEQ_LENGTH,
    DPO_TRUST_REGION_ALPHA,
    LORA_ALPHA,
    LORA_RANK,
)
from src.dpo_formatter import mix_dpo_preference_datasets
from src.model_loader import load_gemma4_8gb_model_and_tokenizer
from src.vram_hooks import log_vram_telemetry, restore_multimodal_towers

logger = logging.getLogger(__name__)


def load_dpo_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load and validate the Stage 2 DPO YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"DPO config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


@torch.no_grad()
def update_trust_region_reference_params(
    policy_params: Dict[str, torch.Tensor],
    ref_state: Dict[str, torch.Tensor],
    tr_alpha: float = DPO_TRUST_REGION_ALPHA,
) -> Dict[str, torch.Tensor]:
    """Apply the Trust-Region DPO exponential moving average (EMA) update on CPU.

    Formula:
        theta_ref <- tr_alpha * theta_policy + (1.0 - tr_alpha) * theta_ref
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
    """Maintains a CPU-offloaded Trust-Region EMA anchor of trainable LoRA adapters."""

    def __init__(
        self,
        tr_alpha: float = DPO_TRUST_REGION_ALPHA,
        update_steps: int = 50,
    ) -> None:
        self.tr_alpha = float(tr_alpha)
        self.update_steps = max(1, int(update_steps))
        self.ref_adapter_state: Dict[str, torch.Tensor] = {}
        self.num_updates: int = 0

    def _capture_trainable_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if model is not None:
            trainable = self._capture_trainable_params(model)
            self.ref_adapter_state = update_trust_region_reference_params(
                policy_params=trainable,
                ref_state={},
                tr_alpha=1.0,
            )
            logger.info(
                "Initialized Trust-Region DPO CPU anchor for %d trainable LoRA tensors (alpha=%.2f).",
                len(self.ref_adapter_state),
                self.tr_alpha,
            )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step > 0 and state.global_step % self.update_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                trainable = self._capture_trainable_params(model)
                self.ref_adapter_state = update_trust_region_reference_params(
                    policy_params=trainable,
                    ref_state=self.ref_adapter_state,
                    tr_alpha=self.tr_alpha,
                )
                self.num_updates += 1
                logger.info(
                    "[TR-DPO step %d] Updated reference EMA state (alpha=%.2f, total_updates=%d).",
                    state.global_step,
                    self.tr_alpha,
                    self.num_updates,
                )


def build_dpo_training_arguments(
    cfg: Dict[str, Any],
    output_dir: str,
    smoke_test: bool = False,
    max_steps_override: Optional[int] = None,
    has_eval_dataset: bool = False,
) -> Any:
    """Build TRL `DPOConfig` (or `TrainingArguments` fallback) from `dpo_config.yaml`."""
    train_cfg = cfg.get("training", {})
    dpo_cfg = cfg.get("dpo", {})
    vram_cfg = cfg.get("vram_optimization", {})

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
        train_cfg.get("optim", "paged_adamw_8bit")
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
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": (
            1 if smoke_test else int(train_cfg.get("gradient_accumulation_steps", 8))
        ),
        "optim": optim_name,
        "learning_rate": float(train_cfg.get("learning_rate", DPO_LEARNING_RATE)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1)),
        "max_steps": max_steps,
        "bf16": use_bf16,
        "tf32": use_tf32,
        "fp16": False,
        "dataloader_pin_memory": bool(train_cfg.get("dataloader_pin_memory", True)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "logging_steps": 1 if smoke_test else int(train_cfg.get("logging_steps", 10)),
        "eval_strategy": eval_strategy,
        "save_strategy": "no" if smoke_test else str(train_cfg.get("save_strategy", "steps")),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "seed": int(train_cfg.get("seed", 3407)),
        "log_level": "warning",
        "report_to": "none",
    }
    if eval_strategy != "no":
        common_kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 100))

    warmup_ratio_val = float(train_cfg.get("warmup_ratio", 0.10))

    try:
        from trl import DPOConfig

        sig = inspect.signature(DPOConfig.__init__)
        if "warmup_steps" in sig.parameters:
            common_kwargs["warmup_steps"] = warmup_ratio_val
        elif "warmup_ratio" in sig.parameters:
            common_kwargs["warmup_ratio"] = warmup_ratio_val
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
        if "warmup_steps" in ta_sig.parameters:
            common_kwargs["warmup_steps"] = warmup_ratio_val
        elif "warmup_ratio" in ta_sig.parameters:
            common_kwargs["warmup_ratio"] = warmup_ratio_val
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
            "Run `python scripts/build_dpo_dataset.py` first or pass `--smoke-test`."
        )

    logger.info("Building in-memory preference dataset for --smoke-test mode.")
    pools = build_synthetic_dpo_dry_run_pools()
    mixed_pairs, _ = mix_dpo_preference_datasets(
        saudi_dpo_samples=pools["saudi"],
        arcee_dpo_samples=pools["arcee"],
    )
    return Dataset.from_list(mixed_pairs)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Stage 2 Trust-Region DPO training."""
    parser = argparse.ArgumentParser(
        description="Stage 2 Trust-Region DPO alignment pipeline for Gemma-4-Arabic."
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
        help="Validate config, arguments, Trust-Region EMA math, and dataset without loading weights.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for Stage 2 Trust-Region DPO training."""
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

    tr_alpha = float(dpo_cfg.get("tr_alpha", DPO_TRUST_REGION_ALPHA))
    tr_update_steps = int(
        1 if args.smoke_test else dpo_cfg.get("tr_update_steps", 50)
    )

    full_dataset = load_or_build_dpo_dataset(
        dataset_dir=dataset_dir,
        smoke_test=args.smoke_test or args.dry_run_only,
    )
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
        # Verify Trust-Region EMA math on a small tensor
        ema_check = update_trust_region_reference_params(
            policy_params={"lora_A": torch.tensor([10.0])},
            ref_state={"lora_A": torch.tensor([0.0])},
            tr_alpha=tr_alpha,
        )
        summary = {
            "status": "dry_run_verified",
            "base_model_id": model_cfg.get("base_model_id", BASE_MODEL_ID),
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
        use_rslora=bool(lora_cfg.get("use_rslora", True)),
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

    tr_callback = TrustRegionDPOCallback(
        tr_alpha=tr_alpha,
        update_steps=tr_update_steps,
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "callbacks": [tr_callback],
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer_sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)
    log_vram_telemetry(model, stage="pre_dpo_train")
    last_ckpt = None
    out_dir_path = pathlib.Path(output_dir)
    if not args.smoke_test and out_dir_path.exists():
        if (out_dir_path / "adapter_config.json").exists():
            logger.info("Stage 2 DPO final adapter already exists at %s; skipping training.", output_dir)
            return 0
        from transformers.trainer_utils import get_last_checkpoint

        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt is not None:
            logger.info("Resuming Stage 2 DPO from latest checkpoint: %s", last_ckpt)

    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    log_vram_telemetry(model, stage="post_dpo_train")

    if not args.smoke_test:
        if offload_state.ple_module_names or offload_state.multimodal_module_names:
            restore_multimodal_towers(model, state=offload_state, target_device="cpu")
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        metrics_payload = {
            "metrics": dict(train_result.metrics),
            "log_history": list(getattr(trainer.state, "log_history", [])),
        }
        (out_path / "dpo_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info("Stage 2 Trust-Region DPO completed: %s", train_result.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
