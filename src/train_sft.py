"""Stage 1 Supervised Fine-Tuning (SFT) training pipeline for Gemma-4-Arabic.

Configures:
- Unsloth 4-bit NF4 base model (`unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`) with
  8.0 GB VRAM CPU forward hook on `embed_tokens_per_layer` and CPU offload for
  `audio_tower` and `vision_tower`.
- rsLoRA (`use_rslora=True`, `r=32`, `lora_alpha=64`, `lora_dropout=0.0`, `bias="none"`,
  `use_gradient_checkpointing="unsloth"`).
- `TRL SFTTrainer` with `optim="paged_adamw_8bit"`, `per_device_train_batch_size=1`,
  `gradient_accumulation_steps=16`, `max_seq_length=2048`, `lr_scheduler_type="cosine"`,
  and `--smoke-test` (`max_steps=2`) verification mode.
"""

import argparse
import inspect
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

import torch
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_sft_dataset import build_synthetic_dry_run_pools
from src.constants import BASE_MODEL_ID, DEFAULT_MAX_SEQ_LENGTH, LORA_ALPHA, LORA_RANK
from src.model_loader import load_gemma4_8gb_model_and_tokenizer
from src.replay_mixer import mix_arabic_and_replay_samples
from src.vram_hooks import log_vram_telemetry, restore_multimodal_towers

logger = logging.getLogger(__name__)


def load_sft_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load and validate the Stage 1 SFT YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"SFT config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def build_sft_training_arguments(
    cfg: Dict[str, Any],
    output_dir: str,
    smoke_test: bool = False,
    max_steps_override: Optional[int] = None,
) -> Any:
    """Build TRL `SFTConfig` (or `TrainingArguments` fallback) from `sft_config.yaml`."""
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    max_seq_length = int(model_cfg.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH))
    max_steps = (
        max_steps_override
        if max_steps_override is not None
        else (2 if smoke_test else int(train_cfg.get("max_steps", -1)))
    )

    use_bf16 = bool(train_cfg.get("bf16", True)) and torch.cuda.is_available()
    optim_name = str(
        train_cfg.get("optim", "paged_adamw_8bit")
        if torch.cuda.is_available()
        else "adamw_torch"
    )

    common_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": (
            1 if smoke_test else int(train_cfg.get("gradient_accumulation_steps", 16))
        ),
        "optim": optim_name,
        "learning_rate": float(train_cfg.get("learning_rate", 1.0e-4)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.05)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 2)),
        "max_steps": max_steps,
        "bf16": use_bf16,
        "fp16": False,
        "logging_steps": 1 if smoke_test else int(train_cfg.get("logging_steps", 10)),
        "save_strategy": "no" if smoke_test else str(train_cfg.get("save_strategy", "steps")),
        "save_steps": int(train_cfg.get("save_steps", 250)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "seed": int(train_cfg.get("seed", 3407)),
        "report_to": "none",
    }

    neftune_alpha = train_cfg.get("neftune_noise_alpha", 5.0)
    if neftune_alpha is not None:
        common_kwargs["neftune_noise_alpha"] = float(neftune_alpha)

    try:
        from trl import SFTConfig

        sig = inspect.signature(SFTConfig.__init__)
        if "max_seq_length" in sig.parameters:
            common_kwargs["max_seq_length"] = max_seq_length
        elif "max_length" in sig.parameters:
            common_kwargs["max_length"] = max_seq_length
        if "dataset_text_field" in sig.parameters:
            common_kwargs["dataset_text_field"] = "text"
        if "packing" in sig.parameters:
            common_kwargs["packing"] = bool(train_cfg.get("packing", False))
        return SFTConfig(**common_kwargs)
    except ImportError:
        from transformers import TrainingArguments

        return TrainingArguments(**common_kwargs)


def load_or_build_sft_dataset(
    dataset_dir: pathlib.Path,
    smoke_test: bool = False,
) -> Any:
    """Load the prepared Stage 1 SFT dataset from disk or synthesize smoke-test samples."""
    from datasets import Dataset, load_from_disk

    if dataset_dir.exists() and not smoke_test:
        logger.info("Loading Stage 1 SFT dataset from disk: %s", dataset_dir)
        return load_from_disk(str(dataset_dir))

    if dataset_dir.exists() and smoke_test:
        ds = load_from_disk(str(dataset_dir))
        return ds.select(range(min(len(ds), 8)))

    if not smoke_test:
        raise FileNotFoundError(
            f"Prepared SFT dataset not found at {dataset_dir}. "
            "Run `python scripts/build_sft_dataset.py` first or pass `--smoke-test`."
        )

    logger.info("Building in-memory 75/25 mixed SFT dataset for --smoke-test mode.")
    pools = build_synthetic_dry_run_pools()
    mixed_samples, _ = mix_arabic_and_replay_samples(
        arabic_samples=pools["arabic"],
        replay_samples=pools["replay"],
        arabic_ratio=0.75,
        replay_ratio=0.25,
    )
    return Dataset.from_list(mixed_samples)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Stage 1 SFT training."""
    parser = argparse.ArgumentParser(
        description="Stage 1 rsLoRA Supervised Fine-Tuning (SFT) pipeline for Gemma-4-Arabic."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "sft_config.yaml"),
        help="Path to Stage 1 SFT YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the built Stage 1 SFT dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save Stage 1 rsLoRA checkpoints.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 2-step (max_steps=2) smoke test to verify pipeline configuration.",
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
        help="Validate config, arguments, and dataset without loading model weights.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for Stage 1 rsLoRA SFT training."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)

    cfg = load_sft_config(pathlib.Path(args.config))
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("datasets", {})

    output_dir = str(args.output_dir or train_cfg.get("output_dir", "outputs/sft_stage1"))
    dataset_dir = pathlib.Path(
        args.dataset_dir or ds_cfg.get("output_dataset_dir", "data/sft_mixed")
    )

    sft_args = build_sft_training_arguments(
        cfg=cfg,
        output_dir=output_dir,
        smoke_test=args.smoke_test,
        max_steps_override=args.max_steps,
    )
    train_dataset = load_or_build_sft_dataset(
        dataset_dir=dataset_dir,
        smoke_test=args.smoke_test or args.dry_run_only,
    )

    if args.dry_run_only:
        summary = {
            "status": "dry_run_verified",
            "base_model_id": model_cfg.get("base_model_id", BASE_MODEL_ID),
            "lora_r": int(lora_cfg.get("r", LORA_RANK)),
            "lora_alpha": int(lora_cfg.get("lora_alpha", LORA_ALPHA)),
            "use_rslora": bool(lora_cfg.get("use_rslora", True)),
            "max_steps": sft_args.max_steps,
            "lr_scheduler_type": str(sft_args.lr_scheduler_type),
            "dataset_rows": len(train_dataset),
            "output_dir": output_dir,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    from trl import SFTTrainer

    model, tokenizer, offload_state = load_gemma4_8gb_model_and_tokenizer(
        model_name=str(model_cfg.get("base_model_id", BASE_MODEL_ID)),
        max_seq_length=int(model_cfg.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)),
        dtype=str(model_cfg.get("dtype", "bfloat16")),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        attach_lora=True,
        lora_r=int(lora_cfg.get("r", LORA_RANK)),
        lora_alpha=int(lora_cfg.get("lora_alpha", LORA_ALPHA)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        use_rslora=bool(lora_cfg.get("use_rslora", True)),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=int(lora_cfg.get("random_state", 3407)),
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_sig.parameters:
        trainer_kwargs["max_seq_length"] = int(
            model_cfg.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
        )

    trainer = SFTTrainer(**trainer_kwargs)
    log_vram_telemetry(model, stage="pre_sft_train")
    train_result = trainer.train()
    log_vram_telemetry(model, stage="post_sft_train")

    if not args.smoke_test:
        restore_multimodal_towers(model, state=offload_state, target_device="cpu")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    logger.info("Stage 1 SFT completed: %s", train_result.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
