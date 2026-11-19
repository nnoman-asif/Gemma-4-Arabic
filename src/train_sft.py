"""Stage 1 Supervised Fine-Tuning (SFT) training pipeline for Gemma-4-Arabic (E4B Dense & 26B-A4B MoE).

Key architectural guarantees:
1. Pre-tokenizes every conversation using `build_tokenized_sft_Turn_masks` (`is_26b_moe`-aware)
   with exact response-only `labels` (`-100` on `<bos>`, system, user, and `<|turn>model\\n`)
   BEFORE `TRL SFTTrainer` `bfd` sequence packing runs.
2. Verified pre-train assertions:
   - Zero `<unk>` (`id=3`) legacy turn tokens.
   - Non-empty supervised assistant completion tokens (`labels != -100`) and masked prompt header.
   - On `Gemma-4-26B-A4B` (`architecture_type="moe"`), asserts `>= 250M` trainable parameters
     (`~498.5M` at `r=16` across attention + shared MLP + all 128 3D routed experts).
3. Automatic intermediate `checkpoint-*` cleanup after final adapter verification to protect disk space.
"""

import argparse
import inspect
import json
import logging
import os
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

# Enforce deterministic SDPA / MoE / decoder-block kernels across Stage 1 SFT and Stage 2 DPO
# (matching `src/train_dpo.py` before Unsloth initializes module patches).
os.environ["UNSLOTH_BLOCK_COMPILE"] = "0"
os.environ["UNSLOTH_MOE_LORA_FOLD"] = "0"
os.environ["UNSLOTH_MOE_GC_KEEP_BASE"] = "0"
os.environ["UNSLOTH_MOE_ROUTED_KERNEL"] = "0"
os.environ["UNSLOTH_HD512_ATTENTION"] = "0"

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
import yaml

from scripts.build_sft_dataset import build_synthetic_dry_run_pools
from src.chat_template import (
    build_tokenized_sft_Turn_masks,
    normalize_messages_schema,
    parse_reasoning_and_content,
    strip_embedded_thinking_from_content,
)
from src.constants import (
    BASE_MODEL_ID,
    DEFAULT_MAX_SEQ_LENGTH,
    LORA_ALPHA,
    LORA_RANK,
    load_dotenv_if_present,
)
from src.replay_mixer import mix_arabic_and_replay_samples
from src.vram_hooks import get_vram_telemetry, log_vram_telemetry, restore_multimodal_towers

load_dotenv_if_present()
logger = logging.getLogger(__name__)


def load_sft_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load and validate the Stage 1 SFT YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"SFT config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _is_moe_model_cfg(model_cfg: Dict[str, Any]) -> bool:
    arch = str(model_cfg.get("architecture_type", "")).strip().lower()
    base_id = str(model_cfg.get("base_model_id", "")).strip().lower()
    return arch == "moe" or "26b" in base_id or "a4b" in base_id


def _has_arabic_script(text: str, min_chars: int = 8) -> bool:
    ar_chars = 0
    for ch in text:
        if "\u0600" <= ch <= "\u06ff":
            ar_chars += 1
            if ar_chars >= min_chars:
                return True
    return False


def _collect_concise_prior_turn_pools(
    dataset: Any,
    max_pair_chars: int = 900,
    max_pool_size: int = 2048,
) -> Tuple[List[Tuple[Dict[str, str], Dict[str, str]]], List[Tuple[Dict[str, str], Dict[str, str]]]]:
    """Build deterministic Arabic and English `(user, stripped_assistant)` prior-turn context pools.

    In Google's official Gemma 4 `E4B` `chat_template.jinja`, multi-turn / few-shot conversations
    with `enable_thinking=True` place `<|turn>system\\n<|think|>\\n<turn|>\\n` at token 0 and strip
    `<|channel>thought...<channel|>` from all prior assistant turns (`loop.index0 < last_user_idx`),
    emitting `<|channel>thought\\n` ONLY on the final active assistant turn.
    Prepending 2–3 masked (`assistant_masks = 0`) stripped prior turns onto a 25% stride of single-turn
    thinking SFT examples explicitly teaches `E4B` to preserve `<|channel>thought\\n` activation across
    multi-turn conversations and few-shot reasoning prompts without supervising any context tokens.
    """
    ar_pool: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    en_pool: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    n_rows = len(dataset)
    for idx in range(n_rows):
        if len(ar_pool) >= max_pool_size and len(en_pool) >= max_pool_size:
            break
        row = dataset[idx]
        norm_msgs = normalize_messages_schema(row.get("messages") or [])
        non_sys = [m for m in norm_msgs if m["role"] in ("user", "assistant")]
        if len(non_sys) < 2 or non_sys[0]["role"] != "user" or non_sys[1]["role"] != "assistant":
            continue
        u_txt = non_sys[0]["content"].strip()
        a_txt = strip_embedded_thinking_from_content(non_sys[1]["content"]).strip()
        if len(u_txt) < 6 or len(a_txt) < 8 or (len(u_txt) + len(a_txt)) > max_pair_chars:
            continue
        pair = (
            {"role": "user", "content": u_txt, "reasoning_content": ""},
            {"role": "assistant", "content": a_txt, "reasoning_content": ""},
        )
        if _has_arabic_script(u_txt + " " + a_txt):
            if len(ar_pool) < max_pool_size:
                ar_pool.append(pair)
        else:
            if len(en_pool) < max_pool_size:
                en_pool.append(pair)
    return ar_pool, en_pool


def pretokenize_sft_dataset(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
    is_26b_moe: bool = False,
    cache_dir: Optional[pathlib.Path] = None,
    augment_multiturn_thinking: bool = True,
) -> Any:
    """Convert raw `messages` SFT rows into `input_ids`, `attention_mask`, and response-only `assistant_masks`.

    Note: `"labels"` is intentionally NOT pre-materialized as a column here so Unsloth's
    `sft_prepare_dataset` keeps TRL's `seq_lengths`-aware `DataCollatorForLanguageModeling`
    (instead of switching to `DataCollatorForSeq2Seq`, which cannot pad jagged `seq_lengths`
    lists produced by `bfd` sequence packing), while both Unsloth and TRL automatically build
    `labels = [tid if m else -100]` from `assistant_masks` immediately prior to `pack_dataset`.
    """
    from datasets import Dataset, load_from_disk

    do_multiturn_aug = bool(augment_multiturn_thinking and not is_26b_moe)
    cache_meta = {
        "schema": "v2_multiturn_thinking_gate",
        "num_input_rows": len(dataset),
        "max_seq_length": int(max_seq_length),
        "is_26b_moe": bool(is_26b_moe),
        "multiturn_thinking_aug": do_multiturn_aug,
    }
    if cache_dir is not None:
        ready_file = cache_dir / "cache_ready.json"
        arrow_dir = cache_dir / "arrow"
        if ready_file.is_file() and arrow_dir.is_dir():
            try:
                saved_meta = json.loads(ready_file.read_text(encoding="utf-8"))
                if saved_meta == cache_meta:
                    cached_ds = load_from_disk(str(arrow_dir))
                    if len(cached_ds) > 0:
                        logger.info(
                            "Loaded cached pre-tokenized SFT dataset (%d rows) from %s.",
                            len(cached_ds),
                            arrow_dir,
                        )
                        return cached_ds
            except Exception as cache_exc:
                logger.warning("Pre-tokenized SFT cache invalid (%s); rebuilding.", cache_exc)
                shutil.rmtree(cache_dir, ignore_errors=True)

    ar_prior_pool: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    en_prior_pool: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    if do_multiturn_aug and len(dataset) >= 8:
        ar_prior_pool, en_prior_pool = _collect_concise_prior_turn_pools(dataset)

    tokenized_rows: List[Dict[str, Any]] = []
    skipped = 0
    single_thinking_seen = 0
    multiturn_aug_count = 0

    for idx in range(len(dataset)):
        row = dataset[idx]
        msgs = row.get("messages")
        enable_thinking = bool(row.get("enable_thinking", False))

        if isinstance(msgs, list) and msgs:
            tok_out: Optional[Dict[str, List[int]]] = None
            if do_multiturn_aug and enable_thinking and (len(ar_prior_pool) >= 3 or len(en_prior_pool) >= 3):
                norm_msgs = normalize_messages_schema(msgs)
                asst_count = sum(1 for m in norm_msgs if m["role"] == "assistant")
                if asst_count == 1 and norm_msgs and norm_msgs[-1]["role"] == "assistant":
                    single_thinking_seen += 1
                    if (single_thinking_seen - 1) % 4 == 0:
                        sys_msgs = [norm_msgs[0]] if norm_msgs[0]["role"] in ("system", "developer") else []
                        core_msgs = norm_msgs[1:] if sys_msgs else norm_msgs
                        target_u = core_msgs[0]["content"] if core_msgs else ""
                        prefer_ar = _has_arabic_script(target_u)
                        cand_pool = (
                            ar_prior_pool
                            if (prefer_ar and len(ar_prior_pool) >= 3)
                            else (en_prior_pool if len(en_prior_pool) >= 3 else ar_prior_pool)
                        )
                        if len(cand_pool) >= 3:
                            num_prior = 2 if ((single_thinking_seen - 1) % 8 == 0) else 3
                            start_cursor = ((single_thinking_seen - 1) * 3) % len(cand_pool)
                            chosen_priors: List[Dict[str, str]] = []
                            picked_pairs = 0
                            for offset in range(len(cand_pool)):
                                p_u, p_a = cand_pool[(start_cursor + offset) % len(cand_pool)]
                                if p_u["content"] == target_u:
                                    continue
                                chosen_priors.append(p_u)
                                chosen_priors.append(p_a)
                                picked_pairs += 1
                                if picked_pairs >= num_prior:
                                    break
                            if picked_pairs >= 2:
                                aug_msgs = sys_msgs + chosen_priors + core_msgs
                                tok_out = build_tokenized_sft_Turn_masks(
                                    messages=aug_msgs,
                                    tokenizer=tokenizer,
                                    enable_thinking=True,
                                    max_seq_length=max_seq_length,
                                    is_26b_moe=is_26b_moe,
                                )
                                if tok_out is not None:
                                    multiturn_aug_count += 1

            if tok_out is None:
                tok_out = build_tokenized_sft_Turn_masks(
                    messages=msgs,
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                    max_seq_length=max_seq_length,
                    is_26b_moe=is_26b_moe,
                )
            if tok_out is not None:
                ids = tok_out["input_ids"]
                masks = tok_out["assistant_masks"]
                tokenized_rows.append(
                    {
                        "input_ids": ids,
                        "attention_mask": tok_out["attention_mask"],
                        "assistant_masks": masks,
                    }
                )
                continue

        skipped += 1

    if not tokenized_rows:
        raise RuntimeError(
            f"Pre-tokenization produced 0 valid rows out of {len(dataset)} input SFT examples."
        )

    # Hard invariant assertions on Sample 0 before GPU training starts
    sample0 = tokenized_rows[0]
    if sample0["assistant_masks"][0] != 0:
        raise RuntimeError(
            "SFT pre-tokenization invariant failed: leading <bos>/prompt token is not masked (assistant_masks[0] != 0)!"
        )
    if not any(m == 1 for m in sample0["assistant_masks"]):
        raise RuntimeError(
            "SFT pre-tokenization invariant failed: sample 0 has 0 supervised assistant tokens!"
        )
    unk_id = getattr(tokenizer, "unk_token_id", 3)
    if unk_id is not None and int(unk_id) in sample0["input_ids"][:12]:
        raise RuntimeError(
            f"SFT pre-tokenization detected <unk> (id={unk_id}) in turn header! Legacy markers must never appear."
        )

    sup_tokens = sum(sum(r["assistant_masks"]) for r in tokenized_rows)
    tot_tokens = sum(len(r["input_ids"]) for r in tokenized_rows)
    logger.info(
        "Pre-tokenized %d SFT examples (skipped=%d, is_26b_moe=%s, multiturn_thinking_aug=%d, max_seq_len=%d): "
        "%d / %d tokens supervised (%.1f%% assistant completion tokens).",
        len(tokenized_rows),
        skipped,
        is_26b_moe,
        multiturn_aug_count,
        max_seq_length,
        sup_tokens,
        tot_tokens,
        100.0 * sup_tokens / max(1, tot_tokens),
    )
    built_ds = Dataset.from_list(tokenized_rows)
    if cache_dir is not None:
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            arrow_dir = cache_dir / "arrow"
            arrow_dir.parent.mkdir(parents=True, exist_ok=True)
            built_ds.save_to_disk(str(arrow_dir))
            atomic_write_json(cache_dir / "cache_ready.json", cache_meta)
        except Exception as save_cache_exc:
            logger.warning("Could not persist pre-tokenized SFT cache at %s: %s", cache_dir, save_cache_exc)
    return built_ds


def _prune_intermediate_checkpoints(output_dir: pathlib.Path) -> int:
    """Delete intermediate `checkpoint-*` and `.pretokenized_cache` folders once final adapter + metrics exist."""
    if not is_adapter_stage_complete(output_dir, "training_metrics.json"):
        return 0
    removed = 0
    for child in output_dir.iterdir():
        if child.is_dir() and (child.name.startswith("checkpoint-") or child.name == ".pretokenized_cache"):
            shutil.rmtree(child, ignore_errors=True)
            if child.name.startswith("checkpoint-"):
                removed += 1
    if removed:
        logger.info("Pruned %d intermediate SFT checkpoint-* directories under %s.", removed, output_dir)
    return removed


def build_sft_training_arguments(
    cfg: Dict[str, Any],
    output_dir: str,
    smoke_test: bool = False,
    max_steps_override: Optional[int] = None,
    has_eval_dataset: bool = False,
) -> Any:
    """Build TRL `SFTConfig` (or `TrainingArguments` fallback) from YAML configuration."""
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    use_gc = bool(lora_cfg.get("use_gradient_checkpointing", True))

    max_seq_length = int(model_cfg.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH))
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
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": (
            1 if smoke_test else int(train_cfg.get("gradient_accumulation_steps", 8))
        ),
        "optim": optim_name,
        "learning_rate": float(train_cfg.get("learning_rate", 5.0e-5)),
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
        common_kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 250))

    warmup_ratio_val = float(train_cfg.get("warmup_ratio", 0.05))
    neftune_alpha = train_cfg.get("neftune_noise_alpha")
    if neftune_alpha is not None and float(neftune_alpha) > 0:
        common_kwargs["neftune_noise_alpha"] = float(neftune_alpha)

    try:
        from trl import SFTConfig

        sig = inspect.signature(SFTConfig.__init__)
        if "warmup_ratio" in sig.parameters:
            common_kwargs["warmup_ratio"] = warmup_ratio_val
        elif "warmup_steps" in sig.parameters:
            common_kwargs["warmup_steps"] = warmup_ratio_val
        if "max_seq_length" in sig.parameters:
            common_kwargs["max_seq_length"] = max_seq_length
        elif "max_length" in sig.parameters:
            common_kwargs["max_length"] = max_seq_length
        if "packing" in sig.parameters:
            common_kwargs["packing"] = (
                False if smoke_test else bool(train_cfg.get("packing", False))
            )
        if "completion_only_loss" in sig.parameters:
            common_kwargs["completion_only_loss"] = bool(
                train_cfg.get("completion_only_loss", True)
            )
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            common_kwargs = {k: v for k, v in common_kwargs.items() if k in sig.parameters}
        return SFTConfig(**common_kwargs)
    except ImportError:
        from transformers import TrainingArguments

        ta_sig = inspect.signature(TrainingArguments.__init__)
        if "warmup_ratio" in ta_sig.parameters:
            common_kwargs["warmup_ratio"] = warmup_ratio_val
        elif "warmup_steps" in ta_sig.parameters:
            common_kwargs["warmup_steps"] = warmup_ratio_val
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ta_sig.parameters.values()):
            common_kwargs = {k: v for k, v in common_kwargs.items() if k in ta_sig.parameters}
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
            "Run `python scripts/build_sft_dataset.py --pull-from-hub` first or pass `--smoke-test`."
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
        description="Stage 1 Supervised Fine-Tuning (SFT) pipeline for Gemma-4-Arabic."
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
        help="Directory to save Stage 1 LoRA checkpoints.",
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
    """Entry point for Stage 1 SFT training."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    configure_clean_logging()
    args = parse_args(argv)

    cfg = load_sft_config(pathlib.Path(args.config))
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("datasets", {})
    vram_cfg = cfg.get("vram_optimization", {})

    output_dir = str(args.output_dir or train_cfg.get("output_dir", "outputs/sft_stage1"))
    dataset_dir = pathlib.Path(
        args.dataset_dir or ds_cfg.get("output_dataset_dir", "data/sft_mixed")
    )
    is_moe = _is_moe_model_cfg(model_cfg)
    max_seq_len = int(model_cfg.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH))

    full_dataset = load_or_build_sft_dataset(
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

    sft_args = build_sft_training_arguments(
        cfg=cfg,
        output_dir=output_dir,
        smoke_test=args.smoke_test,
        max_steps_override=args.max_steps,
        has_eval_dataset=(eval_dataset is not None),
    )
    configure_clean_logging()

    if args.dry_run_only:
        summary = {
            "status": "dry_run_verified",
            "base_model_id": model_cfg.get("base_model_id", BASE_MODEL_ID),
            "is_moe": is_moe,
            "max_seq_length": max_seq_len,
            "lora_r": int(lora_cfg.get("r", LORA_RANK)),
            "lora_alpha": int(lora_cfg.get("lora_alpha", LORA_ALPHA)),
            "use_rslora": bool(lora_cfg.get("use_rslora", False)),
            "offload_ple_to_cpu": bool(vram_cfg.get("offload_ple_to_cpu", True)),
            "offload_multimodal_towers_to_cpu": bool(
                vram_cfg.get("offload_multimodal_towers_to_cpu", True)
            ),
            "freeze_moe_router": bool(vram_cfg.get("freeze_moe_router", True)),
            "max_steps": sft_args.max_steps,
            "lr_scheduler_type": str(sft_args.lr_scheduler_type),
            "dataset_rows": len(train_dataset),
            "output_dir": output_dir,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    out_dir_path = pathlib.Path(output_dir)
    if not args.smoke_test and is_adapter_stage_complete(
        out_dir_path, "training_metrics.json", cleanup_partial_root=True
    ):
        logger.info("Stage 1 SFT final adapter already verified at %s; skipping training.", output_dir)
        return 0

    if is_moe and not args.smoke_test and torch.cuda.is_available():
        verify_moe_peft_version()

    from trl import SFTTrainer

    model, tokenizer, offload_state = load_gemma4_8gb_model_and_tokenizer(
        model_name=str(model_cfg.get("base_model_id", BASE_MODEL_ID)),
        max_seq_length=max_seq_len,
        dtype=str(model_cfg.get("dtype", "bfloat16")),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        attach_lora=True,
        lora_r=int(lora_cfg.get("r", LORA_RANK)),
        lora_alpha=int(lora_cfg.get("lora_alpha", LORA_ALPHA)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        use_rslora=bool(lora_cfg.get("use_rslora", False)),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=int(lora_cfg.get("random_state", 3407)),
        target_modules=lora_cfg.get("target_modules"),
        exclude_modules=lora_cfg.get("exclude_modules"),
        offload_ple_to_cpu=bool(vram_cfg.get("offload_ple_to_cpu", True)),
        offload_multimodal_towers_to_cpu=bool(
            vram_cfg.get("offload_multimodal_towers_to_cpu", True)
        ),
        freeze_moe_router=bool(vram_cfg.get("freeze_moe_router", True)),
    )

    # Hard MoE expert parameter count & 3D ParamWrapper layout verification on 26B-A4B
    log_vram_telemetry(model, stage="pre_sft_train")
    if is_moe and (not args.smoke_test or torch.cuda.is_available()):
        verify_moe_lora_integrity(
            model,
            is_moe=is_moe,
            smoke_test=args.smoke_test,
            expect_trained_adapter=False,
            require_frozen_router=bool(vram_cfg.get("freeze_moe_router", True)),
        )

    # Pre-tokenize with exact response-only labels (-100 on all prompt tokens BEFORE packing)
    pretok_cache_root = (
        out_dir_path / ".pretokenized_cache" / ("moe" if is_moe else "dense")
        if not args.smoke_test
        else None
    )
    train_dataset = pretokenize_sft_dataset(
        train_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        is_26b_moe=is_moe,
        cache_dir=(pretok_cache_root / "train") if pretok_cache_root is not None else None,
    )
    if eval_dataset is not None:
        eval_dataset = pretokenize_sft_dataset(
            eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_len,
            is_26b_moe=is_moe,
            cache_dir=(pretok_cache_root / "eval") if pretok_cache_root is not None else None,
        )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer_sig = inspect.signature(SFTTrainer.__init__)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_sig.parameters.values()
    )
    if "processing_class" in trainer_sig.parameters or has_var_kw:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_sig.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_len

    trainer = SFTTrainer(**trainer_kwargs)

    # Hard post-packing & data_collator verification (catches any collator/seq_lengths mismatch before step 1)
    if len(trainer.train_dataset) > 0:
        first_prepared = trainer.train_dataset[0]
        if "labels" in first_prepared:
            if first_prepared["labels"][0] != -100 or not any(
                lbl != -100 for lbl in first_prepared["labels"]
            ):
                raise RuntimeError(
                    "CRITICAL SFTTrainer dataset preparation error: `trainer.train_dataset[0]['labels']` "
                    "is not prompt-masked (`labels[0] == -100` with supervised completion tokens)!"
                )
        elif "assistant_masks" not in first_prepared:
            raise RuntimeError(
                "CRITICAL SFTTrainer dataset preparation error: `trainer.train_dataset[0]` has neither "
                "`labels` nor `assistant_masks` — response-only masking would be lost!"
            )
        probe_rows = [
            trainer.train_dataset[0],
            trainer.train_dataset[min(1, len(trainer.train_dataset) - 1)],
        ]
        collated_probe = trainer.data_collator(probe_rows)
        if (
            "labels" not in collated_probe
            or int(collated_probe["labels"][0, 0].item()) != -100
            or int((collated_probe["labels"] != -100).sum().item()) == 0
        ):
            raise RuntimeError(
                "CRITICAL SFTTrainer data_collator probe failed: collated batch is missing "
                "prompt-masked `labels` (`labels[0, 0] == -100` and `sum(labels != -100) > 0`)!"
            )

    last_ckpt = None
    if not args.smoke_test and out_dir_path.exists():
        if is_adapter_stage_complete(out_dir_path, "training_metrics.json"):
            logger.info("Stage 1 SFT final adapter already verified at %s; skipping training.", output_dir)
            return 0
        last_ckpt = find_latest_valid_checkpoint(out_dir_path)
        if last_ckpt is not None:
            logger.info("Resuming Stage 1 SFT from latest verified checkpoint: %s", last_ckpt)

    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    log_vram_telemetry(model, stage="post_sft_train")

    if not args.smoke_test:
        if offload_state.ple_module_names or offload_state.multimodal_module_names:
            restore_multimodal_towers(model, state=offload_state, target_device="cpu")
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        verify_saved_moe_adapter_config(out_path, is_moe=is_moe)
        proc_or_tok = getattr(tokenizer, "_gemma4_processor", tokenizer)
        proc_or_tok.save_pretrained(output_dir)
        metrics_payload = {
            "metrics": dict(train_result.metrics),
            "trainable_params": int(get_vram_telemetry(model).get("trainable_params", 0)),
            "log_history": list(getattr(trainer.state, "log_history", [])),
        }
        atomic_write_json(out_path / "training_metrics.json", metrics_payload)
        _prune_intermediate_checkpoints(out_path)

    logger.info("Stage 1 SFT completed: %s", train_result.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
