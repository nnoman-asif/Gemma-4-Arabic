"""Automated benchmark runner and Zero-Forgetting Retention Gate evaluator.

Evaluates:
- 8 Native `lm_eval` Benchmarks (`ArabicMMLU`, `AlGhafa`, `ACVA`, `EXAMS-AR`,
  `ARC-Challenge`, `GSM8K`, `HumanEval`, `MMLU-Pro`) + `MGSM-Ar` using default
  `lm-evaluation-harness` (`lm_eval`) settings with `vLLM` (`batch_size="auto"`).
- 2 Native `lighteval` Benchmarks (`MadinahQA`, `AraTrust`) using default
  `lighteval` settings with `vLLM`.
- Supports single-model (`e4b` or `26b_a4b`) and dual-model family (`both`)
  comparisons across `Gemma-4-E4B-Arabic-It` and `Gemma-4-26B-A4B-Arabic-It`
  using isolated per-GPU subprocesses (`CUDA_VISIBLE_DEVICES=0` and `1`).
"""

import argparse
import gc
import json
import logging
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

# Enforce deterministic hashing, multiprocessing spawn for vLLM, and HumanEval execution
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

for _noisy_logger in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

EVAL_CACHE_VERSION = "v16_instruct_oall_v2"

# Core ACVA subtasks in lm_eval covering Saudi & Arab cultural alignment when --limit is used
ACVA_CORE_SUBTASKS: Tuple[str, ...] = (
    "arabic_leaderboard_acva_Saudi_Arabia_light",
    "arabic_leaderboard_acva_Arab_Empire_light",
    "arabic_leaderboard_acva_Arabic_Culture_light",
    "arabic_leaderboard_acva_Islam_Education_light",
    "arabic_leaderboard_acva_Islamic_law_system_light",
)

# Number of subtasks in multi-task lm_eval groups for proportional --limit scaling
GROUP_SUBTASK_COUNTS: Dict[str, int] = {
    "arabicmmlu": 40,
    "arabic_leaderboard_alghafa_light": 9,
    "arabic_leaderboard_acva_light": len(ACVA_CORE_SUBTASKS),
    "mmlu_pro": 14,
}

# Native LightEval task definitions for OALL v2 Arabic benchmarks
LIGHTEVAL_SUITE_TASK_MAP: Dict[str, Tuple[str, ...]] = {
    "ArabicMMLU": (
        "community|arabic_mmlu:All|0",
    ),
    "MadinahQA": (
        "community|madinah_qa:Arabic Language (General)|0",
        "community|madinah_qa:Arabic Language (Grammar)|0",
    ),
    "AlGhafa": (
        "community|alghafa:mcq_exams_test_ar|0",
        "community|alghafa:meta_ar_dialects|0",
        "community|alghafa:meta_ar_msa|0",
        "community|alghafa:multiple_choice_facts_truefalse_balanced_task|0",
        "community|alghafa:multiple_choice_grounded_statement_soqal_task|0",
        "community|alghafa:multiple_choice_grounded_statement_xglue_mlqa_task|0",
        "community|alghafa:multiple_choice_rating_sentiment_no_neutral_task|0",
        "community|alghafa:multiple_choice_rating_sentiment_task|0",
        "community|alghafa:multiple_choice_sentiment_task|0",
    ),
    "AraTrust": (
        "community|aratrust:Trustfulness|0",
        "community|aratrust:MentalHealth|0",
        "community|aratrust:PhysicalHealth|0",
        "community|aratrust:Offensive|0",
        "community|aratrust:Ethics|0",
        "community|aratrust:Privacy|0",
        "community|aratrust:Unfairness|0",
        "community|aratrust:Illegal|0",
    ),
}


def load_eval_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load and parse `configs/eval_config.yaml`."""
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _resolve_hf_eval_cache_dir() -> Optional[str]:
    """Return `/dev/shm/hf_eval_cache` on Linux to avoid root disk usage, or None otherwise."""
    shm_root = pathlib.Path("/dev/shm")
    if shm_root.exists() and os.access(str(shm_root), os.W_OK):
        cache_dir = shm_root / "hf_eval_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)
    return None


def _patch_gemma4_config_for_lm_eval(model: Any) -> None:
    """Expose `text_config` attributes at top level of `Gemma4Config` for `lm_eval` compatibility."""
    cfg = getattr(model, "config", None) or model
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    if cfg is not None:
        try:
            cfg.tie_word_embeddings = True
        except Exception:
            pass
    if cfg is not None and text_cfg is not None:
        try:
            text_cfg.tie_word_embeddings = True
        except Exception:
            pass
        for attr in (
            "vocab_size",
            "eos_token_id",
            "bos_token_id",
            "pad_token_id",
            "max_position_embeddings",
            "hidden_size",
        ):
            if getattr(cfg, attr, None) is None and getattr(text_cfg, attr, None) is not None:
                setattr(cfg, attr, getattr(text_cfg, attr))


def _has_separate_lm_head_in_checkpoint(model_dir: pathlib.Path) -> bool:
    """Return True only if `lm_head.weight` is explicitly stored as an independent tensor in `model_dir`."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        try:
            idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
            weight_map = idx_data.get("weight_map", {})
            return any(k in weight_map for k in ("lm_head.weight", "language_model.lm_head.weight"))
        except Exception:
            return False
    st_path = model_dir / "model.safetensors"
    if st_path.exists():
        try:
            import struct

            with st_path.open("rb") as fh:
                header_size_bytes = fh.read(8)
                if len(header_size_bytes) == 8:
                    header_len = struct.unpack("<Q", header_size_bytes)[0]
                    if 0 < header_len < 100_000_000:
                        header_json = json.loads(fh.read(header_len).decode("utf-8"))
                        return any(
                            k in header_json
                            for k in ("lm_head.weight", "language_model.lm_head.weight")
                        )
        except Exception:
            return False
    return False


def _patch_local_checkpoint_config_json(model_dir: pathlib.Path) -> bool:
    """Ensure `"tie_word_embeddings": true` is written to `model_dir / "config.json"` when `lm_head.weight` is tied."""
    cfg_path = model_dir / "config.json"
    if not (model_dir.is_dir() and cfg_path.exists()):
        return False
    if _has_separate_lm_head_in_checkpoint(model_dir):
        return False
    try:
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        changed = False
        if cfg_data.get("tie_word_embeddings") is not True:
            cfg_data["tie_word_embeddings"] = True
            changed = True
        if isinstance(cfg_data.get("text_config"), dict):
            if cfg_data["text_config"].get("tie_word_embeddings") is not True:
                cfg_data["text_config"]["tie_word_embeddings"] = True
                changed = True
        if changed:
            cfg_path.write_text(
                json.dumps(cfg_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "Patched %s on disk with tie_word_embeddings=true (root + text_config).",
                cfg_path,
            )
        return True
    except Exception as exc:
        logger.warning("Could not inspect/patch %s: %s", cfg_path, exc)
        return False


def _prepare_gemma4_config_for_load(model_path: str) -> Any:
    """Ensure `tie_word_embeddings=True` on disk (if local) and in `AutoConfig` before `from_pretrained`."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model  # type: ignore

        if not hasattr(Gemma4Model, "get_input_embeddings"):
            Gemma4Model.get_input_embeddings = (  # type: ignore[attr-defined]
                lambda self: self.language_model.get_input_embeddings()
            )
    except Exception:
        pass

    for candidate_dir in (
        pathlib.Path(model_path),
        PROJECT_ROOT / "outputs" / "merged_gemma4_e4b_arabic" / "merged_16bit",
        PROJECT_ROOT / "outputs" / "merged_gemma4_26b_a4b_arabic" / "merged_16bit",
    ):
        if candidate_dir.is_dir():
            _patch_local_checkpoint_config_json(candidate_dir)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    try:
        config.tie_word_embeddings = True
    except Exception:
        pass
    if getattr(config, "text_config", None) is not None:
        try:
            config.text_config.tie_word_embeddings = True
        except Exception:
            pass
    _patch_gemma4_config_for_lm_eval(config)
    return config


def _ensure_gemma4_tied_lm_head(model: Any, tag: str = "") -> None:
    """Guarantee `lm_head.weight` shares the exact storage pointer of `embed_tokens.weight`."""
    inner = (
        getattr(model, "model", model)
        if hasattr(model, "_model") or type(model).__name__ == "HFLM"
        else model
    )
    base = inner.get_base_model() if hasattr(inner, "get_base_model") else inner

    for obj in (model, inner, base):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            try:
                cfg.tie_word_embeddings = True
            except Exception:
                pass
            if getattr(cfg, "text_config", None) is not None:
                try:
                    cfg.text_config.tie_word_embeddings = True
                except Exception:
                    pass

    in_emb: Any = None
    if hasattr(base, "get_input_embeddings"):
        try:
            in_emb = base.get_input_embeddings()
        except Exception:
            in_emb = None
    if in_emb is None:
        lm_backbone = getattr(getattr(base, "model", None), "language_model", None)
        in_emb = getattr(lm_backbone, "embed_tokens", None) or getattr(
            getattr(base, "model", base), "embed_tokens", None
        )

    out_emb: Any = getattr(base, "lm_head", None)
    if out_emb is None and hasattr(base, "get_output_embeddings"):
        try:
            out_emb = base.get_output_embeddings()
        except Exception:
            out_emb = None
    if out_emb is None and hasattr(base, "model"):
        out_emb = getattr(base.model, "lm_head", None)

    if (
        in_emb is not None
        and out_emb is not None
        and hasattr(in_emb, "weight")
        and hasattr(out_emb, "weight")
        and in_emb.weight is not None
        and out_emb.weight is not None
    ):
        if in_emb.weight.data_ptr() != out_emb.weight.data_ptr():
            out_emb.weight = in_emb.weight
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info(
                "[%s] Explicitly tied lm_head.weight -> embed_tokens.weight (data_ptr=%d)",
                tag or "gemma4",
                in_emb.weight.data_ptr(),
            )
        assert out_emb.weight.data_ptr() == in_emb.weight.data_ptr(), (
            f"[{tag or 'gemma4'}] lm_head.weight data_ptr ({out_emb.weight.data_ptr()}) "
            f"!= embed_tokens.weight data_ptr ({in_emb.weight.data_ptr()})"
        )


def compute_effective_task_limit(
    task_name: str,
    limit: Optional[int],
) -> Optional[int]:
    """Scale `--limit` across multi-subtask `lm_eval` groups so total samples stay near `limit`."""
    if limit is None or limit <= 0:
        return None
    num_subtasks = GROUP_SUBTASK_COUNTS.get(task_name, 1)
    if num_subtasks <= 1:
        return int(limit)
    return max(4, int(math.ceil(float(limit) / float(num_subtasks))))


# ---------------------------------------------------------------------------
# MGSM-Ar Arabic-Indic Numeral & Target Extraction Helpers
# ---------------------------------------------------------------------------

_EASTERN_ARABIC_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def _normalize_arabic_digits(text: str) -> str:
    """Translate Eastern Arabic (`٠-٩`) and Extended Arabic-Indic (`۰-۹`) digits to ASCII `0-9`."""
    return str(text or "").translate(_EASTERN_ARABIC_DIGIT_MAP)


def _extract_mgsm_ar_target_number(answer: str) -> str:
    """Extract normalized ASCII target number from `Arabic-gsm8k-v2` gold `answer` strings."""
    normalized = _normalize_arabic_digits(str(answer or "")).strip()
    m = re.search(r"#+\s*(-?[\d,]+(?:\.\d+)?)", normalized)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", normalized)
    if nums:
        return nums[-1].replace(",", "")
    return normalized


def _process_mgsm_ar_docs(dataset: Any) -> Any:
    """Normalize Eastern Arabic digits and extract `target_number` for `Omartificial-Intelligence-Space/Arabic-gsm8k-v2`."""

    def _map_row(doc: Dict[str, Any]) -> Dict[str, Any]:
        raw_ans = _normalize_arabic_digits(str(doc.get("answer") or "")).strip()
        target_num = _extract_mgsm_ar_target_number(raw_ans)
        if re.search(r"#+\s*-?[\d,]+(?:\.\d+)?", raw_ans):
            norm_ans = re.sub(r"#+\s*(-?[\d,]+(?:\.\d+)?)\s*$", f"#### {target_num}", raw_ans)
        elif target_num:
            norm_ans = f"{raw_ans}\n#### {target_num}"
        else:
            norm_ans = raw_ans
        return {
            "question": str(doc.get("question") or "").strip(),
            "answer": norm_ans,
            "target_number": target_num,
        }

    return dataset.map(_map_row)


def _normalize_arabic_digits_filter(
    resps: List[Any],
    docs: List[Dict[str, Any]],
) -> List[List[str]]:
    """Custom `lm_eval` filter that converts Eastern Arabic digits (`٠-٩`) in model generations to ASCII `0-9`."""
    _ = docs
    out: List[List[str]] = []
    for inst in resps:
        if isinstance(inst, (list, tuple)):
            out.append([_normalize_arabic_digits(str(r)) for r in inst])
        else:
            out.append([_normalize_arabic_digits(str(inst))])
    return out


def build_custom_lm_eval_task_dict(task_name: str, num_fewshot: int = 0) -> Dict[str, Any]:
    """Return task configuration dictionary for `mgsm_ar` (and metadata stubs for legacy callers)."""
    key = task_name.strip().lower()
    if key == "mgsm_ar":
        return {
            "task": "mgsm_ar",
            "dataset_path": "Omartificial-Intelligence-Space/Arabic-gsm8k-v2",
            "dataset_name": "default",
            "training_split": "main_train",
            "test_split": "main_test",
            "fewshot_split": "main_train",
            "num_fewshot": int(num_fewshot),
            "output_type": "generate_until",
            "process_docs": _process_mgsm_ar_docs,
            "doc_to_text": "السؤال: {{question}}\nالإجابة:",
            "doc_to_target": "{{target_number}}",
            "fewshot_config": {
                "sampler": "default",
                "split": "main_train",
                "doc_to_target": "{{answer}}",
            },
            "generation_kwargs": {
                "until": [
                    "السؤال:",
                    "\nالسؤال",
                    "Question:",
                    "<|turn>",
                    "<turn|>",
                    "<eos>",
                    "</s>",
                    "<|im_end|>",
                ],
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 256,
            },
            "filter_list": [
                {
                    "name": "flexible-extract",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _normalize_arabic_digits_filter,
                        },
                        {
                            "function": "regex",
                            "group_select": -1,
                            "regex_pattern": "(-?[$0-9.,]{2,})|(-?[0-9]+)",
                        },
                        {"function": "take_first"},
                    ],
                },
                {
                    "name": "strict-match",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _normalize_arabic_digits_filter,
                        },
                        {
                            "function": "regex",
                            "regex_pattern": "#### (\\-?[0-9\\.\\,]+)",
                        },
                        {"function": "take_first"},
                    ],
                },
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "ignore_case": True,
                    "ignore_punctuation": True,
                    "regexes_to_ignore": [",", "\\$", "\\.$"],
                }
            ],
            "metadata": {"version": 1.0},
        }
    if key == "arabicmmlu":
        return {
            "task": "arabicmmlu",
            "dataset_path": "MBZUAI/ArabicMMLU",
            "output_type": "multiple_choice",
            "num_fewshot": int(num_fewshot),
        }
    if key == "madinahqa":
        return {
            "task": "madinahqa",
            "dataset_path": "MBZUAI/MadinahQA",
            "output_type": "multiple_choice",
            "num_fewshot": int(num_fewshot),
        }
    if key == "aratrust":
        return {
            "task": "aratrust",
            "dataset_path": "asas-ai/AraTrust",
            "output_type": "multiple_choice",
            "num_fewshot": int(num_fewshot),
        }
    raise KeyError(f"Unknown custom task dictionary key: '{task_name}'")


def resolve_lm_eval_task_specs(
    task_name: str,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
) -> Tuple[List[Union[str, Dict[str, Any]]], str, Optional[int]]:
    """Resolve a benchmark's `lm_eval_task` into `(tasks_arg, primary_result_key, effective_limit)`.

    Uses native built-in `lm_eval` tasks for `arabicmmlu`, `arabic_leaderboard_alghafa_light`,
    `arabic_leaderboard_acva_light`, `arabic_exams`, `arc_challenge`, `gsm8k`, `humaneval`,
    and `leaderboard_mmlu_pro`.
    """
    clean_name = task_name.strip()
    eff_limit = compute_effective_task_limit(clean_name, limit)

    if clean_name.lower() == "mgsm_ar":
        custom_dict = build_custom_lm_eval_task_dict(clean_name, num_fewshot=num_fewshot)
        return ([custom_dict], str(custom_dict["task"]), eff_limit)

    if clean_name == "arabic_leaderboard_acva_light" and limit is not None and limit > 0:
        return (list(ACVA_CORE_SUBTASKS), clean_name, eff_limit)

    return ([clean_name], clean_name, eff_limit)


def evaluate_retention_gate(
    anchor_results: Dict[str, Dict[str, Any]],
    min_retention_ratio: float = 0.998,
    max_allowed_drop_pct: float = 0.20,
    sweep_lambdas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Verify zero catastrophic forgetting across all 4 English/Math/Code anchors."""
    violations: List[str] = []
    ratios: List[float] = []

    for name, entry in anchor_results.items():
        base = float(entry["baseline_score"])
        tuned = float(entry["evaluated_score"])
        delta = round(tuned - base, 4)
        ratio = round(tuned / base, 6) if base > 0 else 1.0
        ratios.append(ratio)

        allowed_drop = float(entry.get("max_regression_pct", max_allowed_drop_pct))
        n_samples = int(entry.get("n_samples", 0) or 0)
        if 0 < n_samples <= 250:
            single_item_pct = 100.0 / float(n_samples)
            effective_allowed_drop = max(allowed_drop, single_item_pct + 0.05)
            effective_min_ratio = (
                min(min_retention_ratio, max(0.0, (base - effective_allowed_drop) / base))
                if base > 0
                else min_retention_ratio
            )
        else:
            effective_allowed_drop = allowed_drop
            effective_min_ratio = min_retention_ratio

        passed = (delta >= -effective_allowed_drop) and (ratio >= effective_min_ratio)
        entry["delta"] = delta
        entry["retention_ratio"] = ratio
        entry["passed"] = passed
        if not passed:
            violations.append(name)

    min_observed_ratio = min(ratios) if ratios else 1.0
    gate_passed = len(violations) == 0
    recommended_lambda = 0.85 if gate_passed else (
        min(sweep_lambdas) if sweep_lambdas else 0.80
    )

    return {
        "gate_passed": gate_passed,
        "min_observed_retention_ratio": round(min_observed_ratio, 6),
        "required_min_retention_ratio": min_retention_ratio,
        "max_allowed_drop_pct": max_allowed_drop_pct,
        "violated_anchors": violations,
        "recommended_ties_lambda": recommended_lambda,
    }


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Render a clean GitHub-Flavored Markdown report using only standard ASCII punctuation."""
    lines: List[str] = [
        "# Gemma-4-Arabic Family Benchmark Evaluation Report",
        "",
        f"- **Model Evaluated:** `{report_data['model_path']}`",
        f"- **Evaluation Mode:** `{report_data['mode']}`",
        f"- **Zero-Forgetting Retention Gate:** `{'PASSED' if report_data['retention_gate']['gate_passed'] else 'FAILED'}` "
        f"(Min Retention Ratio: `{report_data['retention_gate']['min_observed_retention_ratio']:.4f}`, "
        f"Threshold: `{report_data['retention_gate']['required_min_retention_ratio']:.4f}`)",
        "",
        f"## 1. Primary Arabic and Saudi Benchmarks ({len(report_data['arabic_benchmarks'])} Suites)",
        "",
        "| Benchmark | Task (`lm_eval` / `lighteval`) | Metric | Shots | Base NLL | Tuned NLL | Baseline (%) | Evaluated (%) | Gain (% pts) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, item in report_data["arabic_benchmarks"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        gain = ev - base
        sign = "+" if gain >= 0 else ""
        b_nll = f"{float(item['base_nll']):.4f}" if item.get("base_nll") is not None else "N/A"
        t_nll = f"{float(item['tuned_nll']):.4f}" if item.get("tuned_nll") is not None else "N/A"
        lines.append(
            f"| **{name}** | `{item['lm_eval_task']}` | `{item['metric']}` | "
            f"{item['num_fewshot']} | `{b_nll}` | `{t_nll}` | {base:.2f} | **{ev:.2f}** | "
            f"`{sign}{gain:.2f}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Zero-Forgetting Retention Gate Anchors (4 Suites)",
            "",
            "| Anchor | Domain | Task (`lm_eval`) | Shots | Base NLL | Tuned NLL | Baseline (%) | Evaluated (%) | Delta (% pts) | Retention Ratio |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for name, item in report_data["retention_anchors"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        delta = float(item["delta"])
        ratio = float(item["retention_ratio"])
        sign = "+" if delta >= 0 else ""
        b_nll = f"{float(item['base_nll']):.4f}" if item.get("base_nll") is not None else "N/A"
        t_nll = f"{float(item['tuned_nll']):.4f}" if item.get("tuned_nll") is not None else "N/A"
        lines.append(
            f"| **{name}** | `{item['domain']}` | `{item['lm_eval_task']}` | "
            f"{item['num_fewshot']} | `{b_nll}` | `{t_nll}` | {base:.2f} | **{ev:.2f}** | "
            f"`{sign}{delta:.2f}` | `{ratio:.4f}` |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_lm_eval_results_json(
    results_payload: Dict[str, Any],
    task_name: str,
    metric_hint: str = "acc_norm",
) -> float:
    """Extract the primary percentage score (`0.0` to `100.0`) from an `lm_eval` JSON dict."""
    results_block = results_payload.get("results", {})
    if not isinstance(results_block, dict) or not results_block:
        raise ValueError(f"lm_eval results JSON has no 'results' entries for task '{task_name}'.")

    candidate_keys = [
        f"{metric_hint},none",
        f"{metric_hint},flexible-extract",
        f"{metric_hint},strict-match",
        f"{metric_hint},create_test",
        metric_hint,
        "acc_norm,none",
        "acc,none",
        "exact_match,custom-extract",
        "exact_match,remove_whitespace",
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
        "pass@1,create_test",
        "pass@1,none",
        "acc_norm",
        "acc",
        "exact_match",
        "pass@1",
    ]

    task_metrics = results_block.get(task_name)
    if task_metrics is None:
        subtask_dicts = [v for v in results_block.values() if isinstance(v, dict)]
        if not subtask_dicts:
            raise ValueError(f"No valid metric dictionary found in lm_eval results for '{task_name}'.")
        subtask_scores: List[float] = []
        for sub_d in subtask_dicts:
            for key in candidate_keys:
                if key in sub_d and isinstance(sub_d[key], (int, float)):
                    val = float(sub_d[key])
                    subtask_scores.append(val * 100.0 if val <= 1.0 else val)
                    break
        if subtask_scores:
            return round(sum(subtask_scores) / len(subtask_scores), 2)
        task_metrics = subtask_dicts[0]

    for key in candidate_keys:
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            raw_val = float(task_metrics[key])
            return round(raw_val * 100.0 if raw_val <= 1.0 else raw_val, 2)

    for key, val in task_metrics.items():
        if "stderr" not in str(key) and isinstance(val, (int, float)):
            raw_val = float(val)
            return round(raw_val * 100.0 if raw_val <= 1.0 else raw_val, 2)

    raise ValueError(f"Could not locate numeric metric '{metric_hint}' in lm_eval output: {task_metrics}")


def extract_lm_eval_detailed_metrics(
    results_payload: Dict[str, Any],
    task_name: str,
    metric_hint: str = "acc_norm",
) -> Dict[str, Any]:
    """Extract `eval_score_pct`, `top1_acc_pct` (unnormalized `acc`), `nll` (`GoldNLL`), and `n_samples`."""
    eval_score_pct = parse_lm_eval_results_json(
        results_payload=results_payload,
        task_name=task_name,
        metric_hint=metric_hint,
    )
    try:
        top1_acc_pct = parse_lm_eval_results_json(
            results_payload=results_payload,
            task_name=task_name,
            metric_hint="acc",
        )
    except Exception:
        top1_acc_pct = eval_score_pct

    samples_block = results_payload.get("samples", {})
    gold_nlls: List[float] = []
    n_samples = 0

    if isinstance(samples_block, dict) and samples_block:
        for _subtask, sample_list in samples_block.items():
            if not isinstance(sample_list, list):
                continue
            n_samples += len(sample_list)
            for sample in sample_list:
                if not isinstance(sample, dict):
                    continue
                target = sample.get("target")
                resps = sample.get("resps")
                if isinstance(target, int) and isinstance(resps, list) and 0 <= target < len(resps):
                    choice_resp = resps[target]
                    while isinstance(choice_resp, (list, tuple)) and len(choice_resp) > 0:
                        if isinstance(choice_resp[0], (int, float)):
                            ll = float(choice_resp[0])
                            if math.isfinite(ll):
                                gold_nlls.append(max(0.0, -ll))
                            break
                        choice_resp = choice_resp[0]

    if n_samples == 0:
        n_samples_block = results_payload.get("n-samples", {})
        if isinstance(n_samples_block, dict):
            for v in n_samples_block.values():
                if isinstance(v, dict) and "effective" in v:
                    n_samples += int(v["effective"])
                elif isinstance(v, int):
                    n_samples += v

    mean_nll: Optional[float] = (
        round(sum(gold_nlls) / len(gold_nlls), 4) if gold_nlls else None
    )
    return {
        "eval_score_pct": eval_score_pct,
        "raw_signal_pct": eval_score_pct,
        "top1_acc_pct": top1_acc_pct,
        "nll": mean_nll,
        "ppl": round(math.exp(min(mean_nll, 12.0)), 2) if mean_nll is not None else None,
        "n_samples": n_samples,
    }


def parse_lighteval_results_json(
    lighteval_payload: Dict[str, Any],
    suite_name: str,
) -> Dict[str, Any]:
    """Extract percentage score (`0.0` to `100.0`) for OALL v2 Arabic benchmarks from `lighteval` results."""
    results_block = lighteval_payload.get("results", {})
    if not isinstance(results_block, dict) or not results_block:
        raise ValueError(f"LightEval payload has no 'results' entries for '{suite_name}'.")

    prefix_map = {
        "arabicmmlu": "arabic_mmlu",
        "madinahqa": "madinah_qa",
        "alghafa": "alghafa",
        "acva": "acva",
        "exams-ar": "arabic_exams",
        "aratrust": "aratrust",
    }
    prefix = prefix_map.get(suite_name.strip().lower(), suite_name.strip().lower())
    avg_candidates = (
        f"community|{prefix}:_average|0",
        f"community:{prefix}:_average:0",
        f"lighteval|{prefix}:_average|0",
        f"lighteval:{prefix}:_average:0",
        f"{prefix}:_average|0",
        f"{prefix}:_average:0",
        f"community|{prefix}:All|0",
        f"community:{prefix}:All:0",
        f"{prefix}:All|0",
        f"{prefix}:All:0",
        f"community|{prefix}|0",
        f"community:{prefix}:0",
        f"{prefix}|0",
        f"{prefix}:0",
    )

    matched_metrics: List[Dict[str, Any]] = []
    for avg_key in avg_candidates:
        if avg_key in results_block and isinstance(results_block[avg_key], dict):
            matched_metrics = [results_block[avg_key]]
            break

    if not matched_metrics:
        for k, v in results_block.items():
            if prefix in str(k).lower() and "_average" in str(k).lower() and isinstance(v, dict):
                matched_metrics = [v]
                break

    if not matched_metrics:
        for k, v in results_block.items():
            if prefix in str(k).lower() and isinstance(v, dict):
                matched_metrics.append(v)

    if not matched_metrics:
        raise ValueError(f"Could not find '{prefix}' task scores in LightEval results: {list(results_block.keys())}")

    scores: List[float] = []
    for mdict in matched_metrics:
        found = False
        for mkey in ("acc_norm", "loglikelihood_acc_norm", "acc", "loglikelihood_acc", "accuracy"):
            if mkey in mdict and isinstance(mdict[mkey], (int, float)):
                val = float(mdict[mkey])
                scores.append(val * 100.0 if val <= 1.0 else val)
                found = True
                break
        if not found:
            for mkey, mval in mdict.items():
                if (
                    str(mkey).startswith("acc")
                    and not str(mkey).endswith("_stderr")
                    and isinstance(mval, (int, float))
                ):
                    val = float(mval)
                    scores.append(val * 100.0 if val <= 1.0 else val)
                    found = True
                    break
        if not found:
            for mkey, mval in mdict.items():
                if "stderr" not in str(mkey) and isinstance(mval, (int, float)):
                    val = float(mval)
                    scores.append(val * 100.0 if val <= 1.0 else val)
                    break

    if not scores:
        raise ValueError(f"No numeric metric found for '{suite_name}' in LightEval results: {matched_metrics}")

    mean_score = round(sum(scores) / len(scores), 2)
    return {
        "eval_score_pct": mean_score,
        "raw_signal_pct": mean_score,
        "top1_acc_pct": mean_score,
        "nll": None,
        "ppl": None,
        "n_samples": 0,
    }


def _destroy_vllm_and_free_vram(lm_obj: Any = None, model_obj: Any = None) -> None:
    """Shut down any active vLLM engine / distributed group and release GPU VRAM before starting the next runner."""
    try:
        if lm_obj is not None:
            inner_vllm = getattr(lm_obj, "model", None)
            if inner_vllm is not None:
                try:
                    inner_vllm.llm_engine.engine_core.shutdown()
                except Exception:
                    pass
                del inner_vllm
            del lm_obj
    except Exception:
        pass
    try:
        if model_obj is not None:
            del model_obj
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        import ray  # type: ignore

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_lighteval_tasks_subprocess(
    model_path: str,
    suite_names: List[str],
    device: str = "cuda:0",
    limit: Optional[int] = None,
    max_seq_len: int = 4096,
    batch_size: int = 16,
    use_vllm: bool = True,
    tag: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Run `MadinahQA` and `AraTrust` via `lighteval` in an isolated subprocess and return parsed metrics."""
    if not suite_names:
        return {}

    task_strings: List[str] = []
    for sname in suite_names:
        task_strings.extend(LIGHTEVAL_SUITE_TASK_MAP.get(sname, ()))
    if not task_strings:
        return {}

    model_dir = pathlib.Path(model_path)
    load_target_path = model_path
    if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        load_target_path = str(adapter_cfg.get("base_model_name_or_path") or model_path).strip()
    if pathlib.Path(load_target_path).is_dir():
        _patch_local_checkpoint_config_json(pathlib.Path(load_target_path))

    # Scale per-subtask max_samples across subtasks when --limit is provided
    per_subtask_samples: Optional[int] = None
    if limit is not None and limit > 0:
        per_subtask_samples = max(10, int(math.ceil(float(limit) / 2.0)))

    tasks_csv = ",".join(task_strings)
    logger.info(
        "[%s] Running LightEval on %s (backend=%s, max_samples=%s)...",
        tag,
        ", ".join(suite_names),
        "vllm" if use_vllm else "accelerate",
        str(per_subtask_samples),
    )

    child_env = os.environ.copy()
    if ":" in device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_idx = device.split(":")[1]
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")

    with tempfile.TemporaryDirectory(prefix="lighteval_run_") as tmp_dir:
        out_json_file = pathlib.Path(tmp_dir) / "lighteval_results.json"
        runner_code = f"""
import json, logging, os, pathlib, sys
for _n in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
sys.path.insert(0, {repr(str(PROJECT_ROOT))})
from scripts.run_benchmarks import _prepare_gemma4_config_for_load
try:
    _prepare_gemma4_config_for_load({repr(load_target_path)})
except Exception:
    pass

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

evaluation_tracker = EvaluationTracker(
    output_dir={repr(str(tmp_dir))},
    save_details=False,
    push_to_hub=False,
)

use_vllm = {repr(bool(use_vllm))}
results_payload = None

if use_vllm:
    try:
        from lighteval.models.vllm.vllm_model import VLLMModelConfig
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.VLLM,
            custom_tasks_directory="lighteval.tasks.multilingual.tasks.arabic",
            load_tasks_multilingual=False,
            max_samples={repr(per_subtask_samples)},
            bootstrap_iters=0,
        )
        model_config = VLLMModelConfig(
            model_name={repr(load_target_path)},
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_length={int(max_seq_len)},
            trust_remote_code=True,
        )
        pipeline = Pipeline(
            tasks={repr(tasks_csv)},
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )
        pipeline.evaluate()
        results_payload = pipeline.get_results()
    except Exception as vllm_exc:
        sys.stderr.write(f"[LightEval] vLLM backend failed ({{vllm_exc}}); falling back to Accelerate...\\n")

if results_payload is None:
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="lighteval.tasks.multilingual.tasks.arabic",
        load_tasks_multilingual=False,
        max_samples={repr(per_subtask_samples)},
        bootstrap_iters=0,
    )
    model_config = TransformersModelConfig(
        model_name={repr(load_target_path)},
        dtype="bfloat16",
        batch_size={int(batch_size)},
        max_length={int(max_seq_len)},
        trust_remote_code=True,
    )
    pipeline = Pipeline(
        tasks={repr(tasks_csv)},
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    pipeline.evaluate()
    results_payload = pipeline.get_results()

pathlib.Path({repr(str(out_json_file))}).write_text(
    json.dumps({{"results": results_payload.get("results", {{}})}}, indent=2),
    encoding="utf-8",
)
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_json_file.exists():
            err_tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise RuntimeError(
                f"[{tag}] LightEval subprocess failed (exit={proc.returncode}): {err_tail}"
            )
        raw_lighteval = json.loads(out_json_file.read_text(encoding="utf-8"))

    parsed_out: Dict[str, Dict[str, Any]] = {}
    for sname in suite_names:
        metrics = parse_lighteval_results_json(raw_lighteval, sname)
        metrics["n_samples"] = int(limit or 0)
        parsed_out[sname] = metrics
        logger.info(
            "[%s] arabic/%s (lighteval) -> acc_norm=%.2f%% (N=%d)",
            tag,
            sname,
            metrics["eval_score_pct"],
            metrics["n_samples"],
        )
    return parsed_out


def run_lm_eval_task(
    model_path: str,
    task_name: str,
    num_fewshot: int,
    batch_size: int = 8,
    device: str = "cuda:0",
    limit: Optional[int] = None,
    metric_hint: str = "acc_norm",
    use_vllm: bool = False,
    apply_chat_template: bool = False,
) -> float:
    """Run a single `lm_eval` task via `lm_eval.simple_evaluate` and return the score."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    tasks_arg, primary_key, eff_limit = resolve_lm_eval_task_specs(
        task_name=task_name,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    if pathlib.Path(model_path).is_dir():
        try:
            _prepare_gemma4_config_for_load(model_path)
        except Exception:
            pass
    model_backend = "vllm" if use_vllm else "hf"
    model_args = (
        f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,"
        f"gpu_memory_utilization=0.85,max_model_len=4096"
        if use_vllm
        else f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,tie_word_embeddings=True"
    )
    gen_kwargs = "max_gen_toks=512" if task_name.strip() == "mmlu_pro" else None
    payload = lm_eval.simple_evaluate(
        model=model_backend,
        model_args=model_args,
        tasks=tasks_arg,
        num_fewshot=num_fewshot,
        batch_size="auto" if use_vllm else batch_size,
        device=device,
        limit=eff_limit,
        gen_kwargs=gen_kwargs,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        confirm_run_unsafe_code=True,
        apply_chat_template=apply_chat_template,
        bootstrap_iters=0,
        task_manager=TaskManager(),
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"lm_eval.simple_evaluate returned empty payload for task '{task_name}'.")
    return parse_lm_eval_results_json(
        results_payload=payload,
        task_name=primary_key,
        metric_hint=metric_hint,
    )


def _run_lm_eval_suites_subprocess(
    model_path: str,
    device: str,
    cfg: Dict[str, Any],
    tag: str,
    limit: Optional[int] = None,
    use_vllm: bool = True,
    apply_chat_template: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the `lm_eval` tasks for `model_path` in an isolated child subprocess so the OS reclaims 100% of vLLM's 85% GPU memory pool upon completion."""
    child_env = os.environ.copy()
    if ":" in device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_idx = device.split(":")[1]
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    child_env.setdefault("HF_ALLOW_CODE_EVAL", "1")

    with tempfile.TemporaryDirectory(prefix="lm_eval_run_") as tmp_dir:
        cfg_json_file = pathlib.Path(tmp_dir) / "eval_cfg.json"
        out_json_file = pathlib.Path(tmp_dir) / "lm_eval_out.json"
        cfg_json_file.write_text(json.dumps(cfg), encoding="utf-8")

        runner_code = f"""
import json, logging, pathlib, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
for _n in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
sys.path.insert(0, {repr(str(PROJECT_ROOT))})
from scripts.run_benchmarks import _evaluate_checkpoint_with_lm_eval

cfg = json.loads(pathlib.Path({repr(str(cfg_json_file))}).read_text(encoding="utf-8"))
res = _evaluate_checkpoint_with_lm_eval(
    model_path={repr(str(model_path))},
    device="cuda:0",
    cfg=cfg,
    tag={repr(str(tag))},
    limit={repr(limit)},
    use_vllm={repr(bool(use_vllm))},
    apply_chat_template={repr(bool(apply_chat_template))},
    _in_lm_eval_subprocess=True,
    _skip_lighteval=True,
)
pathlib.Path({repr(str(out_json_file))}).write_text(
    json.dumps(res, indent=2),
    encoding="utf-8",
)
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
        )
        if proc.returncode != 0 or not out_json_file.exists():
            raise RuntimeError(
                f"[{tag}] lm_eval child subprocess failed with exit code {proc.returncode}"
            )
        return json.loads(out_json_file.read_text(encoding="utf-8"))


def _evaluate_checkpoint_with_lm_eval(
    model_path: str,
    device: str,
    cfg: Dict[str, Any],
    tag: str,
    limit: Optional[int] = None,
    use_vllm: bool = True,
    apply_chat_template: bool = False,
    _in_lm_eval_subprocess: bool = False,
    _skip_lighteval: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Evaluate a checkpoint using native `lm_eval` (8 tasks + MGSM-Ar) and native `lighteval` (MadinahQA + AraTrust)."""
    eval_cfg = cfg.get("evaluation", {})
    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})
    model_key_prefix = tag.split(":")[0] if ":" in tag else "e4b"
    model_specific_cfg = cfg.get("models", {}).get(model_key_prefix, {})
    default_bs = 8 if model_key_prefix == "26b_a4b" else int(eval_cfg.get("batch_size", 32))
    batch_size = int(model_specific_cfg.get("batch_size", default_bs))
    max_seq_len = int(eval_cfg.get("max_seq_length", 4096))

    lighteval_suites: List[str] = [
        sname
        for sname, spec in ar_defs.items()
        if str(spec.get("runner", "lm_eval")).strip().lower() == "lighteval"
        and sname in LIGHTEVAL_SUITE_TASK_MAP
    ]

    if use_vllm and isinstance(model_path, str) and not _in_lm_eval_subprocess:
        out = _run_lm_eval_suites_subprocess(
            model_path=model_path,
            device=device,
            cfg=cfg,
            tag=tag,
            limit=limit,
            use_vllm=use_vllm,
            apply_chat_template=apply_chat_template,
        )
        if lighteval_suites and not _skip_lighteval:
            le_metrics = run_lighteval_tasks_subprocess(
                model_path=model_path,
                suite_names=lighteval_suites,
                device=device,
                limit=limit,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                use_vllm=use_vllm,
                tag=tag,
            )
            for sname, smetrics in le_metrics.items():
                out["arabic"][sname] = smetrics
        return out

    import lm_eval
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    cache_dir = _resolve_hf_eval_cache_dir()
    if cache_dir:
        os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

    _prepare_gemma4_config_for_load(
        str(
            json.loads((pathlib.Path(model_path) / "adapter_config.json").read_text(encoding="utf-8")).get(
                "base_model_name_or_path", model_path
            )
        )
        if (pathlib.Path(model_path) / "adapter_config.json").exists()
        and not (pathlib.Path(model_path) / "config.json").exists()
        else model_path
    )

    logger.info(
        "[%s] Initializing lm_eval model (%s) on %s: %s",
        tag,
        "vLLM" if use_vllm else f"HFLM (batch_size={batch_size})",
        device,
        model_path,
    )
    model_obj: Any = None
    lm_obj: Any = None
    effective_use_vllm = bool(use_vllm)

    if effective_use_vllm:
        try:
            from lm_eval.models.vllm_causallms import VLLM

            model_dir = pathlib.Path(model_path)
            if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
                adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
                base_model_path = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
                lm_obj = VLLM(
                    pretrained=base_model_path,
                    lora_local_path=str(model_path),
                    dtype="bfloat16",
                    trust_remote_code=True,
                    gpu_memory_utilization=0.85,
                    max_model_len=max_seq_len,
                    batch_size="auto",
                )
            else:
                lm_obj = VLLM(
                    pretrained=model_path,
                    dtype="bfloat16",
                    trust_remote_code=True,
                    gpu_memory_utilization=0.85,
                    max_model_len=max_seq_len,
                    batch_size="auto",
                )
        except Exception as vllm_exc:
            logger.warning(
                "[%s] vLLM initialization failed (%s); falling back to HFLM (batch_size=%d).",
                tag,
                vllm_exc,
                batch_size,
            )
            _destroy_vllm_and_free_vram()
            effective_use_vllm = False

    if not effective_use_vllm:
        model_dir = pathlib.Path(model_path)
        is_peft_adapter = (model_dir / "adapter_config.json").exists() and not (
            model_dir / "config.json"
        ).exists()
        load_target_path = model_path
        if is_peft_adapter:
            adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
            load_target_path = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
            if not load_target_path:
                raise ValueError(
                    f"Unmerged PEFT adapter directory '{model_path}' is missing 'base_model_name_or_path' in adapter_config.json"
                )
            logger.info(
                "[%s] Detected unmerged PEFT adapter at %s -> loading base model %s",
                tag,
                model_path,
                load_target_path,
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(load_target_path, trust_remote_code=True)

        hf_config = _prepare_gemma4_config_for_load(load_target_path)
        try:
            model_obj = AutoModelForImageTextToText.from_pretrained(
                load_target_path,
                config=hf_config,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
            )
        except Exception:
            model_obj = AutoModelForCausalLM.from_pretrained(
                load_target_path,
                config=hf_config,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
            )
        _ensure_gemma4_tied_lm_head(model_obj, tag=tag)
        if is_peft_adapter:
            from peft import PeftModel

            model_obj = PeftModel.from_pretrained(model_obj, model_path)
            _ensure_gemma4_tied_lm_head(model_obj, tag=f"{tag}:peft")
        _patch_gemma4_config_for_lm_eval(model_obj)
        model_obj.eval()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lm_obj = HFLM(
            pretrained=model_obj,
            tokenizer=tokenizer,
            backend="causal",
            batch_size=batch_size,
            max_length=max_seq_len,
            trust_remote_code=True,
        )
        _ensure_gemma4_tied_lm_head(lm_obj, tag=f"{tag}:hflm")

    task_manager = TaskManager()
    out: Dict[str, Dict[str, Dict[str, Any]]] = {"arabic": {}, "retention": {}}

    suite_groups = (
        ("arabic", ar_defs),
        ("retention", ret_defs),
    )
    for cat, suite_defs in suite_groups:
        for suite_name, spec in suite_defs.items():
            runner = str(spec.get("runner", "lm_eval")).strip().lower()
            if runner == "lighteval" and suite_name in LIGHTEVAL_SUITE_TASK_MAP:
                continue

            task_name = str(spec["lm_eval_task"])
            num_fewshot = int(spec.get("num_fewshot", 0))
            metric_hint = str(spec.get("metric", "acc_norm"))
            tasks_arg, primary_key, eff_limit = resolve_lm_eval_task_specs(
                task_name=task_name,
                num_fewshot=num_fewshot,
                limit=limit,
            )
            logger.info(
                "[%s] Running lm_eval on %s/%s (task=%s, shots=%d, limit=%s, metric=%s)...",
                tag,
                cat,
                suite_name,
                task_name,
                num_fewshot,
                str(eff_limit),
                metric_hint,
            )
            gen_kwargs = "max_gen_toks=512" if task_name.strip() == "mmlu_pro" else None
            payload = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=tasks_arg,
                num_fewshot=num_fewshot,
                batch_size="auto" if effective_use_vllm else batch_size,
                limit=eff_limit,
                gen_kwargs=gen_kwargs,
                random_seed=42,
                numpy_random_seed=42,
                torch_random_seed=42,
                confirm_run_unsafe_code=True,
                apply_chat_template=apply_chat_template,
                log_samples=True,
                bootstrap_iters=0,
                task_manager=task_manager,
            )
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"[{tag}] lm_eval.simple_evaluate returned None for {cat}/{suite_name} ({task_name})."
                )
            metrics = extract_lm_eval_detailed_metrics(
                results_payload=payload,
                task_name=primary_key,
                metric_hint=metric_hint,
            )
            out[cat][suite_name] = metrics
            nll_display = f"{metrics['nll']:.4f}" if metrics.get("nll") is not None else "N/A"
            logger.info(
                "[%s] %s/%s -> %s=%.2f%% | HardTop1=%.2f%% | GoldNLL=%s (N=%d)",
                tag,
                cat,
                suite_name,
                metric_hint,
                metrics["eval_score_pct"],
                metrics["top1_acc_pct"],
                nll_display,
                metrics["n_samples"],
            )

    # Cleanly destroy the lm_eval vLLM/HFLM instance before starting LightEval on the same GPU
    _destroy_vllm_and_free_vram(lm_obj=lm_obj, model_obj=model_obj)
    lm_obj = None
    model_obj = None

    if lighteval_suites and not _skip_lighteval:
        le_metrics = run_lighteval_tasks_subprocess(
            model_path=model_path,
            suite_names=lighteval_suites,
            device=device,
            limit=limit,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            use_vllm=use_vllm,
            tag=tag,
        )
        for sname, smetrics in le_metrics.items():
            out["arabic"][sname] = smetrics

    return out


def run_inprocess_gpu_eval(
    model_key: str,
    base_model_id: str,
    tuned_model_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    target_device: str = "cuda:0",
    limit: Optional[int] = None,
    use_cached_eval: bool = False,
    use_vllm: bool = True,
    apply_chat_template: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run standardized `lm_eval` + `lighteval` GPU evaluation across all 11 benchmarks for `base` and `tuned`."""
    cache_path = PROJECT_ROOT / "outputs" / "eval_reports" / f"live_raw_{model_key}.json"
    if use_cached_eval and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("eval_version") == EVAL_CACHE_VERSION
                and "base" in cached
                and "tuned" in cached
            ):
                logger.info("Loaded verified %s cache from %s", EVAL_CACHE_VERSION, cache_path)
                return cached
        except Exception as exc:
            logger.warning("Could not read cached live_raw_%s.json: %s", model_key, exc)

    if cfg is None:
        cfg = load_eval_config(PROJECT_ROOT / "configs" / "eval_config.yaml")

    import torch

    if not torch.cuda.is_available():
        return None

    use_dev = target_device
    if ":" in use_dev:
        idx = int(use_dev.split(":")[1])
        if idx >= torch.cuda.device_count():
            use_dev = "cuda:0"

    base_metrics = _evaluate_checkpoint_with_lm_eval(
        model_path=base_model_id,
        device=use_dev,
        cfg=cfg,
        tag=f"{model_key}:base",
        limit=limit,
        use_vllm=use_vllm,
        apply_chat_template=apply_chat_template,
    )
    tuned_metrics = _evaluate_checkpoint_with_lm_eval(
        model_path=tuned_model_path,
        device=use_dev,
        cfg=cfg,
        tag=f"{model_key}:tuned",
        limit=limit,
        use_vllm=use_vllm,
        apply_chat_template=apply_chat_template,
    )

    payload = {
        "eval_version": EVAL_CACHE_VERSION,
        "model_key": model_key,
        "base_model_id": base_model_id,
        "tuned_model_path": tuned_model_path,
        "device": use_dev,
        "limit": limit,
        "base": base_metrics,
        "tuned": tuned_metrics,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %s live GPU evaluation metrics to %s", EVAL_CACHE_VERSION, cache_path)
    return payload


def build_benchmark_results(
    cfg: Dict[str, Any],
    model_path: str,
    dry_run: bool = False,
    limit: Optional[int] = None,
    model_key: str = "e4b",
    eval_base: bool = False,
    use_cached_eval: bool = False,
    no_reference_fallback: bool = True,
    use_vllm: bool = False,
    apply_chat_template: bool = False,
) -> Dict[str, Any]:
    """Run all 7 Arabic benchmarks (`lm_eval` + `lighteval`) and 4 retention anchors (`lm_eval`)."""
    eval_cfg = cfg.get("evaluation", {})
    gate_cfg = cfg.get("retention_gate", {})
    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})

    models_cfg = cfg.get("models", {})
    model_entry = models_cfg.get(model_key, {}) if isinstance(models_cfg, dict) else {}
    base_model_id = str(model_entry.get("base_model_id", model_path))
    target_device = str(model_entry.get("device", eval_cfg.get("device", "cuda:0")))

    live_gpu_data: Optional[Dict[str, Any]] = None
    if not dry_run:
        live_gpu_data = run_inprocess_gpu_eval(
            model_key=model_key,
            base_model_id=base_model_id,
            tuned_model_path=model_path,
            cfg=cfg,
            target_device=target_device,
            limit=limit,
            use_cached_eval=use_cached_eval,
            use_vllm=use_vllm,
            apply_chat_template=apply_chat_template,
        )

    arabic_results: Dict[str, Dict[str, Any]] = {}
    for name, spec in ar_defs.items():
        base_score = float(spec.get("baseline_score", 0.0))
        base_nll: Optional[float] = None
        tuned_nll: Optional[float] = None

        if dry_run:
            evaluated_score = base_score
        elif (
            live_gpu_data is not None
            and name in live_gpu_data.get("base", {}).get("arabic", {})
            and name in live_gpu_data.get("tuned", {}).get("arabic", {})
        ):
            b_item = live_gpu_data["base"]["arabic"][name]
            t_item = live_gpu_data["tuned"]["arabic"][name]
            base_nll = float(b_item["nll"]) if b_item.get("nll") is not None else None
            tuned_nll = float(t_item["nll"]) if t_item.get("nll") is not None else None
            base_score = round(float(b_item["eval_score_pct"]), 2)
            evaluated_score = round(float(t_item["eval_score_pct"]), 2)
        else:
            if eval_base or base_score == 0.0 or no_reference_fallback:
                try:
                    base_score = run_lm_eval_task(
                        model_path=str(base_model_id),
                        task_name=str(spec["lm_eval_task"]),
                        num_fewshot=int(spec.get("num_fewshot", 0)),
                        batch_size=int(eval_cfg.get("batch_size", 16)),
                        device=target_device,
                        limit=limit,
                        metric_hint=str(spec.get("metric", "acc_norm")),
                        use_vllm=use_vllm,
                        apply_chat_template=apply_chat_template,
                    )
                except Exception as exc:
                    if no_reference_fallback:
                        raise RuntimeError(
                            f"Base task '{name}' ({spec['lm_eval_task']}) failed and "
                            f"--no-reference-fallback is active: {exc}"
                        ) from exc
                    logger.warning("Base eval fallback for %s (%s): %s", name, spec["lm_eval_task"], exc)
            try:
                evaluated_score = run_lm_eval_task(
                    model_path=model_path,
                    task_name=str(spec["lm_eval_task"]),
                    num_fewshot=int(spec.get("num_fewshot", 0)),
                    batch_size=int(eval_cfg.get("batch_size", 16)),
                    device=target_device,
                    limit=limit,
                    metric_hint=str(spec.get("metric", "acc_norm")),
                    use_vllm=use_vllm,
                    apply_chat_template=apply_chat_template,
                )
            except Exception as exc:
                if no_reference_fallback:
                    raise RuntimeError(
                        f"Tuned task '{name}' ({spec['lm_eval_task']}) failed and "
                        f"--no-reference-fallback is active: {exc}"
                    ) from exc
                logger.warning(
                    "Live task '%s' (%s) failed: %s",
                    name,
                    spec["lm_eval_task"],
                    exc,
                )
                evaluated_score = base_score

        entry_dict: Dict[str, Any] = {
            "lm_eval_task": spec["lm_eval_task"],
            "lighteval_task": spec.get("lighteval_task", ""),
            "runner": spec.get("runner", "lm_eval"),
            "metric": spec["metric"],
            "num_fewshot": int(spec.get("num_fewshot", 0)),
            "baseline_score": base_score,
            "evaluated_score": evaluated_score,
            "gain": round(evaluated_score - base_score, 4),
        }
        if base_nll is not None and tuned_nll is not None:
            entry_dict["base_nll"] = round(base_nll, 4)
            entry_dict["tuned_nll"] = round(tuned_nll, 4)
        arabic_results[name] = entry_dict

    anchor_results: Dict[str, Dict[str, Any]] = {}
    for name, spec in ret_defs.items():
        base_score = float(spec.get("baseline_score", spec.get("min_acceptable_score", 0.0)))
        base_nll = None
        tuned_nll = None
        n_samples_anchor = int(limit or 0) if limit else 0

        if dry_run:
            evaluated_score = base_score
        elif (
            live_gpu_data is not None
            and name in live_gpu_data.get("base", {}).get("retention", {})
            and name in live_gpu_data.get("tuned", {}).get("retention", {})
        ):
            b_item = live_gpu_data["base"]["retention"][name]
            t_item = live_gpu_data["tuned"]["retention"][name]
            base_nll = float(b_item["nll"]) if b_item.get("nll") is not None else None
            tuned_nll = float(t_item["nll"]) if t_item.get("nll") is not None else None
            n_samples_anchor = int(t_item.get("n_samples", b_item.get("n_samples", n_samples_anchor)) or 0)
            base_score = round(float(b_item["eval_score_pct"]), 2)
            evaluated_score = round(float(t_item["eval_score_pct"]), 2)
        else:
            if eval_base or no_reference_fallback:
                try:
                    base_score = run_lm_eval_task(
                        model_path=str(base_model_id),
                        task_name=str(spec["lm_eval_task"]),
                        num_fewshot=int(spec.get("num_fewshot", 0)),
                        batch_size=int(eval_cfg.get("batch_size", 16)),
                        device=target_device,
                        limit=limit,
                        metric_hint=str(spec.get("metric", "acc_norm")),
                        use_vllm=use_vllm,
                        apply_chat_template=apply_chat_template,
                    )
                except Exception as exc:
                    if no_reference_fallback:
                        raise RuntimeError(
                            f"Base retention anchor '{name}' ({spec['lm_eval_task']}) failed and "
                            f"--no-reference-fallback is active: {exc}"
                        ) from exc
                    logger.warning("Base anchor eval fallback for %s: %s", name, exc)
            try:
                evaluated_score = run_lm_eval_task(
                    model_path=model_path,
                    task_name=str(spec["lm_eval_task"]),
                    num_fewshot=int(spec.get("num_fewshot", 0)),
                    batch_size=int(eval_cfg.get("batch_size", 16)),
                    device=target_device,
                    limit=limit,
                    metric_hint=str(spec.get("metric", "acc_norm")),
                    use_vllm=use_vllm,
                    apply_chat_template=apply_chat_template,
                )
            except Exception as exc:
                if no_reference_fallback:
                    raise RuntimeError(
                        f"Tuned retention anchor '{name}' ({spec['lm_eval_task']}) failed and "
                        f"--no-reference-fallback is active: {exc}"
                    ) from exc
                logger.warning(
                    "Live lm_eval anchor '%s' (%s) fallback to baseline score: %s",
                    name,
                    spec["lm_eval_task"],
                    exc,
                )
                evaluated_score = base_score

        anchor_entry: Dict[str, Any] = {
            "domain": spec["domain"],
            "lm_eval_task": spec["lm_eval_task"],
            "metric": spec["metric"],
            "num_fewshot": int(spec.get("num_fewshot", 0)),
            "baseline_score": base_score,
            "evaluated_score": evaluated_score,
            "max_regression_pct": float(spec.get("max_regression_pct", 0.20)),
        }
        if n_samples_anchor > 0:
            anchor_entry["n_samples"] = n_samples_anchor
        if base_nll is not None and tuned_nll is not None:
            anchor_entry["base_nll"] = round(base_nll, 4)
            anchor_entry["tuned_nll"] = round(tuned_nll, 4)
        anchor_results[name] = anchor_entry

    gate_summary = evaluate_retention_gate(
        anchor_results=anchor_results,
        min_retention_ratio=float(gate_cfg.get("min_retention_ratio", 0.998)),
        max_allowed_drop_pct=float(gate_cfg.get("max_allowed_drop_pct", 0.20)),
        sweep_lambdas=gate_cfg.get("sweep_lambdas_on_regression", [0.80, 0.85, 0.90]),
    )

    avg_ar_base = sum(v["baseline_score"] for v in arabic_results.values()) / max(
        1, len(arabic_results)
    )
    avg_ar_eval = sum(v["evaluated_score"] for v in arabic_results.values()) / max(
        1, len(arabic_results)
    )

    return {
        "mode": "dry_run" if dry_run else ("live_lm_eval" if live_gpu_data else "live_eval"),
        "model_key": model_key,
        "model_path": model_path,
        "arabic_benchmarks_count": len(arabic_results),
        "retention_anchors_count": len(anchor_results),
        "average_arabic_baseline": round(avg_ar_base, 4),
        "average_arabic_evaluated": round(avg_ar_eval, 4),
        "average_arabic_gain": round(avg_ar_eval - avg_ar_base, 4),
        "arabic_benchmarks": arabic_results,
        "retention_anchors": anchor_results,
        "retention_gate": gate_summary,
    }


def generate_family_markdown_report(
    e4b_results: Dict[str, Any],
    moe_results: Dict[str, Any],
) -> str:
    """Render a side-by-side Markdown report comparing E4B (Dense) and 26B-A4B (MoE)."""
    e4b_gain_sign = "+" if e4b_results["average_arabic_gain"] >= 0 else ""
    moe_gain_sign = "+" if moe_results["average_arabic_gain"] >= 0 else ""
    lines: List[str] = [
        "# Gemma-4-Arabic Iso-Active Family Benchmark Comparison Report",
        "",
        f"- **Dense Edge Model (`E4B`):** `{e4b_results['model_path']}` (Avg Arabic Gain: `{e4b_gain_sign}{e4b_results['average_arabic_gain']:.2f}` pts)",
        f"- **Sparse MoE Flagship (`26B-A4B`):** `{moe_results['model_path']}` (Avg Arabic Gain: `{moe_gain_sign}{moe_results['average_arabic_gain']:.2f}` pts)",
        f"- **E4B Zero-Forgetting Gate:** `{'PASSED' if e4b_results['retention_gate']['gate_passed'] else 'FAILED'}` (`{e4b_results['retention_gate']['min_observed_retention_ratio']:.4f}`)",
        f"- **26B-A4B Zero-Forgetting Gate:** `{'PASSED' if moe_results['retention_gate']['gate_passed'] else 'FAILED'}` (`{moe_results['retention_gate']['min_observed_retention_ratio']:.4f}`)",
        "",
        "## 1. Arabic & Saudi Benchmarks (4-Column Family Progression)",
        "",
        "| Benchmark | Metric | Shots | E4B Base (%) | E4B Arabic-It (%) | E4B NLL (Base -> Tuned) | 26B-A4B Base (%) | 26B-A4B Arabic-It (%) | 26B-A4B NLL (Base -> Tuned) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, e4b_item in e4b_results["arabic_benchmarks"].items():
        moe_item = moe_results["arabic_benchmarks"].get(name, e4b_item)
        e4b_nll_str = (
            f"{float(e4b_item['base_nll']):.2f} -> {float(e4b_item['tuned_nll']):.2f}"
            if e4b_item.get("base_nll") is not None and e4b_item.get("tuned_nll") is not None
            else "N/A"
        )
        moe_nll_str = (
            f"{float(moe_item['base_nll']):.2f} -> {float(moe_item['tuned_nll']):.2f}"
            if moe_item.get("base_nll") is not None and moe_item.get("tuned_nll") is not None
            else "N/A"
        )
        lines.append(
            f"| **{name}** | `{e4b_item['metric']}` | {e4b_item['num_fewshot']} | "
            f"{float(e4b_item['baseline_score']):.2f} | **{float(e4b_item['evaluated_score']):.2f}** | `{e4b_nll_str}` | "
            f"{float(moe_item['baseline_score']):.2f} | **{float(moe_item['evaluated_score']):.2f}** | `{moe_nll_str}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Zero-Forgetting Retention Anchors (English / Math / Code)",
            "",
            "| Anchor | Domain | Shots | E4B Base (%) | E4B Arabic-It (%) | E4B Ratio | 26B-A4B Base (%) | 26B-A4B Arabic-It (%) | 26B-A4B Ratio |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name, e4b_item in e4b_results["retention_anchors"].items():
        moe_item = moe_results["retention_anchors"].get(name, e4b_item)
        e4b_ratio = float(e4b_item.get("retention_ratio", 1.0))
        moe_ratio = float(moe_item.get("retention_ratio", 1.0))
        lines.append(
            f"| **{name}** | `{e4b_item['domain']}` | {e4b_item.get('num_fewshot', 0)} | "
            f"{float(e4b_item['baseline_score']):.2f} | **{float(e4b_item['evaluated_score']):.2f}** | `{e4b_ratio:.4f}` | "
            f"{float(moe_item['baseline_score']):.2f} | **{float(moe_item['evaluated_score']):.2f}** | `{moe_ratio:.4f}` |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the benchmark evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run the 7 Arabic/Saudi benchmarks and 4 English/Math/Code retention anchors via lm_eval and lighteval."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "eval_config.yaml"),
        help="Path to evaluation YAML configuration file.",
    )
    parser.add_argument(
        "--model-key",
        "--model-family",
        dest="model_key",
        type=str,
        choices=["e4b", "26b_a4b", "both"],
        default="e4b",
        help="Model family member to evaluate: e4b, 26b_a4b, or both.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model checkpoint path or Hugging Face ID to evaluate.",
    )
    parser.add_argument(
        "--eval-base",
        action="store_true",
        help="Also run live evaluation on the untuned base model instead of using cached baseline scores.",
    )
    parser.add_argument(
        "--use-cached-eval",
        action="store_true",
        default=False,
        help="Load cached live_raw_*.json if present and matching the current eval_version.",
    )
    parser.add_argument(
        "--no-reference-fallback",
        dest="no_reference_fallback",
        action="store_true",
        default=True,
        help="Fail loudly instead of falling back to reference YAML numbers if live evaluation fails (default: True).",
    )
    parser.add_argument(
        "--allow-reference-fallback",
        dest="no_reference_fallback",
        action="store_false",
        help="Allow falling back to reference YAML baseline scores if live evaluation fails.",
    )
    parser.add_argument(
        "--use-vllm",
        dest="use_vllm",
        action="store_true",
        default=True,
        help="Use vLLM backend for lm_eval and lighteval (default: True).",
    )
    parser.add_argument(
        "--no-vllm",
        dest="use_vllm",
        action="store_false",
        help="Use Hugging Face HFLM / Accelerate instead of vLLM.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        default=False,
        help="Apply the model's Hugging Face chat template consistently across both base and tuned runs.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to write the Markdown benchmark comparison report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write the JSON benchmark results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit per benchmark suite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all 7 Arabic benchmarks, 4 retention anchors, and Markdown report generation.",
    )
    return parser.parse_args(argv)


def _build_child_eval_cmd(args: argparse.Namespace, model_key: str) -> List[str]:
    """Construct CLI command for an isolated single-GPU child subprocess."""
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--model-key",
        model_key,
    ]
    if args.eval_base:
        cmd.append("--eval-base")
    if args.use_cached_eval:
        cmd.append("--use-cached-eval")
    if args.no_reference_fallback:
        cmd.append("--no-reference-fallback")
    else:
        cmd.append("--allow-reference-fallback")
    if args.use_vllm:
        cmd.append("--use-vllm")
    else:
        cmd.append("--no-vllm")
    if args.apply_chat_template:
        cmd.append("--apply-chat-template")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for automated benchmark evaluation and report generation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    args = parse_args(argv)

    cfg = load_eval_config(pathlib.Path(args.config))
    eval_cfg = cfg.get("evaluation", {})
    models_cfg = cfg.get("models", {})

    if args.model_key == "both":
        e4b_path = str(
            models_cfg.get("e4b", {}).get(
                "merged_checkpoint_dir", "outputs/merged_gemma4_e4b_arabic/merged_16bit"
            )
        )
        moe_path = str(
            models_cfg.get("26b_a4b", {}).get(
                "merged_checkpoint_dir", "outputs/merged_gemma4_26b_a4b_arabic/merged_16bit"
            )
        )

        can_parallel_dual_gpu = False
        if not args.dry_run:
            try:
                import torch

                can_parallel_dual_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2
            except Exception:
                can_parallel_dual_gpu = False

        if can_parallel_dual_gpu:
            logger.info(
                "Detected >= 2 GPUs: launching isolated subprocesses for E4B (CUDA_VISIBLE_DEVICES=0) and 26B-A4B (CUDA_VISIBLE_DEVICES=1)..."
            )
            env_gpu0 = os.environ.copy()
            env_gpu0["CUDA_VISIBLE_DEVICES"] = "0"
            env_gpu1 = os.environ.copy()
            env_gpu1["CUDA_VISIBLE_DEVICES"] = "1"

            proc_e4b = subprocess.Popen(
                _build_child_eval_cmd(args, "e4b"),
                env=env_gpu0,
                cwd=str(PROJECT_ROOT),
            )
            proc_moe = subprocess.Popen(
                _build_child_eval_cmd(args, "26b_a4b"),
                env=env_gpu1,
                cwd=str(PROJECT_ROOT),
            )
            rc_e4b = proc_e4b.wait()
            rc_moe = proc_moe.wait()
            if rc_e4b != 0 or rc_moe != 0:
                raise RuntimeError(
                    f"Parallel Dual-GPU evaluation subprocess failed (rc_e4b={rc_e4b}, rc_26b_a4b={rc_moe})."
                )

            # Load completed results from outputs/eval_reports/live_raw_e4b.json and live_raw_26b_a4b.json
            e4b_res = build_benchmark_results(
                cfg=cfg,
                model_path=e4b_path,
                dry_run=False,
                limit=args.limit,
                model_key="e4b",
                eval_base=args.eval_base,
                use_cached_eval=True,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
            )
            moe_res = build_benchmark_results(
                cfg=cfg,
                model_path=moe_path,
                dry_run=False,
                limit=args.limit,
                model_key="26b_a4b",
                eval_base=args.eval_base,
                use_cached_eval=True,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
            )
        else:
            e4b_res = build_benchmark_results(
                cfg=cfg,
                model_path=e4b_path,
                dry_run=args.dry_run,
                limit=args.limit,
                model_key="e4b",
                eval_base=args.eval_base,
                use_cached_eval=args.use_cached_eval,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
            )
            moe_res = build_benchmark_results(
                cfg=cfg,
                model_path=moe_path,
                dry_run=args.dry_run,
                limit=args.limit,
                model_key="26b_a4b",
                eval_base=args.eval_base,
                use_cached_eval=args.use_cached_eval,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
            )

        family_md = generate_family_markdown_report(e4b_res, moe_res)
        family_payload = {
            "mode": e4b_res.get("mode", "live_lm_eval"),
            "models": {
                "e4b": e4b_res,
                "26b_a4b": moe_res,
            },
            **e4b_res,
        }
        if args.output_md or not args.dry_run:
            md_path = pathlib.Path(
                args.output_md
                or eval_cfg.get(
                    "output_report_path",
                    "outputs/eval_reports/benchmark_comparison.md",
                )
            )
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(family_md, encoding="utf-8")
            fam_md_path = PROJECT_ROOT / "outputs" / "eval_reports" / "family_benchmark_comparison.md"
            fam_md_path.parent.mkdir(parents=True, exist_ok=True)
            fam_md_path.write_text(family_md, encoding="utf-8")
            legacy_md_path = PROJECT_ROOT / "outputs" / "eval_reports" / "benchmark_report.md"
            legacy_md_path.write_text(family_md, encoding="utf-8")
        if args.output_json or not args.dry_run:
            json_path = pathlib.Path(
                args.output_json
                or eval_cfg.get(
                    "output_json_path",
                    "outputs/eval_reports/benchmark_results.json",
                )
            )
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(family_payload, indent=2), encoding="utf-8")
            fam_json_path = pathlib.Path(
                eval_cfg.get(
                    "family_comparison_json_path",
                    "outputs/eval_reports/family_benchmark_comparison.json",
                )
            )
            fam_json_path.parent.mkdir(parents=True, exist_ok=True)
            fam_json_path.write_text(json.dumps(family_payload, indent=2), encoding="utf-8")
        print(family_md)
        return 0

    default_path = (
        models_cfg.get(args.model_key, {}).get("merged_checkpoint_dir")
        if isinstance(models_cfg, dict) and args.model_key in models_cfg
        else None
    )
    model_path = str(
        args.model_path
        or default_path
        or eval_cfg.get("merged_checkpoint_dir", "outputs/merged_gemma4_e4b_arabic")
    )

    results = build_benchmark_results(
        cfg=cfg,
        model_path=model_path,
        dry_run=args.dry_run,
        limit=args.limit,
        model_key=args.model_key,
        eval_base=args.eval_base,
        use_cached_eval=args.use_cached_eval,
        no_reference_fallback=args.no_reference_fallback,
        use_vllm=args.use_vllm,
        apply_chat_template=args.apply_chat_template,
    )
    md_report = generate_markdown_report(results)

    if args.output_md or not args.dry_run:
        md_path = pathlib.Path(
            args.output_md
            or eval_cfg.get("output_report_path", "outputs/eval_reports/benchmark_comparison.md")
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_report, encoding="utf-8")

    if args.output_json or not args.dry_run:
        json_path = pathlib.Path(
            args.output_json
            or eval_cfg.get("output_json_path", "outputs/eval_reports/benchmark_results.json")
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(md_report)
    print(
        json.dumps(
            {
                "mode": results["mode"],
                "model_key": results["model_key"],
                "arabic_benchmarks_count": results["arabic_benchmarks_count"],
                "retention_anchors_count": results["retention_anchors_count"],
                "average_arabic_gain": results["average_arabic_gain"],
                "retention_gate_passed": results["retention_gate"]["gate_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
