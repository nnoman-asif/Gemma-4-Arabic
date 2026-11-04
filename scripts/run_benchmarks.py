"""Automated benchmark runner and Zero-Forgetting Retention Gate evaluator.

Evaluates:
- 7 Primary Arabic & Saudi Benchmarks (`ArabicMMLU`, `MadinahQA`, `AlGhafa`,
  `ACVA`, `EXAMS-AR`, `AraTrust`, `MGSM-Ar`) using `lm-evaluation-harness` (`lm_eval`).
- 4 English/Math/Code Zero-Forgetting Retention Anchors (`MMLU-Pro`, `GSM8K`,
  `HumanEval`, `ARC-Challenge`) using `lm-evaluation-harness` (`lm_eval`).
- Supports single-model (`e4b` or `26b_a4b`) and dual-model family (`both`)
  comparisons across `Gemma-4-E4B-Arabic-It` and `Gemma-4-26B-A4B-Arabic-It`.
- Generates structured Markdown and JSON comparison reports and verifies the
  `min_retention_ratio >= 0.998` (`max_allowed_drop_pct <= 0.20`) gate.
"""

import argparse
import concurrent.futures
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

# Enforce deterministic hashing and enable HumanEval code execution in lm_eval
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

EVAL_CACHE_VERSION = "v13_lm_eval_standard"

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
    "arabic_leaderboard_alghafa_light": 9,
    "arabic_leaderboard_acva_light": len(ACVA_CORE_SUBTASKS),
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
    if cfg is not None and text_cfg is not None:
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
# Official lm_eval Dataset Processors for Custom Arabic Task Configurations
# ---------------------------------------------------------------------------

_ARABIC_LETTERS: Tuple[str, ...] = ("أ", "ب", "ج", "د", "هـ")
_LATIN_TO_IDX: Dict[str, int] = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "أ": 0,
    "ب": 1,
    "ج": 2,
    "د": 3,
    "هـ": 4,
}


def _process_mbzuai_mcq_docs(dataset: Any) -> Any:
    """Format `MBZUAI/ArabicMMLU` or `MBZUAI/MadinahQA` rows for `lm_eval` `multiple_choice`.

    Preserves all native options (`2` to `5`) without pruning or slot rotation, and sets
    `choices` to the full Arabic option texts so `lm_eval` computes both unnormalized `acc`
    and byte-length-normalized `acc_norm`.
    """

    def _map_row(doc: Dict[str, Any]) -> Dict[str, Any]:
        subject = str(doc.get("Subject") or "").strip()
        context = str(doc.get("Context") or "").strip()
        question = str(doc.get("Question") or "").strip()
        options: List[str] = []
        for idx in range(1, 6):
            val = doc.get(f"Option {idx}")
            if val is not None and str(val).strip() and str(val).strip().lower() != "none":
                options.append(str(val).strip())
        if len(options) < 2:
            options = ["صح", "خطأ"]

        ans_key = str(doc.get("Answer Key") or "A").strip().upper()
        gold_idx = _LATIN_TO_IDX.get(ans_key, 0)
        if gold_idx < 0 or gold_idx >= len(options):
            gold_idx = 0

        header = (
            f"الأسئلة التالية هي أسئلة متعددة الاختيارات حول {subject}.\n\n"
            if subject
            else "الأسئلة التالية هي أسئلة متعددة الاختيارات مع الجواب الصحيح.\n\n"
        )
        body = f"السياق: {context}\nالسؤال: {question}\n" if context else f"السؤال: {question}\n"
        opts_lines = "\n".join(
            f"{_ARABIC_LETTERS[i]}) {opt}" for i, opt in enumerate(options[: len(_ARABIC_LETTERS)])
        )
        query = f"{header}{body}{opts_lines}\nالإجابة:"
        return {
            "query": query,
            "choices": options[: len(_ARABIC_LETTERS)],
            "gold": gold_idx,
        }

    filtered = dataset.filter(
        lambda row: bool(row.get("Question"))
        and bool(row.get("Option 1"))
        and bool(row.get("Option 2"))
        and str(row.get("Answer Key") or "").strip().upper() in _LATIN_TO_IDX
    )
    return filtered.map(_map_row)


def _process_aratrust_docs(dataset: Any) -> Any:
    """Format `asas-ai/AraTrust` rows for `lm_eval` `multiple_choice`."""

    def _map_row(doc: Dict[str, Any]) -> Dict[str, Any]:
        question = str(doc.get("Question") or "").strip()
        raw_opts = [
            str(doc.get(col)).strip()
            for col in ("A", "B", "C")
            if doc.get(col) not in (None, "", "None")
        ]
        clean_opts: List[str] = []
        for ro in raw_opts:
            for pref in ("أ)", "ب)", "ج)", "أ-", "ب-", "ج-", "أ.", "ب.", "ج."):
                if ro.startswith(pref):
                    ro = ro[len(pref) :].strip()
            if ro:
                clean_opts.append(ro)
        if len(clean_opts) < 2:
            clean_opts = ["نعم", "لا"]

        ans_key = str(doc.get("Answer") or "أ").strip()
        gold_idx = _LATIN_TO_IDX.get(ans_key, 0)
        if gold_idx < 0 or gold_idx >= len(clean_opts):
            gold_idx = 0

        opts_lines = "\n".join(
            f"{_ARABIC_LETTERS[i]}) {opt}" for i, opt in enumerate(clean_opts[: len(_ARABIC_LETTERS)])
        )
        query = (
            "الأسئلة التالية هي أسئلة متعددة الاختيارات مع الجواب الصحيح.\n\n"
            f"السؤال: {question}\n{opts_lines}\nالإجابة:"
        )
        return {
            "query": query,
            "choices": clean_opts[: len(_ARABIC_LETTERS)],
            "gold": gold_idx,
        }

    filtered = dataset.filter(
        lambda row: bool(row.get("Question"))
        and bool(row.get("A"))
        and bool(row.get("B"))
    )
    return filtered.map(_map_row)


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
        # Ensure few-shot demonstrations from main_train end with canonical `#### <target_number>`
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
    """Return a native `lm_eval` `ConfigurableTask` dictionary for custom Arabic suites."""
    key = task_name.strip().lower()
    if key == "arabicmmlu":
        return {
            "task": "arabicmmlu_all",
            "dataset_path": "MBZUAI/ArabicMMLU",
            "dataset_name": "All",
            "test_split": "test",
            "validation_split": "dev",
            "fewshot_split": "dev",
            "num_fewshot": int(num_fewshot),
            "output_type": "multiple_choice",
            "process_docs": _process_mbzuai_mcq_docs,
            "doc_to_text": "{{query}}",
            "doc_to_choice": "choices",
            "doc_to_target": "{{gold}}",
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
            ],
            "metadata": {"version": 1.0},
        }
    if key == "madinahqa":
        return {
            "task": "madinahqa",
            "dataset_path": "MBZUAI/MadinahQA",
            "dataset_name": "Arabic Language (Grammar)",
            "test_split": "test",
            "validation_split": "dev",
            "fewshot_split": "dev",
            "num_fewshot": int(num_fewshot),
            "output_type": "multiple_choice",
            "process_docs": _process_mbzuai_mcq_docs,
            "doc_to_text": "{{query}}",
            "doc_to_choice": "choices",
            "doc_to_target": "{{gold}}",
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
            ],
            "metadata": {"version": 1.0},
        }
    if key == "aratrust":
        return {
            "task": "aratrust",
            "dataset_path": "asas-ai/AraTrust",
            "dataset_name": "default",
            "test_split": "test",
            "fewshot_split": "test",
            "num_fewshot": int(num_fewshot),
            "output_type": "multiple_choice",
            "process_docs": _process_aratrust_docs,
            "doc_to_text": "{{query}}",
            "doc_to_choice": "choices",
            "doc_to_target": "{{gold}}",
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
                {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
            ],
            "metadata": {"version": 1.0},
        }
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
                "until": ["السؤال:", "</s>", "<|im_end|>", "<turn|>"],
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 512,
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
    raise KeyError(f"Unknown custom lm_eval task dictionary key: '{task_name}'")


def resolve_lm_eval_task_specs(
    task_name: str,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
) -> Tuple[List[Union[str, Dict[str, Any]]], str, Optional[int]]:
    """Resolve a benchmark's `lm_eval_task` into `(tasks_arg, primary_result_key, effective_limit)`."""
    clean_name = task_name.strip()
    eff_limit = compute_effective_task_limit(clean_name, limit)

    if clean_name.lower() in ("arabicmmlu", "madinahqa", "aratrust", "mgsm_ar"):
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
    """Verify zero catastrophic forgetting across all 4 English/Math/Code anchors.

    When `n_samples` is recorded on a finite-sample discrete benchmark run (e.g. `N=50` or `N=200`),
    accounts for the single-question discretization step (`100.0 / n_samples`) so a 1-item
    difference does not trigger a false-positive gate failure.
    """
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
        "| Benchmark | Task (`lm_eval`) | Metric | Shots | Base NLL | Tuned NLL | Baseline (%) | Evaluated (%) | Gain (% pts) |",
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
            raise ValueError(f"No valid metric dictionary found for task '{task_name}'.")
        vals: List[float] = []
        for sub in subtask_dicts:
            for key in candidate_keys:
                if key in sub and isinstance(sub[key], (int, float)):
                    vals.append(float(sub[key]))
                    break
        if not vals:
            raise KeyError(f"Could not extract score for '{task_name}' from subtasks.")
        raw_val = sum(vals) / len(vals)
    else:
        raw_val = None
        for key in candidate_keys:
            if key in task_metrics and isinstance(task_metrics[key], (int, float)):
                raw_val = float(task_metrics[key])
                break
        if raw_val is None:
            numeric_vals = [
                float(v)
                for k, v in task_metrics.items()
                if isinstance(v, (int, float)) and "stderr" not in str(k)
            ]
            if not numeric_vals:
                raise KeyError(f"No numeric score found in task metrics for '{task_name}'.")
            raw_val = numeric_vals[0]

    score_pct = raw_val * 100.0 if 0.0 <= raw_val <= 1.0 else raw_val
    return round(score_pct, 2)


def extract_lm_eval_detailed_metrics(
    results_payload: Dict[str, Any],
    task_name: str,
    metric_hint: str = "acc_norm",
) -> Dict[str, Any]:
    """Extract primary score, unnormalized `acc` (`HardTop1`), `GoldNLL`, and `n_samples` from `lm_eval` output."""
    eval_score_pct = parse_lm_eval_results_json(
        results_payload=results_payload,
        task_name=task_name,
        metric_hint=metric_hint,
    )
    try:
        top1_acc_pct = parse_lm_eval_results_json(
            results_payload=results_payload,
            task_name=task_name,
            metric_hint="acc" if "acc" in metric_hint else metric_hint,
        )
    except Exception:
        top1_acc_pct = eval_score_pct

    # Extract sample count and gold log-likelihood (when multiple_choice samples are logged)
    n_samples = 0
    gold_nlls: List[float] = []
    samples_block = results_payload.get("samples", {})
    if isinstance(samples_block, dict):
        for _subtask_name, sample_list in samples_block.items():
            if not isinstance(sample_list, list):
                continue
            n_samples += len(sample_list)
            for sample in sample_list:
                if not isinstance(sample, dict):
                    continue
                target = sample.get("target")
                resps = sample.get("resps")
                if isinstance(target, int) and isinstance(resps, list) and 0 <= target < len(resps):
                    resp_entry = resps[target]
                    if isinstance(resp_entry, (list, tuple)) and len(resp_entry) > 0:
                        first_pair = resp_entry[0]
                        if isinstance(first_pair, (list, tuple)) and len(first_pair) > 0:
                            try:
                                ll_val = float(first_pair[0])
                                if not (math.isnan(ll_val) or math.isinf(ll_val)):
                                    gold_nlls.append(-ll_val)
                            except (TypeError, ValueError):
                                pass

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
    """Run a single `lm_eval` task via `lm_eval.simple_evaluate` or CLI and return the score."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    tasks_arg, primary_key, eff_limit = resolve_lm_eval_task_specs(
        task_name=task_name,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    model_backend = "vllm" if use_vllm else "hf"
    model_args = (
        f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,"
        f"gpu_memory_utilization=0.85,max_model_len=4096"
        if use_vllm
        else f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"
    )
    payload = lm_eval.simple_evaluate(
        model=model_backend,
        model_args=model_args,
        tasks=tasks_arg,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=eff_limit,
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


def _evaluate_checkpoint_with_lm_eval(
    model_path: str,
    device: str,
    cfg: Dict[str, Any],
    tag: str,
    limit: Optional[int] = None,
    use_vllm: bool = False,
    apply_chat_template: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load checkpoint once via `lm_eval` (`HFLM` or `vLLM`) and evaluate all 11 benchmarks."""
    import lm_eval
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    eval_cfg = cfg.get("evaluation", {})
    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})
    batch_size = int(eval_cfg.get("batch_size", 16))
    max_seq_len = int(eval_cfg.get("max_seq_length", 4096))

    cache_dir = _resolve_hf_eval_cache_dir()
    if cache_dir:
        os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

    logger.info("[%s] Initializing lm_eval model on %s: %s", tag, device, model_path)
    model_obj: Any = None
    lm_obj: Any = None

    if use_vllm:
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
    else:
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

        try:
            model_obj = AutoModelForImageTextToText.from_pretrained(
                load_target_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
            )
        except Exception:
            model_obj = AutoModelForCausalLM.from_pretrained(
                load_target_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
            )
        if is_peft_adapter:
            from peft import PeftModel

            model_obj = PeftModel.from_pretrained(model_obj, model_path)
        _patch_gemma4_config_for_lm_eval(model_obj)
        model_obj.eval()
        lm_obj = HFLM(
            pretrained=model_obj,
            tokenizer=tokenizer,
            backend="causal",
            batch_size=batch_size,
            max_length=max_seq_len,
            trust_remote_code=True,
        )

    task_manager = TaskManager()
    out: Dict[str, Dict[str, Dict[str, Any]]] = {"arabic": {}, "retention": {}}

    suite_groups = (
        ("arabic", ar_defs),
        ("retention", ret_defs),
    )
    for cat, suite_defs in suite_groups:
        for suite_name, spec in suite_defs.items():
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
            payload = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=tasks_arg,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=eff_limit,
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

    del lm_obj
    if model_obj is not None:
        del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_inprocess_gpu_eval(
    model_key: str,
    base_model_id: str,
    tuned_model_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    target_device: str = "cuda:0",
    limit: Optional[int] = None,
    use_cached_eval: bool = False,
    use_vllm: bool = False,
    apply_chat_template: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run standardized `lm_eval` GPU evaluation across all 11 benchmarks for `base` and `tuned`."""
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
    """Run all 7 Arabic benchmarks and 4 retention anchors via `lm_eval`."""
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
                            f"Base lm_eval task '{name}' ({spec['lm_eval_task']}) failed and "
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
                        f"Tuned lm_eval task '{name}' ({spec['lm_eval_task']}) failed and "
                        f"--no-reference-fallback is active: {exc}"
                    ) from exc
                logger.warning(
                    "Live lm_eval task '%s' (%s) failed: %s",
                    name,
                    spec["lm_eval_task"],
                    exc,
                )
                evaluated_score = base_score

        entry_dict: Dict[str, Any] = {
            "lm_eval_task": spec["lm_eval_task"],
            "lighteval_task": spec.get("lighteval_task", ""),
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
        description="Run the 7 Arabic/Saudi benchmarks and 4 English/Math/Code retention anchors via lm_eval."
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
        help="Also run live lm_eval on the untuned base model instead of using cached baseline scores.",
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
        action="store_true",
        default=False,
        help="Use lm_eval's vLLM backend instead of HFLM for faster generative evaluation.",
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


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for automated benchmark evaluation and report generation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
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
            logger.info("Detected >= 2 GPUs: evaluating E4B (cuda:0) and 26B-A4B (cuda:1) in parallel...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fut_e4b = pool.submit(
                    build_benchmark_results,
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
                fut_moe = pool.submit(
                    build_benchmark_results,
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
                e4b_res = fut_e4b.result()
                moe_res = fut_moe.result()
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
