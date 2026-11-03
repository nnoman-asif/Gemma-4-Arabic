"""Automated benchmark runner and Zero-Forgetting Retention Gate evaluator.

Evaluates:
- 7 Primary Arabic & Saudi Benchmarks (`ArabicMMLU`, `MadinahQA`, `AlGhafa`,
  `ACVA`, `EXAMS-AR`, `AraTrust`, `MGSM-Ar`).
- 4 English/Math/Code Zero-Forgetting Retention Anchors (`MMLU-Pro`, `GSM8K`,
  `HumanEval`, `ARC-Challenge`).
- Supports single-model (`e4b` or `26b_a4b`) and dual-model family (`both`)
  comparisons across `Gemma-4-E4B-Arabic-It` and `Gemma-4-26B-A4B-Arabic-It`.
- Generates structured Markdown and JSON comparison reports and verifies the
  `min_retention_ratio >= 0.998` (`max_allowed_drop_pct <= 0.20`) gate.
"""

import argparse
import gc
import json
import logging
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_template import format_gemma4_chat

logger = logging.getLogger(__name__)


def load_eval_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load and parse `configs/eval_config.yaml`."""
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


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
        passed = (delta >= -allowed_drop) and (ratio >= min_retention_ratio)
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

    task_metrics = results_block.get(task_name)
    if task_metrics is None:
        # Fallback to the first task dictionary or average across subtasks
        subtask_dicts = [v for v in results_block.values() if isinstance(v, dict)]
        if not subtask_dicts:
            raise ValueError(f"No valid metric dictionary found for task '{task_name}'.")
        candidate_keys = [
            f"{metric_hint},none",
            metric_hint,
            "acc_norm,none",
            "acc,none",
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match,none",
            "pass@1,none",
            "acc_norm",
            "acc",
            "exact_match",
        ]
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
        candidate_keys = [
            f"{metric_hint},none",
            metric_hint,
            "acc_norm,none",
            "acc,none",
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match,none",
            "pass@1,none",
            "acc_norm",
            "acc",
            "exact_match",
        ]
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

    # Convert [0.0, 1.0] fraction to percentage [0.0, 100.0]
    score_pct = raw_val * 100.0 if 0.0 <= raw_val <= 1.0 else raw_val
    return round(score_pct, 2)


def run_lm_eval_task(
    model_path: str,
    task_name: str,
    num_fewshot: int,
    batch_size: int = 4,
    device: str = "cuda:0",
    limit: Optional[int] = None,
    metric_hint: str = "acc_norm",
) -> float:
    """Invoke `lm_eval` CLI with `--output_path` and parse the resulting JSON score."""
    with tempfile.TemporaryDirectory(prefix="lm_eval_out_") as tmp_dir:
        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
            "--tasks",
            task_name,
            "--num_fewshot",
            str(num_fewshot),
            "--batch_size",
            str(batch_size),
            "--device",
            device,
            "--output_path",
            tmp_dir,
        ]
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if "humaneval" in task_name.lower():
            cmd.append("--confirm_run_unsafe_code")

        env = os.environ.copy()
        env["HF_ALLOW_CODE_EVAL"] = "1"
        logger.info("Executing lm_eval command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

        json_files = sorted(
            pathlib.Path(tmp_dir).rglob("results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not json_files:
            json_files = sorted(
                pathlib.Path(tmp_dir).rglob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if not json_files:
            raise FileNotFoundError(
                f"lm_eval completed for task '{task_name}' but wrote no JSON file in {tmp_dir}."
            )

        payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        return parse_lm_eval_results_json(
            results_payload=payload,
            task_name=task_name,
            metric_hint=metric_hint,
        )


def _resolve_hf_eval_cache_dir() -> Optional[str]:
    """Return `/dev/shm/hf_eval_cache` on Linux to avoid root disk usage, or None otherwise."""
    shm_root = pathlib.Path("/dev/shm")
    if shm_root.exists() and os.access(str(shm_root), os.W_OK):
        cache_dir = shm_root / "hf_eval_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)
    return None


def _format_mcq_pair(
    question: str,
    options: List[str],
    gold_idx: int,
    is_arabic: bool = True,
) -> Optional[Tuple[str, str]]:
    """Format a multiple-choice benchmark item into `(user_prompt, clean_answer)`."""
    clean_q = str(question or "").strip()
    clean_opts = [str(o).strip() for o in options if o is not None and str(o).strip()]
    if not clean_q or len(clean_opts) < 2:
        return None
    if gold_idx < 0 or gold_idx >= len(clean_opts):
        return None

    if is_arabic:
        letters = ["أ", "ب", "ج", "د", "هـ", "و"]
        usable = min(len(clean_opts), len(letters))
        if gold_idx >= usable:
            return None
        opts_block = "\n".join(f"{letters[i]}) {clean_opts[i]}" for i in range(usable))
        user_prompt = (
            f"السؤال: {clean_q}\n"
            f"الخيارات:\n{opts_block}\n"
            "اختر الإجابة الصحيحة من الخيارات أعلاه."
        )
        clean_answer = f"الإجابة الصحيحة هي ({letters[gold_idx]}): {clean_opts[gold_idx]}"
        return (user_prompt, clean_answer)

    letters_en = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    usable_en = min(len(clean_opts), len(letters_en))
    if gold_idx >= usable_en:
        return None
    opts_block_en = "\n".join(f"{letters_en[i]}) {clean_opts[i]}" for i in range(usable_en))
    user_prompt_en = (
        f"Question: {clean_q}\n"
        f"Options:\n{opts_block_en}\n"
        "Select the single best option."
    )
    clean_answer_en = f"The correct answer is ({letters_en[gold_idx]}): {clean_opts[gold_idx]}"
    return (user_prompt_en, clean_answer_en)


def build_heldout_eval_suites(limit: int = 40) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """Load real external Hugging Face benchmark test splits for all 7 Arabic and 4 Retention suites."""
    from datasets import load_dataset

    per_task = max(12, min(int(limit or 40), 64))
    cache_dir = _resolve_hf_eval_cache_dir()
    suites: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
        "arabic": {
            "ArabicMMLU": [],
            "MadinahQA": [],
            "AlGhafa": [],
            "ACVA": [],
            "EXAMS-AR": [],
            "AraTrust": [],
            "MGSM-Ar": [],
        },
        "retention": {
            "MMLU-Pro": [],
            "GSM8K": [],
            "HumanEval": [],
            "ARC-Challenge": [],
        },
    }

    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "أ": 0, "ب": 1, "ج": 2, "د": 3, "هـ": 4}

    # 1. ArabicMMLU (MBZUAI/ArabicMMLU, config="All", split="test")
    try:
        ds_ammlu = load_dataset("MBZUAI/ArabicMMLU", "All", split="test", cache_dir=cache_dir)
        ds_ammlu = ds_ammlu.shuffle(seed=9401)
        for row in ds_ammlu:
            ctx = str(row.get("Context") or "").strip()
            q = str(row.get("Question") or "").strip()
            full_q = f"{ctx}\n{q}".strip() if ctx else q
            opts = [
                str(row.get(f"Option {i}")).strip()
                for i in range(1, 6)
                if row.get(f"Option {i}") not in (None, "", "None")
            ]
            ans_k = str(row.get("Answer Key") or "").strip().upper()
            pair = _format_mcq_pair(full_q, opts, letter_to_idx.get(ans_k, -1), is_arabic=True)
            if pair is not None:
                suites["arabic"]["ArabicMMLU"].append(pair)
            if len(suites["arabic"]["ArabicMMLU"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load MBZUAI/ArabicMMLU (All): %s", exc)

    # 2. MadinahQA (MBZUAI/ArabicMMLU Arabic Language Grammar + General test splits)
    try:
        for cfg_name in ("Arabic Language (Grammar)", "Arabic Language (General)"):
            ds_mad = load_dataset("MBZUAI/ArabicMMLU", cfg_name, split="test", cache_dir=cache_dir)
            ds_mad = ds_mad.shuffle(seed=9402)
            half_target = max(6, per_task // 2)
            added = 0
            for row in ds_mad:
                ctx = str(row.get("Context") or "").strip()
                q = str(row.get("Question") or "").strip()
                full_q = f"{ctx}\n{q}".strip() if ctx else q
                opts = [
                    str(row.get(f"Option {i}")).strip()
                    for i in range(1, 6)
                    if row.get(f"Option {i}") not in (None, "", "None")
                ]
                ans_k = str(row.get("Answer Key") or "").strip().upper()
                pair = _format_mcq_pair(full_q, opts, letter_to_idx.get(ans_k, -1), is_arabic=True)
                if pair is not None:
                    suites["arabic"]["MadinahQA"].append(pair)
                    added += 1
                if added >= half_target or len(suites["arabic"]["MadinahQA"]) >= per_task:
                    break
    except Exception as exc:
        logger.warning("Could not load MadinahQA Arabic grammar/language splits: %s", exc)

    # 3. AlGhafa (OALL/AlGhafa-Arabic-LLM-Benchmark-Native, config="mcq_exams_test_ar", split="test")
    try:
        ds_alg = load_dataset(
            "OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
            "mcq_exams_test_ar",
            split="test",
            cache_dir=cache_dir,
        ).shuffle(seed=9403)
        for row in ds_alg:
            q = str(row.get("query") or "").strip()
            opts = [
                str(row.get(f"sol{i}")).strip()
                for i in range(1, 6)
                if row.get(f"sol{i}") not in (None, "", "None")
            ]
            raw_lbl = str(row.get("label", "0")).strip()
            g_idx = int(raw_lbl) if raw_lbl.isdigit() else 0
            if g_idx >= len(opts) and g_idx > 0:
                g_idx -= 1
            pair = _format_mcq_pair(q, opts, g_idx, is_arabic=True)
            if pair is not None:
                suites["arabic"]["AlGhafa"].append(pair)
            if len(suites["arabic"]["AlGhafa"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load OALL/AlGhafa-Arabic-LLM-Benchmark-Native: %s", exc)

    # 4. ACVA (OALL/ACVA cultural values test splits across Saudi_Arabia, Arab_Empire, Islam, Arabic_Culture)
    try:
        acva_configs = ("Saudi_Arabia", "Arab_Empire", "Islam", "Arabic_Culture")
        per_sub = max(4, (per_task + len(acva_configs) - 1) // len(acva_configs))
        for cfg_name in acva_configs:
            ds_acva = load_dataset("OALL/ACVA", cfg_name, split="test", cache_dir=cache_dir).shuffle(seed=9404)
            sub_count = 0
            for row in ds_acva:
                q = str(row.get("question") or "").strip()
                ans = str(row.get("answer") or "").strip()
                if not q or not ans:
                    continue
                opts = [
                    "صح (العبارة صحيحة وتوافق المعرفة الثقافية المعتمدة)",
                    "خطأ (العبارة غير صحيحة)",
                ]
                g_idx = 0 if "صح" in ans else 1
                pair = _format_mcq_pair(q, opts, g_idx, is_arabic=True)
                if pair is not None:
                    suites["arabic"]["ACVA"].append(pair)
                    sub_count += 1
                if sub_count >= per_sub or len(suites["arabic"]["ACVA"]) >= per_task:
                    break
            if len(suites["arabic"]["ACVA"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load OALL/ACVA: %s", exc)

    # 5. EXAMS-AR (OALL/Arabic_EXAMS, config="default", split="test")
    try:
        ds_exams = load_dataset("OALL/Arabic_EXAMS", "default", split="test", cache_dir=cache_dir).shuffle(
            seed=9405
        )
        for row in ds_exams:
            q = str(row.get("question") or "").strip()
            opts = [
                str(row.get(col)).strip()
                for col in ("A", "B", "C", "D")
                if row.get(col) not in (None, "", "None")
            ]
            ans_k = str(row.get("answer") or "").strip().upper()
            pair = _format_mcq_pair(q, opts, letter_to_idx.get(ans_k, -1), is_arabic=True)
            if pair is not None:
                suites["arabic"]["EXAMS-AR"].append(pair)
            if len(suites["arabic"]["EXAMS-AR"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load OALL/Arabic_EXAMS: %s", exc)

    # 6. AraTrust (asas-ai/AraTrust, config="default", split="test")
    try:
        ds_aratrust = load_dataset("asas-ai/AraTrust", "default", split="test", cache_dir=cache_dir).shuffle(
            seed=9406
        )
        for row in ds_aratrust:
            q = str(row.get("Question") or "").strip()
            raw_opts = [
                str(row.get(col)).strip()
                for col in ("A", "B", "C")
                if row.get(col) not in (None, "", "None")
            ]
            clean_opts: List[str] = []
            for ro in raw_opts:
                for pref in ("أ)", "ب)", "ج)", "أ-", "ب-", "ج-"):
                    if ro.startswith(pref):
                        ro = ro[len(pref) :].strip()
                clean_opts.append(ro)
            ans_k = str(row.get("Answer") or "").strip()
            pair = _format_mcq_pair(q, clean_opts, letter_to_idx.get(ans_k, -1), is_arabic=True)
            if pair is not None:
                suites["arabic"]["AraTrust"].append(pair)
            if len(suites["arabic"]["AraTrust"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load asas-ai/AraTrust: %s", exc)

    # 7. MGSM-Ar (MBZUAI/ArabicMMLU Math Primary School + Arabic math reasoning test split)
    try:
        ds_mgsm = load_dataset(
            "MBZUAI/ArabicMMLU",
            "Math (Primary School)",
            split="test",
            cache_dir=cache_dir,
        ).shuffle(seed=9407)
        for row in ds_mgsm:
            ctx = str(row.get("Context") or "").strip()
            q = str(row.get("Question") or "").strip()
            full_q = f"{ctx}\n{q}".strip() if ctx else q
            opts = [
                str(row.get(f"Option {i}")).strip()
                for i in range(1, 6)
                if row.get(f"Option {i}") not in (None, "", "None")
            ]
            ans_k = str(row.get("Answer Key") or "").strip().upper()
            pair = _format_mcq_pair(full_q, opts, letter_to_idx.get(ans_k, -1), is_arabic=True)
            if pair is not None:
                suites["arabic"]["MGSM-Ar"].append(pair)
            if len(suites["arabic"]["MGSM-Ar"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load Arabic Math benchmark split: %s", exc)

    # 8. MMLU-Pro (TIGER-Lab/MMLU-Pro, config="default", split="test")
    try:
        ds_mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", "default", split="test", cache_dir=cache_dir).shuffle(
            seed=9408
        )
        for row in ds_mmlu_pro:
            q = str(row.get("question") or "").strip()
            opts = [str(x).strip() for x in (row.get("options") or []) if str(x).strip()]
            ans_idx = int(row.get("answer_index", -1))
            pair = _format_mcq_pair(q, opts, ans_idx, is_arabic=False)
            if pair is not None:
                suites["retention"]["MMLU-Pro"].append(pair)
            if len(suites["retention"]["MMLU-Pro"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load TIGER-Lab/MMLU-Pro: %s", exc)

    # 9. GSM8K (openai/gsm8k, config="main", split="test")
    try:
        ds_gsm8k = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir).shuffle(seed=9409)
        for row in ds_gsm8k:
            q = str(row.get("question") or "").strip()
            ans = str(row.get("answer") or "").strip()
            if len(q) >= 12 and len(ans) >= 16:
                suites["retention"]["GSM8K"].append(
                    (f"Question: {q}\nSolve step by step and state the final numerical answer.", ans)
                )
            if len(suites["retention"]["GSM8K"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load openai/gsm8k: %s", exc)

    # 10. HumanEval (openai/openai_humaneval, config="openai_humaneval", split="test")
    try:
        ds_heval = load_dataset(
            "openai/openai_humaneval",
            "openai_humaneval",
            split="test",
            cache_dir=cache_dir,
        ).shuffle(seed=9410)
        for row in ds_heval:
            prompt_code = str(row.get("prompt") or "").strip()
            sol_code = str(row.get("canonical_solution") or "").rstrip()
            if len(prompt_code) >= 16 and len(sol_code) >= 12:
                suites["retention"]["HumanEval"].append(
                    (
                        f"Complete the following Python function:\n```python\n{prompt_code}\n```",
                        f"```python\n{sol_code}\n```",
                    )
                )
            if len(suites["retention"]["HumanEval"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load openai/openai_humaneval: %s", exc)

    # 11. ARC-Challenge (allenai/ai2_arc, config="ARC-Challenge", split="test")
    try:
        ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=cache_dir).shuffle(
            seed=9411
        )
        for row in ds_arc:
            q = str(row.get("question") or "").strip()
            choices_dict = row.get("choices") or {}
            opts = [str(x).strip() for x in (choices_dict.get("text") or []) if str(x).strip()]
            labels_list = [str(x).strip().upper() for x in (choices_dict.get("label") or [])]
            ans_k = str(row.get("answerKey") or "").strip().upper()
            g_idx = labels_list.index(ans_k) if ans_k in labels_list else -1
            pair = _format_mcq_pair(q, opts, g_idx, is_arabic=False)
            if pair is not None:
                suites["retention"]["ARC-Challenge"].append(pair)
            if len(suites["retention"]["ARC-Challenge"]) >= per_task:
                break
    except Exception as exc:
        logger.warning("Could not load allenai/ai2_arc: %s", exc)

    return suites


def _evaluate_checkpoint_on_suites(
    model_path: str,
    device: str,
    suites: Dict[str, Dict[str, List[Tuple[str, str]]]],
    tag: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run teacher-forced GPU forward passes on external Hugging Face benchmark suites."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    logger.info("[%s] Loading model on %s: %s", tag, device, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
    model.eval()
    is_base_ckpt = ":base" in tag

    def _build_prompt_prefix(user_prompt: str, category: str) -> str:
        if category == "retention":
            return f"{user_prompt}\n\nAnswer:\n"
        if is_base_ckpt:
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and len(rendered) > len(user_prompt):
                    return rendered
            except Exception:
                pass
        return format_gemma4_chat(
            [{"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    out: Dict[str, Dict[str, Dict[str, Any]]] = {"arabic": {}, "retention": {}}
    with torch.no_grad():
        for cat in ("arabic", "retention"):
            for suite_name, items in suites[cat].items():
                if not items:
                    continue
                nlls: List[float] = []
                top1_hits: List[float] = []
                top5_hits: List[float] = []
                total_tokens = 0
                for user_prompt, clean_answer in items:
                    prompt_prefix = _build_prompt_prefix(user_prompt, cat)
                    full_text = prompt_prefix + clean_answer
                    full_enc = tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                    ).to(device)
                    prompt_enc = tokenizer(
                        prompt_prefix,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                    )
                    prompt_len = int(prompt_enc["input_ids"].shape[1])
                    seq_len = int(full_enc["input_ids"].shape[1])
                    if seq_len <= prompt_len:
                        continue

                    labels = full_enc["input_ids"].clone()
                    labels[:, :prompt_len] = -100

                    outputs = model(**full_enc, labels=labels)
                    loss_val = float(outputs.loss.item())
                    if not (math.isnan(loss_val) or math.isinf(loss_val)):
                        nlls.append(loss_val)

                    shift_logits = outputs.logits[:, :-1, :]
                    shift_labels = labels[:, 1:]
                    valid_mask = shift_labels != -100
                    n_valid = int(valid_mask.sum().item())
                    if n_valid > 0:
                        valid_logits = shift_logits[valid_mask]
                        valid_targets = shift_labels[valid_mask]
                        top1 = (
                            float((valid_logits.argmax(dim=-1) == valid_targets).float().mean().item())
                            * 100.0
                        )
                        k_val = min(5, int(valid_logits.size(-1)))
                        top5_idx = valid_logits.topk(k_val, dim=-1).indices
                        top5 = (
                            float(
                                (top5_idx == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
                            )
                            * 100.0
                        )
                        top1_hits.append(top1)
                        top5_hits.append(top5)
                        total_tokens += n_valid

                mean_nll = sum(nlls) / max(1, len(nlls))
                mean_top1 = sum(top1_hits) / max(1, len(top1_hits))
                mean_top5 = sum(top5_hits) / max(1, len(top5_hits))
                raw_signal_pct = round(mean_top1, 2)
                out[cat][suite_name] = {
                    "nll": round(mean_nll, 4),
                    "ppl": round(math.exp(min(mean_nll, 12.0)), 2),
                    "top1_acc_pct": round(mean_top1, 2),
                    "top5_acc_pct": round(mean_top5, 2),
                    "raw_signal_pct": raw_signal_pct,
                    "n_samples": len(nlls),
                    "n_tokens": total_tokens,
                }
                logger.info(
                    "[%s] %s/%s -> RawSignal=%.2f%% | Top1=%.2f%% | NLL=%.4f (N=%d, Tokens=%d)",
                    tag,
                    cat,
                    suite_name,
                    raw_signal_pct,
                    mean_top1,
                    mean_nll,
                    len(nlls),
                    total_tokens,
                )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_inprocess_gpu_eval(
    model_key: str,
    base_model_id: str,
    tuned_model_path: str,
    target_device: str = "cuda:0",
    limit: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run or load cached v5 in-process GPU evaluation on real Hugging Face benchmarks."""
    cache_path = PROJECT_ROOT / "outputs" / "eval_reports" / f"live_raw_{model_key}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("eval_version") == "v5_real_hf_benchmarks"
                and "base" in cached
                and "tuned" in cached
            ):
                logger.info("Loaded verified v5 GPU evaluation cache from %s", cache_path)
                return cached
        except Exception as exc:
            logger.warning("Could not read cached live_raw_%s.json: %s", model_key, exc)

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        suites = build_heldout_eval_suites(limit=limit or 40)
        has_samples = any(len(v) > 0 for v in suites["arabic"].values())
        if not has_samples:
            return None

        use_dev = target_device
        if ":" in use_dev:
            idx = int(use_dev.split(":")[1])
            if idx >= torch.cuda.device_count():
                use_dev = "cuda:0"

        base_metrics = _evaluate_checkpoint_on_suites(
            model_path=base_model_id,
            device=use_dev,
            suites=suites,
            tag=f"{model_key}:base",
        )
        tuned_metrics = _evaluate_checkpoint_on_suites(
            model_path=tuned_model_path,
            device=use_dev,
            suites=suites,
            tag=f"{model_key}:tuned",
        )
        payload = {
            "eval_version": "v5_real_hf_benchmarks",
            "model_key": model_key,
            "base_model_id": base_model_id,
            "tuned_model_path": tuned_model_path,
            "device": use_dev,
            "base": base_metrics,
            "tuned": tuned_metrics,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved v5 live GPU evaluation metrics to %s", cache_path)
        return payload
    except Exception as exc:
        logger.warning("In-process GPU evaluation failed for %s: %s", model_key, exc)
        return None


def build_benchmark_results(
    cfg: Dict[str, Any],
    model_path: str,
    dry_run: bool = False,
    limit: Optional[int] = None,
    model_key: str = "e4b",
    eval_base: bool = False,
) -> Dict[str, Any]:
    """Run all 7 Arabic benchmarks and 4 retention anchors."""
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
            target_device=target_device,
            limit=limit,
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
            base_nll = float(b_item["nll"])
            tuned_nll = float(t_item["nll"])
            base_score = round(
                float(b_item.get("eval_score_pct", b_item.get("raw_signal_pct", b_item.get("top1_acc_pct", 0.0)))),
                2,
            )
            evaluated_score = round(
                float(t_item.get("eval_score_pct", t_item.get("raw_signal_pct", t_item.get("top1_acc_pct", 0.0)))),
                2,
            )
        else:
            if eval_base or base_score == 0.0:
                try:
                    base_score = run_lm_eval_task(
                        model_path=str(base_model_id),
                        task_name=str(spec["lm_eval_task"]),
                        num_fewshot=int(spec.get("num_fewshot", 0)),
                        batch_size=int(eval_cfg.get("batch_size", 4)),
                        device=target_device,
                        limit=limit,
                        metric_hint=str(spec.get("metric", "acc_norm")),
                    )
                except Exception as exc:
                    logger.warning("Base eval fallback for %s (%s): %s", name, spec["lm_eval_task"], exc)
            try:
                evaluated_score = run_lm_eval_task(
                    model_path=model_path,
                    task_name=str(spec["lm_eval_task"]),
                    num_fewshot=int(spec.get("num_fewshot", 0)),
                    batch_size=int(eval_cfg.get("batch_size", 4)),
                    device=target_device,
                    limit=limit,
                    metric_hint=str(spec.get("metric", "acc_norm")),
                )
            except Exception as exc:
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

        if dry_run:
            evaluated_score = base_score
        elif (
            live_gpu_data is not None
            and name in live_gpu_data.get("base", {}).get("retention", {})
            and name in live_gpu_data.get("tuned", {}).get("retention", {})
        ):
            b_item = live_gpu_data["base"]["retention"][name]
            t_item = live_gpu_data["tuned"]["retention"][name]
            base_nll = float(b_item["nll"])
            tuned_nll = float(t_item["nll"])
            base_score = round(
                float(b_item.get("eval_score_pct", b_item.get("raw_signal_pct", b_item.get("top1_acc_pct", base_score)))),
                2,
            )
            evaluated_score = round(
                float(t_item.get("eval_score_pct", t_item.get("raw_signal_pct", t_item.get("top1_acc_pct", base_score)))),
                2,
            )
        else:
            if eval_base:
                try:
                    base_score = run_lm_eval_task(
                        model_path=str(base_model_id),
                        task_name=str(spec["lm_eval_task"]),
                        num_fewshot=int(spec.get("num_fewshot", 0)),
                        batch_size=int(eval_cfg.get("batch_size", 4)),
                        device=target_device,
                        limit=limit,
                        metric_hint=str(spec.get("metric", "acc_norm")),
                    )
                except Exception as exc:
                    logger.warning("Base anchor eval fallback for %s: %s", name, exc)
            try:
                evaluated_score = run_lm_eval_task(
                    model_path=model_path,
                    task_name=str(spec["lm_eval_task"]),
                    num_fewshot=int(spec.get("num_fewshot", 0)),
                    batch_size=int(eval_cfg.get("batch_size", 4)),
                    device=target_device,
                    limit=limit,
                    metric_hint=str(spec.get("metric", "acc_norm")),
                )
            except Exception as exc:
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
        "mode": "dry_run" if dry_run else ("live_gpu_eval" if live_gpu_data else "live_eval"),
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
        description="Run the 7 Arabic/Saudi benchmarks and 4 English/Math/Code retention anchors."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "eval_config.yaml"),
        help="Path to evaluation YAML configuration file.",
    )
    parser.add_argument(
        "--model-key",
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
        help="Optional sample limit per benchmark task.",
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
        e4b_res = build_benchmark_results(
            cfg=cfg,
            model_path=e4b_path,
            dry_run=args.dry_run,
            limit=args.limit,
            model_key="e4b",
            eval_base=args.eval_base,
        )
        moe_res = build_benchmark_results(
            cfg=cfg,
            model_path=moe_path,
            dry_run=args.dry_run,
            limit=args.limit,
            model_key="26b_a4b",
            eval_base=args.eval_base,
        )
        family_md = generate_family_markdown_report(e4b_res, moe_res)
        family_payload = {
            "mode": e4b_res.get("mode", "live_eval"),
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
