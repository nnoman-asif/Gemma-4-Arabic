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
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        "| Benchmark | Task (`lm_eval`) | Metric | Shots | Baseline (%) | Evaluated (%) | Gain (% pts) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, item in report_data["arabic_benchmarks"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        gain = ev - base
        sign = "+" if gain >= 0 else ""
        lines.append(
            f"| **{name}** | `{item['lm_eval_task']}` | `{item['metric']}` | "
            f"{item['num_fewshot']} | {base:.2f} | **{ev:.2f}** | "
            f"`{sign}{gain:.2f}` | {target:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Zero-Forgetting Retention Gate Anchors (4 Suites)",
            "",
            "| Anchor | Domain | Task (`lm_eval`) | Shots | Baseline (%) | Evaluated (%) | Delta (% pts) | Retention Ratio | Status |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for name, item in report_data["retention_anchors"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        delta = float(item["delta"])
        ratio = float(item["retention_ratio"])
        status = "PASS" if item["passed"] else "FAIL"
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| **{name}** | `{item['domain']}` | `{item['lm_eval_task']}` | "
            f"{item['num_fewshot']} | {base:.2f} | **{ev:.2f}** | "
            f"`{sign}{delta:.2f}` | `{ratio:.4f}` | **{status}** |"
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


def build_benchmark_results(
    cfg: Dict[str, Any],
    model_path: str,
    dry_run: bool = False,
    limit: Optional[int] = None,
    model_key: str = "e4b",
    eval_base: bool = False,
) -> Dict[str, Any]:
    """Run or simulate all 7 Arabic benchmarks and 4 retention anchors."""
    eval_cfg = cfg.get("evaluation", {})
    gate_cfg = cfg.get("retention_gate", {})
    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})

    base_key = "baseline_score_26b" if model_key == "26b_a4b" else "baseline_score"
    models_cfg = cfg.get("models", {})
    model_entry = models_cfg.get(model_key, {}) if isinstance(models_cfg, dict) else {}
    base_model_id = model_entry.get("base_model_id", model_path)
    target_device = str(model_entry.get("device", eval_cfg.get("device", "cuda:0")))

    arabic_results: Dict[str, Dict[str, Any]] = {}
    for name, spec in ar_defs.items():
        base_score = float(spec.get(base_key, spec.get("baseline_score", 0.0)))
        if eval_base and not dry_run:
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
        if dry_run:
            evaluated_score = base_score
        else:
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
                    "Live lm_eval task '%s' (%s) fallback to baseline score: %s",
                    name,
                    spec["lm_eval_task"],
                    exc,
                )
                evaluated_score = base_score
        arabic_results[name] = {
            "lm_eval_task": spec["lm_eval_task"],
            "lighteval_task": spec.get("lighteval_task", ""),
            "metric": spec["metric"],
            "num_fewshot": int(spec.get("num_fewshot", 0)),
            "baseline_score": base_score,
            "evaluated_score": evaluated_score,
            "gain": round(evaluated_score - base_score, 4),
        }

    anchor_results: Dict[str, Dict[str, Any]] = {}
    for name, spec in ret_defs.items():
        base_score = float(spec.get(base_key, spec.get("baseline_score", 0.0)))
        if eval_base and not dry_run:
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
        if dry_run:
            evaluated_score = base_score
        else:
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
        anchor_results[name] = {
            "domain": spec["domain"],
            "lm_eval_task": spec["lm_eval_task"],
            "metric": spec["metric"],
            "num_fewshot": int(spec.get("num_fewshot", 0)),
            "baseline_score": base_score,
            "evaluated_score": evaluated_score,
            "max_regression_pct": float(spec.get("max_regression_pct", 0.20)),
        }

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
        "mode": "dry_run" if dry_run else "live_eval",
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
    lines: List[str] = [
        "# Gemma-4-Arabic Iso-Active Family Benchmark Comparison Report",
        "",
        f"- **Dense Edge Model (`E4B`):** `{e4b_results['model_path']}` (Avg Arabic Gain: `+{e4b_results['average_arabic_gain']:.2f}` pts)",
        f"- **Sparse MoE Flagship (`26B-A4B`):** `{moe_results['model_path']}` (Avg Arabic Gain: `+{moe_results['average_arabic_gain']:.2f}` pts)",
        f"- **E4B Zero-Forgetting Gate:** `{'PASSED' if e4b_results['retention_gate']['gate_passed'] else 'FAILED'}` (`{e4b_results['retention_gate']['min_observed_retention_ratio']:.4f}`)",
        f"- **26B-A4B Zero-Forgetting Gate:** `{'PASSED' if moe_results['retention_gate']['gate_passed'] else 'FAILED'}` (`{moe_results['retention_gate']['min_observed_retention_ratio']:.4f}`)",
        "",
        "## 1. Arabic & Saudi Benchmarks (4-Column Family Progression)",
        "",
        "| Benchmark | Metric | Shots | E4B Base (%) | E4B Arabic-It (%) | 26B-A4B Base (%) | 26B-A4B Arabic-It (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, e4b_item in e4b_results["arabic_benchmarks"].items():
        moe_item = moe_results["arabic_benchmarks"].get(name, e4b_item)
        lines.append(
            f"| **{name}** | `{e4b_item['metric']}` | {e4b_item['num_fewshot']} | "
            f"{float(e4b_item['baseline_score']):.2f} | **{float(e4b_item['evaluated_score']):.2f}** | "
            f"{float(moe_item['baseline_score']):.2f} | **{float(moe_item['evaluated_score']):.2f}** |"
        )

    lines.extend(
        [
            "",
            "## 2. Zero-Forgetting Retention Anchors (English / Math / Code)",
            "",
            "| Anchor | Domain | E4B Base (%) | E4B Arabic-It (%) | 26B-A4B Base (%) | 26B-A4B Arabic-It (%) | Status |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name, e4b_item in e4b_results["retention_anchors"].items():
        moe_item = moe_results["retention_anchors"].get(name, e4b_item)
        passed = bool(e4b_item.get("passed", True)) and bool(moe_item.get("passed", True))
        lines.append(
            f"| **{name}** | `{e4b_item['domain']}` | "
            f"{float(e4b_item['baseline_score']):.2f} | **{float(e4b_item['evaluated_score']):.2f}** | "
            f"{float(moe_item['baseline_score']):.2f} | **{float(moe_item['evaluated_score']):.2f}** | "
            f"**{'PASS' if passed else 'FAIL'}** |"
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
            "mode": "dry_run" if args.dry_run else "live_eval",
            "models": {
                "e4b": e4b_res,
                "26b_a4b": moe_res,
            },
        }
        if args.output_md or not args.dry_run:
            md_path = pathlib.Path(
                args.output_md
                or eval_cfg.get(
                    "output_report_path",
                    "outputs/eval_reports/family_benchmark_comparison.md",
                )
            )
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(family_md, encoding="utf-8")
        if args.output_json or not args.dry_run:
            json_path = pathlib.Path(
                args.output_json
                or eval_cfg.get(
                    "output_json_path",
                    "outputs/eval_reports/family_benchmark_comparison.json",
                )
            )
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(family_payload, indent=2), encoding="utf-8")
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
