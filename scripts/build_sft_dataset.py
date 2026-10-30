"""CLI script to build, validate, and save the mixed Stage 1 SFT dataset to disk.

Supports:
- All 5 Arabic SFT sources (`VohoAI/voho-saudi-dialogues`, `SaudiArabicLaws` `"moj"` & `"boe"`,
  `FreedomIntelligence/AceGPT-v2-tuning`, `Omartificial-Intelligence-Space/Arabic-Math-SFT`
  with `.remove_columns(["image"])`, and `CohereForAI/aya_collection_language_split`).
- Both Replay Buffer anchors (`HuggingFaceH4/ultrachat_200k` and `nvidia/OpenMathInstruct-2`).
- Strict 75% Arabic / 25% Replay stratified mixing and Gemma 4 thinking-block verification.
- `--dry-run` and `--max-samples` flags for fast validation and smoke testing.
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_template import verify_thinking_block_preserved
from src.constants import SFT_ARABIC_DATASETS, SFT_REPLAY_DATASETS
from src.data_normalizer import (
    drop_pil_image_columns,
    load_and_normalize_arabic_sft_source,
    normalize_acegpt_v2_row,
    normalize_arabic_math_row,
    normalize_aya_arabic_row,
    normalize_saudi_laws_row,
    normalize_voho_saudi_row,
)
from src.replay_mixer import (
    compute_mix_allocation,
    load_and_normalize_replay_source,
    mix_arabic_and_replay_samples,
    normalize_openmath_instruct_row,
    normalize_ultrachat_row,
)

logger = logging.getLogger(__name__)


def build_synthetic_dry_run_pools() -> Dict[str, List[Dict[str, Any]]]:
    """Generate deterministic schema-accurate samples for all 7 Stage 1 SFT sources."""
    class _MockMathDataset:
        column_names = ["image", "problem", "solution"]

        def __init__(self, rows: List[Dict[str, Any]]) -> None:
            self.rows = rows

        def remove_columns(self, cols: List[str]) -> "_MockMathDataset":
            cleaned = [
                {k: v for k, v in r.items() if k not in cols} for r in self.rows
            ]
            return _MockMathDataset(cleaned)

        def __iter__(self):
            return iter(self.rows)

    voho_sample = normalize_voho_saudi_row(
        {
            "messages_json": json.dumps(
                [
                    {
                        "role": "user",
                        "content": "User: يا هلا، كيف أقدر أحجز موعد في الأحوال المدنية عبر أبشر؟",
                    },
                    {
                        "role": "assistant",
                        "content": "المساعد: هلا بك! ادخل منصة أبشر أفراد، ثم اختر المواعيد، ثم الأحوال المدنية وحدد الخدمة والفرع.",
                    },
                ],
                ensure_ascii=False,
            )
        }
    )

    moj_sample = normalize_saudi_laws_row(
        {
            "doc_title": "نظام العمل السعودي",
            "article_title": "المادة الرابعة والسبعون",
            "text": "ينتهي عقد العمل في الأحوال المنصوص عليها في النظام باتفاق الطرفين كتابة.",
        },
        config_name="moj",
    )

    boe_sample = normalize_saudi_laws_row(
        {
            "law": {"title_ar": "نظام الشركات السعودي"},
            "article": {
                "article_title_norm": "المادة الأولى",
                "text": "الشركة كيان قانوني يؤسس وفقا لأحكام النظام بناء على عقد تأسيس أو نظام أساس.",
            },
            "retrieval": {"contextual_text": "نظام الشركات - المادة الأولى"},
        },
        config_name="boe",
    )

    acegpt_sample = normalize_acegpt_v2_row(
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "ما الفرق بين المبتدأ والخبر في النحو العربي؟",
                },
                {
                    "from": "gpt",
                    "value": "المبتدأ هو الاسم المرفوع المتحدث عنه في أول الجملة، والخبر هو الجزء المتمم للفائدة معه.",
                },
            ]
        }
    )

    raw_math_ds = _MockMathDataset(
        [
            {
                "image": "MOCK_PIL_IMAGE_OBJECT",
                "problem": "إذا كان محيط مربع يساوي 36 سم، فما مساحته؟",
                "solution": "طول الضلع = 36 / 4 = 9 سم.\nمساحة المربع = 9 * 9 = 81 سم مربع.",
            }
        ]
    )
    cleaned_math_ds = drop_pil_image_columns(raw_math_ds, ["image"])
    math_rows = list(cleaned_math_ds)
    assert "image" not in math_rows[0], "image column must be dropped before normalization"
    math_sample = normalize_arabic_math_row(math_rows[0])

    aya_sample = normalize_aya_arabic_row(
        {
            "inputs": "استخرج حالة الشحنة رقم 9081 باستخدام أداة تتبع الشحنات.",
            "targets": "الشحنة رقم 9081 في طريقها للتسليم في مدينة الرياض.",
            "tools": [
                {
                    "name": "track_shipment",
                    "parameters": {"shipment_id": "string"},
                }
            ],
            "tool_call": {
                "name": "track_shipment",
                "arguments": {"shipment_id": "9081"},
            },
        }
    )

    ultrachat_sample = normalize_ultrachat_row(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function to compute the greatest common divisor of two integers.",
                },
                {
                    "role": "assistant",
                    "content": "def gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a",
                },
            ]
        }
    )

    openmath_sample = normalize_openmath_instruct_row(
        {
            "problem": "Find the derivative of f(x) = 4x^3 - 6x + 2.",
            "generated_solution": "Applying the power rule term by term: d/dx(4x^3) = 12x^2, d/dx(-6x) = -6, d/dx(2) = 0.",
            "expected_answer": "12x^2 - 6",
        }
    )

    arabic_pool = [
        s
        for s in [
            voho_sample,
            moj_sample,
            boe_sample,
            acegpt_sample,
            math_sample,
            aya_sample,
        ]
        if s is not None
    ]
    replay_pool = [
        s for s in [ultrachat_sample, openmath_sample] if s is not None
    ]
    return {"arabic": arabic_pool, "replay": replay_pool}


def validate_sft_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify every normalized SFT sample preserves Gemma 4 thinking-block invariants."""
    thinking_true_count = 0
    thinking_false_count = 0

    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        enable_thinking = bool(sample.get("enable_thinking", False))
        if not verify_thinking_block_preserved(text, enable_thinking=enable_thinking):
            raise ValueError(
                f"Sample at index {idx} (source={sample.get('source')!r}) failed "
                f"Gemma 4 thinking-block verification (enable_thinking={enable_thinking})."
            )
        if enable_thinking:
            thinking_true_count += 1
        else:
            thinking_false_count += 1

    return {
        "validated_samples": len(samples),
        "thinking_enabled_count": thinking_true_count,
        "thinking_disabled_count": thinking_false_count,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for building the Stage 1 SFT dataset."""
    parser = argparse.ArgumentParser(
        description="Build, validate, and save the Stage 1 SFT dataset (75% Arabic / 25% Replay)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "sft_config.yaml"),
        help="Path to Stage 1 SFT YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the processed Hugging Face dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples to build across Arabic and Replay sources.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast validation of all 7 SFT normalizers and 75/25 stratified mixer.",
    )
    parser.add_argument(
        "--live-stream",
        action="store_true",
        help="When combined with --dry-run, stream live rows from Hugging Face Hub.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for deterministic 75/25 stratified mixing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for Stage 1 SFT dataset construction and verification."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "urllib3", "fsspec"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    args = parse_args(argv)

    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ds_cfg = cfg.get("datasets", {})

    arabic_ratio = float(ds_cfg.get("arabic_mix_ratio", 0.75))
    replay_ratio = float(ds_cfg.get("replay_mix_ratio", 0.25))
    total_target = int(
        args.max_samples
        if args.max_samples is not None
        else (12 if args.dry_run else ds_cfg.get("total_target_samples", 100000))
    )
    output_dir = pathlib.Path(
        args.output_dir or ds_cfg.get("output_dataset_dir", "data/sft_mixed")
    )

    allocation = compute_mix_allocation(
        total_samples=total_target,
        arabic_ratio=arabic_ratio,
        replay_ratio=replay_ratio,
    )

    if args.dry_run and not args.live_stream:
        pools = build_synthetic_dry_run_pools()
        arabic_samples = pools["arabic"]
        replay_samples = pools["replay"]
    else:
        arabic_samples = []
        for src_key, count in allocation["arabic_per_source"].items():
            logger.info("Streaming %d samples from Arabic SFT source '%s'...", count, src_key)
            arabic_samples.extend(
                load_and_normalize_arabic_sft_source(
                    source_key=src_key,
                    max_samples=count,
                    streaming=True,
                )
            )
        replay_samples = []
        for src_key, count in allocation["replay_per_source"].items():
            logger.info("Streaming %d samples from Replay source '%s'...", count, src_key)
            replay_samples.extend(
                load_and_normalize_replay_source(
                    source_key=src_key,
                    max_samples=count,
                    streaming=True,
                )
            )

    mixed_samples, mix_stats = mix_arabic_and_replay_samples(
        arabic_samples=arabic_samples,
        replay_samples=replay_samples,
        arabic_ratio=arabic_ratio,
        replay_ratio=replay_ratio,
        seed=args.seed,
    )
    validation_stats = validate_sft_samples(mixed_samples)

    summary = {
        "mode": "dry_run" if args.dry_run else "production",
        "arabic_sources_configured": list(SFT_ARABIC_DATASETS.keys()),
        "replay_sources_configured": list(SFT_REPLAY_DATASETS.keys()),
        "allocation": allocation,
        "mix_stats": mix_stats,
        "validation_stats": validation_stats,
        "output_dir": str(output_dir),
    }

    if not args.dry_run:
        from datasets import Dataset

        output_dir.mkdir(parents=True, exist_ok=True)
        hf_dataset = Dataset.from_list(mixed_samples)
        hf_dataset.save_to_disk(str(output_dir))
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d mixed SFT examples to %s", len(mixed_samples), output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
