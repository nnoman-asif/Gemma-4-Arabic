"""CLI script to build, validate, and save the Stage 2 DPO dataset to disk.

Supports:
1. `HeshamHaroon/arabic-dialect-dpo` (`name="saudi"`, explicitly validated).
2. `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` (unifying `rejected[0]["content"]`
   from `chosen[0]["content"]` so both preference branches share the exact same prompt).
3. Gemma 4 thinking-block preservation check on all `chosen` and `rejected` completions.
4. `--dry-run` and `--max-samples` flags for fast validation and smoke testing.
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
from src.constants import ARABIC_DIALECT_DPO_REPO, DPO_DATASETS
from src.dpo_formatter import (
    load_and_normalize_dpo_source,
    mix_dpo_preference_datasets,
    normalize_arabic_dialect_dpo_row,
    normalize_arcee_arabic_dpo_row,
    validate_dpo_load_config,
)

logger = logging.getLogger(__name__)


def build_synthetic_dpo_dry_run_pools() -> Dict[str, List[Dict[str, Any]]]:
    """Generate deterministic schema-accurate preference pairs for Stage 2 DPO dry-run."""
    validated_cfg = validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "saudi")
    assert validated_cfg == "saudi"

    saudi_samples = [
        normalize_arabic_dialect_dpo_row(
            {
                "prompt": "وش الإجراءات المطلوبة لإصدار رخصة قيادة في السعودية؟",
                "chosen": "تبدأ بحجز موعد في مدرسة القيادة عبر منصة أبشر، وإجراء الفحص الطبي، ثم اجتياز التدريب النظري والعملي.",
                "rejected": "ما عندي فكرة عن الموضوع.",
                "dialect": "saudi",
            }
        ),
        normalize_arabic_dialect_dpo_row(
            {
                "prompt": "وين أقدر أزور أماكن تراثية في الرياض؟",
                "chosen": "تقدر تزور حي الطريف التاريخي في الدرعية، وقصر المصمك، ومركز الملك عبدالعزيز التاريخي بالمربع.",
                "rejected": "لا توجد أماكن معروفة.",
                "dialect": "saudi",
            }
        ),
    ]

    arcee_samples = [
        normalize_arcee_arabic_dpo_row(
            {
                "chosen": [
                    {
                        "role": "user",
                        "content": "اشرح أهمية التوازن البيئي وكيفية المحافظة عليه.",
                    },
                    {
                        "role": "assistant",
                        "content": "التوازن البيئي يضمن استدامة الموارد الطبيعية وتكامل السلاسل الغذائية، ويتم الحفاظ عليه بتقليل الانبعاثات وحماية التنوع الحيوي.",
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "ترجمة بديلة مختلفة للسؤال يجب توحيدها مع نص السؤال المختار.",
                    },
                    {
                        "role": "assistant",
                        "content": "التوازن البيئي غير مهم.",
                    },
                ],
            }
        ),
        normalize_arcee_arabic_dpo_row(
            {
                "chosen": [
                    {
                        "role": "user",
                        "content": "ما هي خطوات حل المعادلة الخطية من الدرجة الأولى؟",
                    },
                    {
                        "role": "assistant",
                        "content": "نجمع الحدود المتشابهة، ثم ننقل الثوابت إلى طرف والمتغيرات إلى الطرف الآخر، وأخيرا نقسم على معامل المتغير.",
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "سؤال مختلف في صياغته الأولية قبل التوحيد.",
                    },
                    {
                        "role": "assistant",
                        "content": "نخمن الرقم مباشرة.",
                    },
                ],
            }
        ),
        normalize_arcee_arabic_dpo_row(
            {
                "chosen": [
                    {
                        "role": "user",
                        "content": "ما دور هيئة البيانات والذكاء الاصطناعي (سدايا)؟",
                    },
                    {
                        "role": "assistant",
                        "content": "تتولى سدايا قيادة الأجندة الوطنية للبيانات والذكاء الاصطناعي في المملكة وتطوير المنصات الحكومية الذكية.",
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "نص سؤال غير مطابق في الفرع المرفوض.",
                    },
                    {
                        "role": "assistant",
                        "content": "لا أعلم.",
                    },
                ],
            }
        ),
    ]

    return {
        "saudi": [s for s in saudi_samples if s is not None],
        "arcee": [s for s in arcee_samples if s is not None],
    }


def validate_dpo_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify prompt unification and Gemma 4 thinking-block preservation on DPO pairs."""
    unified_prompt_matches = 0
    for idx, pair in enumerate(pairs):
        enable_thinking = bool(pair.get("enable_thinking", False))
        chosen_str = pair.get("chosen", "")
        rejected_str = pair.get("rejected", "")

        if not verify_thinking_block_preserved(chosen_str, enable_thinking=enable_thinking):
            raise ValueError(f"DPO pair {idx} chosen completion missing thinking block.")
        if not verify_thinking_block_preserved(rejected_str, enable_thinking=enable_thinking):
            raise ValueError(f"DPO pair {idx} rejected completion missing thinking block.")

        c_msgs = pair.get("chosen_messages", [])
        r_msgs = pair.get("rejected_messages", [])
        if not c_msgs or not r_msgs or c_msgs[0].get("content") != r_msgs[0].get("content"):
            raise ValueError(
                f"DPO pair {idx} failed prompt unification between chosen[0] and rejected[0]."
            )
        unified_prompt_matches += 1

    return {
        "validated_pairs": len(pairs),
        "unified_prompt_matches": unified_prompt_matches,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for building the Stage 2 DPO dataset."""
    parser = argparse.ArgumentParser(
        description="Build, validate, and save the Stage 2 DPO preference dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "dpo_config.yaml"),
        help="Path to Stage 2 DPO YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the processed Hugging Face DPO dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total preference pairs to build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast validation of DPO normalizers, Saudi config check, and prompt unification.",
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
        help="Random seed for shuffling preference pairs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for Stage 2 DPO dataset construction and verification."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)

    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ds_cfg = cfg.get("datasets", {})
    pref_sources = ds_cfg.get("preference_sources", {})

    saudi_weight = float(
        pref_sources.get("arabic_dialect_dpo_saudi", {}).get("weight", 0.40)
    )
    arcee_weight = float(
        pref_sources.get("arabic_orpo_dpo_mix_40k", {}).get("weight", 0.60)
    )
    total_target = int(
        args.max_samples
        if args.max_samples is not None
        else (5 if args.dry_run else ds_cfg.get("total_target_pairs", 20000))
    )
    output_dir = pathlib.Path(
        args.output_dir or ds_cfg.get("output_dataset_dir", "data/dpo_mixed")
    )

    saudi_target = max(1, int(round(total_target * saudi_weight)))
    arcee_target = max(1, total_target - saudi_target)

    if args.dry_run and not args.live_stream:
        pools = build_synthetic_dpo_dry_run_pools()
        saudi_samples = pools["saudi"][:saudi_target]
        arcee_samples = pools["arcee"][:arcee_target]
    else:
        saudi_samples = load_and_normalize_dpo_source(
            source_key="arabic_dialect_dpo_saudi",
            max_samples=saudi_target,
            streaming=True,
        )
        arcee_samples = load_and_normalize_dpo_source(
            source_key="arabic_orpo_dpo_mix_40k",
            max_samples=arcee_target,
            streaming=True,
        )

    mixed_pairs, mix_stats = mix_dpo_preference_datasets(
        saudi_dpo_samples=saudi_samples,
        arcee_dpo_samples=arcee_samples,
        saudi_weight=saudi_weight,
        arcee_weight=arcee_weight,
        seed=args.seed,
    )
    validation_stats = validate_dpo_pairs(mixed_pairs)

    summary = {
        "mode": "dry_run" if args.dry_run else "production",
        "dpo_sources_configured": list(DPO_DATASETS.keys()),
        "saudi_config_verified": validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "saudi"),
        "mix_stats": mix_stats,
        "validation_stats": validation_stats,
        "output_dir": str(output_dir),
    }

    if not args.dry_run:
        from datasets import Dataset

        output_dir.mkdir(parents=True, exist_ok=True)
        hf_dataset = Dataset.from_list(mixed_pairs)
        hf_dataset.save_to_disk(str(output_dir))
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d DPO preference pairs to %s", len(mixed_pairs), output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
