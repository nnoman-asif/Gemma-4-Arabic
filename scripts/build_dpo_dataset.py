"""CLI script to build, decontaminate, validate, or pull the V3 Stage 2 DPO dataset (6,000 pairs).

Supports:
1. `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` (`arabic_orpo_dpo_mix_40k` non-thinking MSA — 2,600 pairs
   + `arabic_reasoning_thinking_dpo` `<|channel>thought` CoT — 1,600 pairs; 70% Arabic total).
2. `argilla/distilabel-math-preference-dpo` (`distilabel_math_thinking_dpo` `<|channel>thought` CoT — 1,000 pairs)
   + `argilla/ultrafeedback-binarized-preferences-cleaned` (`ultrafeedback_english_dpo` — 800 pairs; 30% English/STEM).
3. Cross-stage prompt deduplication against `data/sft_mixed` (0% SFT-to-DPO prompt overlap), strict sentence
   terminal punctuation, and tight length balance (`0.55..1.75`) to prevent DPO verbosity drift.
4. `--pull-from-hub` fast path to download the pre-built verified preference corpus from
   `noman-asif/Arabic-Preference-DPO-6k` in seconds on rented GPU instances.
"""

import argparse
import hashlib
import json
import logging
import pathlib
import shutil
import sys
from typing import Any, Dict, List, Optional, Set

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_template import verify_thinking_block_preserved
from src.constants import (
    ARABIC_DIALECT_DPO_REPO,
    DPO_DATASETS,
    PRIMARY_MODEL_ID,
    PUBLISHED_DPO_DATASET_REPO,
    load_dotenv_if_present,
)
from src.data_normalizer import BenchmarkDecontaminator
from src.dpo_formatter import (
    load_and_normalize_dpo_source,
    mix_dpo_preference_datasets,
    normalize_arabic_dialect_dpo_row,
    normalize_arabic_reasoning_thinking_dpo_row,
    normalize_arcee_arabic_dpo_row,
    normalize_distilabel_math_thinking_dpo_row,
    normalize_ultrafeedback_dpo_row,
    validate_dpo_load_config,
)

logger = logging.getLogger(__name__)


def _load_optional_gemma4_tokenizer(model_id: str = PRIMARY_MODEL_ID) -> Optional[Any]:
    """Load the Gemma 4 tokenizer for exact token-length filtering when available."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id)
    except Exception:
        return None


def build_synthetic_dpo_dry_run_pools() -> Dict[str, List[Dict[str, Any]]]:
    """Generate deterministic schema-accurate preference pairs for all 3 Stage 2 DPO sources."""
    validated_cfg = validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "saudi")
    assert validated_cfg == "saudi"

    saudi_samples = [
        normalize_arabic_dialect_dpo_row(
            {
                "prompt": "وش الإجراءات المطلوبة لإصدار رخصة قيادة في السعودية؟",
                "chosen": "تبدأ بحجز موعد في مدرسة القيادة عبر منصة أبشر، وإجراء الفحص الطبي المعتمد، ثم اجتياز التدريب النظري والعملي.",
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
                        "content": "التوازن البيئي غير مهم في الطبيعة.",
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
                        "content": "نجمع الحدود المتشابهة، ثم ننقل الثوابت إلى طرف والمتغيرات إلى الطرف الآخر، وأخيراً نقسم على معامل المتغير.",
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "سؤال مختلف في صياغته الأولية قبل التوحيد.",
                    },
                    {
                        "role": "assistant",
                        "content": "نخمن الرقم مباشرة دون خطوات.",
                    },
                ],
            }
        ),
    ]

    english_samples = [
        normalize_ultrafeedback_dpo_row(
            {
                "prompt": "Explain the time and space complexity of merge sort on an array of n elements.",
                "chosen-rating": 5.0,
                "rejected-rating": 3.0,
                "chosen": [
                    {
                        "role": "user",
                        "content": "Explain the time and space complexity of merge sort on an array of n elements.",
                    },
                    {
                        "role": "assistant",
                        "content": "Merge sort runs in O(n log n) time in the best, average, and worst cases because it halves the array log2(n) times and merges n elements per level, using O(n) auxiliary space.",
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "Explain the time and space complexity of merge sort on an array of n elements.",
                    },
                    {
                        "role": "assistant",
                        "content": "Merge sort takes O(n^2) time and uses no extra memory.",
                    },
                ],
            }
        )
    ]

    ar_thinking_samples = [
        normalize_arabic_reasoning_thinking_dpo_row(
            {
                "chosen": [
                    {
                        "role": "user",
                        "content": "اشرح خطوة بخطوة كيفية حساب مساحة مثلث قاعدته 10 سم وارتفاعه 6 سم.",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "الخطوة الأولى: نستخدم قانون مساحة المثلث وهو نصف طول القاعدة مضروباً في الارتفاع العمودي.\n\n"
                            "الخطوة الثانية: نعوض القيم المعطاة في المسألة وهي القاعدة 10 سم والارتفاع 6 سم فنحصل على 0.5 * 10 * 6.\n\n"
                            "إذن مساحة المثلث تساوي 30 سنتيمتراً مربعاً."
                        ),
                    },
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": "اشرح خطوة بخطوة كيفية حساب مساحة مثلث قاعدته 10 سم وارتفاعه 6 سم.",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "لحساب مساحة المثلث نقوم أولاً بجمع طول القاعدة مع الارتفاع ثم نضرب الناتج في اثنين.\n\n"
                            "عند تعويض القيم 10 و 6 نحصل على مجموع قدره 16 سم.\n\n"
                            "وبالتالي تكون المساحة هي 60 سنتيمتراً مربعاً."
                        ),
                    },
                ],
            }
        )
    ]

    en_thinking_samples = [
        normalize_distilabel_math_thinking_dpo_row(
            {
                "instruction": "Solve 3x + 5 = 20 step by step and verify the solution.",
                "chosen_rating": 9.5,
                "rejected_rating": 6.0,
                "chosen_response": (
                    "Step 1: Subtract 5 from both sides of the equation 3x + 5 = 20 to isolate the term 3x = 15.\n\n"
                    "Step 2: Divide both sides by 3 to obtain x = 5, and verify that 3(5) + 5 = 20 holds.\n\n"
                    "Therefore, the unique solution is x = 5."
                ),
                "rejected_response": (
                    "Step 1: Add 5 to both sides of the equation 3x + 5 = 20 so that 3x = 25.\n\n"
                    "Step 2: Divide both sides by 5 to isolate x on the left-hand side.\n\n"
                    "Therefore, the solution is x = 8."
                ),
            }
        )
    ]

    return {
        "saudi": [s for s in saudi_samples if s is not None],
        "arcee": [s for s in arcee_samples if s is not None],
        "arabic_thinking": [s for s in ar_thinking_samples if s is not None],
        "english": [s for s in english_samples if s is not None],
        "english_thinking": [s for s in en_thinking_samples if s is not None],
    }


def _collect_sft_prompt_hashes(sft_dir: pathlib.Path) -> Set[bytes]:
    """Load user prompt hashes from `data/sft_mixed` so Stage 2 DPO has 0% prompt overlap with Stage 1 SFT."""
    hashes: Set[bytes] = set()
    if not (sft_dir / "dataset_info.json").exists():
        return hashes
    try:
        from datasets import load_from_disk

        sft_ds = load_from_disk(str(sft_dir))
        for idx in range(len(sft_ds)):
            msgs = sft_ds[idx].get("messages") or []
            for msg in msgs:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    txt = str(msg.get("content") or "").strip().lower()
                    if txt:
                        hashes.add(
                            hashlib.blake2b(
                                txt.encode("utf-8", errors="ignore"),
                                digest_size=12,
                            ).digest()
                        )
                    break
        logger.info(
            "Loaded %d Stage 1 SFT user-prompt hashes from %s for cross-stage DPO deduplication.",
            len(hashes),
            sft_dir,
        )
    except Exception as exc:
        logger.warning("Could not load SFT prompt hashes from %s: %s", sft_dir, exc)
    return hashes


def validate_dpo_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify prompt unification, uniform message schema, and Gemma 4 native token invariants on DPO pairs."""
    unified_prompt_matches = 0
    thinking_pairs = 0
    for idx, pair in enumerate(pairs):
        enable_thinking = bool(pair.get("enable_thinking", False))
        chosen_str = str(pair.get("chosen") or "")
        rejected_str = str(pair.get("rejected") or "")

        if not verify_thinking_block_preserved(chosen_str, enable_thinking=enable_thinking):
            raise ValueError(f"DPO pair {idx} chosen completion failed Gemma 4 token verification.")
        if not verify_thinking_block_preserved(rejected_str, enable_thinking=enable_thinking):
            raise ValueError(f"DPO pair {idx} rejected completion failed Gemma 4 token verification.")

        c_msgs = pair.get("chosen_messages", [])
        r_msgs = pair.get("rejected_messages", [])
        if not c_msgs or not r_msgs or c_msgs[0].get("content") != r_msgs[0].get("content"):
            raise ValueError(
                f"DPO pair {idx} failed prompt unification between chosen[0] and rejected[0]."
            )

        p_msgs = pair.get("prompt_messages", [])
        co_msgs = pair.get("chosen_only_messages", [])
        ro_msgs = pair.get("rejected_only_messages", [])
        if not p_msgs or not co_msgs or not ro_msgs:
            raise ValueError(f"DPO pair {idx} missing conversational TRL message lists.")

        if enable_thinking:
            if not co_msgs[0].get("reasoning_content") or not ro_msgs[0].get("reasoning_content"):
                raise ValueError(f"DPO thinking pair {idx} missing reasoning_content.")
            thinking_pairs += 1

        unified_prompt_matches += 1

    return {
        "validated_pairs": len(pairs),
        "unified_prompt_matches": unified_prompt_matches,
        "thinking_pairs": thinking_pairs,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for building or pulling the Stage 2 DPO dataset."""
    parser = argparse.ArgumentParser(
        description="Build, decontaminate, validate, or pull the Stage 2 DPO preference dataset."
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
        "--sft-dataset-dir",
        type=str,
        default="data/sft_mixed",
        help="Optional Stage 1 SFT dataset path to exclude overlapping user prompts.",
    )
    parser.add_argument(
        "--max-samples",
        "--total-pairs",
        dest="max_samples",
        type=int,
        default=None,
        help="Maximum total preference pairs to build.",
    )
    parser.add_argument(
        "--pull-from-hub",
        action="store_true",
        help=f"Pull pre-built verified DPO dataset directly from `{PUBLISHED_DPO_DATASET_REPO}`.",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=PUBLISHED_DPO_DATASET_REPO,
        help="Hugging Face dataset repository ID when using --pull-from-hub.",
    )
    parser.add_argument(
        "--skip-decontamination",
        action="store_true",
        help="Skip live benchmark test-set loading for 13-gram decontamination.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild or re-download even if output-dir already exists.",
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
    """Entry point for Stage 2 DPO dataset construction, decontamination, and verification."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv_if_present()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpx2", "httpcore", "urllib3", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    args = parse_args(argv)

    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ds_cfg = cfg.get("datasets", {})
    pref_sources = ds_cfg.get("preference_sources", {})

    base_total = sum(max(0, s.target_samples) for s in DPO_DATASETS.values()) or 6000
    default_weights = {
        k: round(max(0, s.target_samples) / float(base_total), 4)
        for k, s in DPO_DATASETS.items()
    }

    saudi_weight = float(
        pref_sources.get("arabic_dialect_dpo_saudi", {}).get(
            "weight", default_weights.get("arabic_dialect_dpo_saudi", 0.0)
        )
    )
    arcee_weight = float(
        pref_sources.get("arabic_orpo_dpo_mix_40k", {}).get(
            "weight", default_weights.get("arabic_orpo_dpo_mix_40k", 0.4333)
        )
    )
    ar_think_weight = float(
        pref_sources.get("arabic_reasoning_thinking_dpo", {}).get(
            "weight", default_weights.get("arabic_reasoning_thinking_dpo", 0.2667)
        )
    )
    english_weight = float(
        pref_sources.get("ultrafeedback_english_dpo", {}).get(
            "weight", default_weights.get("ultrafeedback_english_dpo", 0.1333)
        )
    )
    en_think_weight = float(
        pref_sources.get("distilabel_math_thinking_dpo", {}).get(
            "weight", default_weights.get("distilabel_math_thinking_dpo", 0.1667)
        )
    )

    total_target = int(
        args.max_samples
        if args.max_samples is not None
        else (6 if args.dry_run else ds_cfg.get("total_target_pairs", 6000))
    )
    output_dir = pathlib.Path(
        args.output_dir or ds_cfg.get("output_dataset_dir", "data/dpo_mixed")
    )

    if not args.dry_run and not args.force and (output_dir / "dataset_info.json").exists():
        logger.info("Stage 2 DPO dataset already exists at %s; skipping (pass --force to rebuild).", output_dir)
        return 0

    if args.pull_from_hub and not args.dry_run:
        from datasets import load_dataset

        logger.info("Pulling pre-built verified DPO dataset from Hugging Face Hub: %s ...", args.hub_repo)
        hf_ds = load_dataset(args.hub_repo, split="train")
        if args.max_samples is not None and args.max_samples < len(hf_ds):
            hf_ds = hf_ds.select(range(args.max_samples))
        rows_for_check = [hf_ds[i] for i in range(len(hf_ds))]
        validation_stats = validate_dpo_pairs(rows_for_check)
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        hf_ds.save_to_disk(str(output_dir))
        summary = {
            "mode": "pulled_from_hub",
            "hub_repo": args.hub_repo,
            "total_pairs": len(hf_ds),
            "validation_stats": validation_stats,
            "output_dir": str(output_dir),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d verified DPO pairs from '%s' to %s", len(hf_ds), args.hub_repo, output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    saudi_target = int(round(total_target * saudi_weight)) if saudi_weight > 0 else 0
    ar_think_target = max(1, int(round(total_target * ar_think_weight))) if ar_think_weight > 0 else 0
    en_think_target = max(1, int(round(total_target * en_think_weight))) if en_think_weight > 0 else 0
    english_target = max(1, int(round(total_target * english_weight))) if english_weight > 0 else 0
    arcee_target = max(
        1,
        total_target - saudi_target - ar_think_target - en_think_target - english_target,
    )

    if args.dry_run and not args.live_stream:
        pools = build_synthetic_dpo_dry_run_pools()
        saudi_samples = pools["saudi"][: max(1, saudi_target)] if saudi_target > 0 else []
        arcee_samples = pools["arcee"][: max(1, arcee_target)]
        ar_think_samples = pools["arabic_thinking"][: max(1, ar_think_target)]
        english_samples = pools["english"][: max(1, english_target)]
        en_think_samples = pools["english_thinking"][: max(1, en_think_target)]
        decontam_summary: Dict[str, int] = {}
        sft_dedup_count = 0
    else:
        tokenizer = _load_optional_gemma4_tokenizer(PRIMARY_MODEL_ID)
        decontaminator = BenchmarkDecontaminator.from_eval_benchmarks(
            load_live=not (args.skip_decontamination or args.dry_run)
        )
        decontam_summary = dict(decontaminator.loaded_benchmarks)
        seen_prompts: Set[bytes] = (
            _collect_sft_prompt_hashes(pathlib.Path(args.sft_dataset_dir))
            if not args.dry_run
            else set()
        )
        sft_dedup_count = len(seen_prompts)

        saudi_samples = []
        if saudi_target > 0:
            logger.info(
                "Streaming %d samples from DPO source 'arabic_dialect_dpo_saudi'...",
                saudi_target,
            )
            saudi_samples = load_and_normalize_dpo_source(
                source_key="arabic_dialect_dpo_saudi",
                max_samples=saudi_target,
                streaming=True,
                tokenizer=tokenizer,
                decontaminator=decontaminator,
                seen_prompts=seen_prompts,
            )

        ar_think_samples = []
        if ar_think_target > 0:
            logger.info(
                "Streaming %d samples from DPO source 'arabic_reasoning_thinking_dpo'...",
                ar_think_target,
            )
            ar_think_samples = load_and_normalize_dpo_source(
                source_key="arabic_reasoning_thinking_dpo",
                max_samples=ar_think_target,
                streaming=True,
                tokenizer=tokenizer,
                decontaminator=decontaminator,
                seen_prompts=seen_prompts,
            )

        arabic_target_sum = saudi_target + arcee_target + ar_think_target
        effective_arcee_target = (
            max(arcee_target, arabic_target_sum - len(saudi_samples) - len(ar_think_samples))
            if not args.dry_run
            else arcee_target
        )
        logger.info(
            "Streaming %d samples from DPO source 'arabic_orpo_dpo_mix_40k'...",
            effective_arcee_target,
        )
        arcee_samples = load_and_normalize_dpo_source(
            source_key="arabic_orpo_dpo_mix_40k",
            max_samples=effective_arcee_target,
            streaming=True,
            tokenizer=tokenizer,
            decontaminator=decontaminator,
            seen_prompts=seen_prompts,
        )

        en_think_samples = []
        if en_think_target > 0:
            logger.info(
                "Streaming %d samples from DPO source 'distilabel_math_thinking_dpo'...",
                en_think_target,
            )
            en_think_samples = load_and_normalize_dpo_source(
                source_key="distilabel_math_thinking_dpo",
                max_samples=en_think_target,
                streaming=True,
                tokenizer=tokenizer,
                decontaminator=decontaminator,
                seen_prompts=seen_prompts,
            )

        english_samples = []
        if english_target > 0:
            collected_so_far = (
                len(saudi_samples)
                + len(ar_think_samples)
                + len(arcee_samples)
                + len(en_think_samples)
            )
            effective_eng_target = (
                max(english_target, total_target - collected_so_far)
                if not args.dry_run
                else english_target
            )
            logger.info(
                "Streaming %d samples from DPO source 'ultrafeedback_english_dpo'...",
                effective_eng_target,
            )
            english_samples = load_and_normalize_dpo_source(
                source_key="ultrafeedback_english_dpo",
                max_samples=effective_eng_target,
                streaming=True,
                tokenizer=tokenizer,
                decontaminator=decontaminator,
                seen_prompts=seen_prompts,
            )

    mixed_pairs, mix_stats = mix_dpo_preference_datasets(
        saudi_dpo_samples=saudi_samples,
        arcee_dpo_samples=arcee_samples,
        english_dpo_samples=english_samples,
        arabic_thinking_samples=ar_think_samples,
        english_thinking_samples=en_think_samples,
        saudi_weight=saudi_weight,
        arcee_weight=arcee_weight,
        english_weight=english_weight,
        arabic_thinking_weight=ar_think_weight,
        english_thinking_weight=en_think_weight,
        seed=args.seed,
    )
    validation_stats = validate_dpo_pairs(mixed_pairs)

    summary = {
        "mode": "dry_run" if args.dry_run else "production",
        "dpo_sources_configured": list(DPO_DATASETS.keys()),
        "saudi_config_verified": validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "saudi"),
        "sft_prompt_hashes_excluded": sft_dedup_count,
        "decontaminated_benchmarks": decontam_summary,
        "mix_stats": mix_stats,
        "validation_stats": validation_stats,
        "output_dir": str(output_dir),
    }

    if not args.dry_run:
        from datasets import Dataset

        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
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
