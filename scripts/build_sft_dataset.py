"""CLI script to build, decontaminate, validate, or pull the Stage 1 SFT dataset (120,000 rows).

Supports:
- All 9 curated public (`gated=False`) Arabic & Saudi SFT sources (90,000 rows / 75.0%):
  1. `AdaMLLab/smolkalam-arabic-conversational-sft` (`Mixture_of_Thoughts_science_no_think`)
  2. `AdaMLLab/smolkalam-arabic-conversational-sft` (`OpenThoughts3_50K` — verified `<think>...</think>`)
  3. `AdaMLLab/smolkalam-arabic-conversational-sft` (`OpenHermes_2.5_no_think`)
  4. `oddadmix/arabic-math-reasoning-synth`
  5. `FreedomIntelligence/alpaca-gpt4-arabic`
  6. `Gheras/Medical_Reasoning_Arabic_Dataset`
  7. `CohereLabs/aya_dataset` (human-annotated Arabic)
  8. `VohoAI/voho-saudi-dialogues`
  9. `SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus` (`moj` & `boe`, active statutes)
- All 6 Replay Buffer anchors (30,000 rows / 25.0%):
  `nvidia/OpenCodeReasoning`, `open-r1/codeforces-cots`, `m-a-p/CodeFeedback-Filtered-Instruction`,
  `nvidia/OpenMathInstruct-2` (full CoT + `\\boxed{}`), `nvidia/OpenScienceReasoning-2`, and
  `HuggingFaceH4/ultrachat_200k`.
- 13-gram benchmark decontamination against all Arabic & English evaluation test sets.
- `--pull-from-hub` fast path to download the pre-built verified corpus from
  `noman-asif/Arabic-Reasoning-Instruct-120k` in seconds on rented GPU instances.
"""

import argparse
import hashlib
import json
import logging
import pathlib
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_template import verify_thinking_block_preserved
from src.constants import (
    PRIMARY_MODEL_ID,
    PUBLISHED_SFT_DATASET_REPO,
    SFT_ARABIC_DATASETS,
    SFT_REPLAY_DATASETS,
    load_dotenv_if_present,
)
from src.data_normalizer import (
    BenchmarkDecontaminator,
    drop_pil_image_columns,
    load_and_normalize_arabic_sft_source,
    normalize_alpaca_gpt4_arabic_row,
    normalize_arabic_math_synth_row,
    normalize_aya_dataset_arabic_row,
    normalize_medical_reasoning_arabic_row,
    normalize_saudi_laws_row,
    normalize_smolkalam_row,
    normalize_voho_saudi_row,
)
from src.replay_mixer import (
    compute_mix_allocation,
    load_and_normalize_replay_source,
    mix_arabic_and_replay_samples,
    normalize_codefeedback_row,
    normalize_codeforces_cot_row,
    normalize_opencode_reasoning_row,
    normalize_openmath_instruct_row,
    normalize_openscience_reasoning_row,
    normalize_ultrachat_row,
)

logger = logging.getLogger(__name__)


def _load_optional_gemma4_tokenizer(model_id: str = PRIMARY_MODEL_ID) -> Optional[Any]:
    """Load the Gemma 4 tokenizer for exact token-length filtering when available."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        logger.info("Loaded Gemma 4 tokenizer '%s' for exact token-length filtering.", model_id)
        return tok
    except Exception as exc:
        logger.warning(
            "Could not load AutoTokenizer '%s' (%s); using conservative character-based token estimate.",
            model_id,
            exc,
        )
        return None


def build_synthetic_dry_run_pools() -> Dict[str, List[Dict[str, Any]]]:
    """Generate deterministic schema-accurate samples covering all 9 Arabic + 6 Replay sources."""

    class _MockMathDataset:
        column_names = ["image", "instruction", "reasoning", "answer"]

        def __init__(self, rows: List[Dict[str, Any]]) -> None:
            self.rows = rows

        def remove_columns(self, cols: List[str]) -> "_MockMathDataset":
            cleaned = [{k: v for k, v in r.items() if k not in cols} for r in self.rows]
            return _MockMathDataset(cleaned)

        def __iter__(self):
            return iter(self.rows)

    smol_sci = normalize_smolkalam_row(
        {
            "LR": 0.99,
            "SCR": 1.0,
            "messages": [
                {"role": "user", "content": "اشرح قانون نيوتن الثاني للحركة مع ذكر العلاقة الرياضية."},
                {
                    "role": "assistant",
                    "content": "ينص قانون نيوتن الثاني على أن القوة المحصلة المؤثرة في جسم تساوي حاصل ضرب كتلته في تسارعه، أي القوة = الكتلة × التسارع.",
                },
            ],
        },
        source_key="smolkalam_arabic_science",
        require_thinking=False,
    )

    smol_cot = normalize_smolkalam_row(
        {
            "LR": 0.98,
            "SCR": 0.98,
            "messages": [
                {"role": "user", "content": "إذا كان لدينا متتالية حسابية حدها الأول 3 وأساسها 4، فما هو الحد العاشر؟"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\nالحد النوني في المتتالية الحسابية يُعطى بالعلاقة: ح_ن = أ + (ن - 1) × د.\n"
                        "هنا أ = 3، د = 4، ن = 10.\nإذن ح_10 = 3 + 9 × 4 = 3 + 36 = 39.\n</think>\n"
                        "الحد العاشر في هذه المتتالية الحسابية يساوي 39."
                    ),
                },
            ],
        },
        source_key="smolkalam_arabic_reasoning_cot",
        require_thinking=True,
    )

    smol_oh = normalize_smolkalam_row(
        {
            "LR": 0.99,
            "SCR": 1.0,
            "messages": [
                {"role": "user", "content": "لخص أهم فوائد الطاقة الشمسية في المملكة العربية السعودية."},
                {
                    "role": "assistant",
                    "content": "تُسهم الطاقة الشمسية في تنويع مصادر الطاقة وتقليل الانبعاثات الكربونية والاستفادة من الإشعاع الشمسي المرتفع طوال العام.",
                },
            ],
        },
        source_key="smolkalam_arabic_openhermes",
        require_thinking=False,
    )

    raw_math_ds = _MockMathDataset(
        [
            {
                "image": "MOCK_PIL_IMAGE_OBJECT",
                "instruction": "إذا كان محيط مربع يساوي 36 سم، فما مساحته؟",
                "reasoning": "طول ضلع المربع = المحيط ÷ 4 = 36 ÷ 4 = 9 سم. مساحة المربع = طول الضلع × نفسه = 9 × 9 = 81 سم².",
                "answer": "مساحة المربع تساوي 81 سنتيمتراً مربعاً.",
            }
        ]
    )
    cleaned_math_ds = drop_pil_image_columns(raw_math_ds, ["image"])
    math_rows = list(cleaned_math_ds)
    assert "image" not in math_rows[0]
    ar_math = normalize_arabic_math_synth_row(math_rows[0])

    alpaca_ar = normalize_alpaca_gpt4_arabic_row(
        {
            "conversations": [
                {"from": "human", "value": "ما الفرق بين المبتدأ والخبر في النحو العربي؟"},
                {
                    "from": "gpt",
                    "value": "المبتدأ هو الاسم المرفوع المتحدث عنه في أول الجملة الاسمية، والخبر هو الجزء المتمم للفائدة معه.",
                },
            ]
        }
    )

    med_ar = normalize_medical_reasoning_arabic_row(
        {
            "Question": "ما هي الأعراض الشائعة لفقر الدم الناتج عن نقص الحديد وكيف يتم تشخيصه مخبرياً؟",
            "Complex_CoT": "نقص الحديد يؤدي إلى انخفاض تصنيع الهيموغلوبين في كريات الدم الحمراء مما يقلل نقل الأكسجين للأنسجة ويظهر مخبرياً بانخفاض الفيريتين.",
            "Response": "تشمل الأعراض الإرهاق الشديد وشحوب الجلد وضيق التنفس، ويُشخّص عبر فحص صورة الدم الكاملة ومستوى الفيريتين في الدم.",
        }
    )

    aya_ar = normalize_aya_dataset_arabic_row(
        {
            "language_code": "arb",
            "inputs": "اذكر ثلاثة من أبرز معالم مدينة العلا التاريخية في السعودية.",
            "targets": "من أبرز معالم العلا: مدائن صالح (الحِجر)، وجبل الفيل، والبلدة القديمة في العلا.",
        }
    )

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
            "text": "ينتهي عقد العمل في الأحوال المنصوص عليها في النظام باتفاق الطرفين كتابة بشرط أن تكون موافقة العامل كتابية.",
        },
        config_name="moj",
    )

    boe_sample = normalize_saudi_laws_row(
        {
            "law": {"title_ar": "نظام الشركات السعودي", "status": "ساري"},
            "article": {
                "article_title_norm": "المادة الثانية",
                "text": "الشركة كيان قانوني يؤسس وفقاً لأحكام النظام بناء على عقد تأسيس أو نظام أساس يلتزم بمقتضاه شخصان أو أكثر.",
            },
        },
        config_name="boe",
    )

    ocr_sample = normalize_opencode_reasoning_row(
        {
            "input": "Write a Python function `two_sum(nums, target)` returning indices of two numbers that add up to `target`.",
            "output": (
                "<think>\nWe can iterate once through `nums` while maintaining a hash map of seen values to indices for O(n) time.\n</think>\n"
                "```python\ndef two_sum(nums: list[int], target: int) -> list[int]:\n"
                "    seen = {}\n"
                "    for i, x in enumerate(nums):\n"
                "        if target - x in seen:\n"
                "            return [seen[target - x], i]\n"
                "        seen[x] = i\n"
                "    return []\n```"
            ),
        }
    )

    cf_sample = normalize_codeforces_cot_row(
        {
            "finish_reason": "stop",
            "prompt": "Given an integer n (1 <= n <= 10^9), determine whether n is divisible by both 2 and 3.",
            "generation": (
                "<think>\nAn integer is divisible by both 2 and 3 if and only if it is divisible by lcm(2, 3) = 6.\n</think>\n"
                "```python\nn = int(input().strip())\nprint('YES' if n % 6 == 0 else 'NO')\n```"
            ),
        }
    )

    codefb_sample = normalize_codefeedback_row(
        {
            "lang": "python",
            "query": "Implement binary search on a sorted Python list and return the target index or -1.",
            "answer": "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        }
    )

    openmath_sample = normalize_openmath_instruct_row(
        {
            "problem": "Find the derivative of f(x) = 4x^3 - 6x + 2 at x = 2.",
            "generated_solution": "Differentiating term by term gives f'(x) = 12x^2 - 6. Evaluating at x = 2 gives 12(4) - 6 = 42.",
            "expected_answer": "42",
        }
    )

    osr_sample = normalize_openscience_reasoning_row(
        {
            "input": "What is the pH of a neutral aqueous solution at 25 degrees Celsius?",
            "output": (
                "<think>\nAt 25 C, Kw = 1.0e-14, so [H+] = 1.0e-7 M and pH = -log10(1.0e-7) = 7.0.\n</think>\n"
                "The pH of a neutral aqueous solution at 25 °C is \\boxed{7.0}."
            ),
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

    base_arabic = [
        s
        for s in [
            smol_sci,
            smol_cot,
            smol_oh,
            ar_math,
            alpaca_ar,
            med_ar,
            aya_ar,
            voho_sample,
            moj_sample,
            boe_sample,
        ]
        if s is not None
    ]
    # Pad to 18 Arabic so 18 Arabic + 6 Replay = 24 (exact 75% / 25%)
    while len(base_arabic) < 18:
        base_arabic.extend(base_arabic[: 18 - len(base_arabic)])

    replay_pool = [
        s
        for s in [
            ocr_sample,
            cf_sample,
            codefb_sample,
            openmath_sample,
            osr_sample,
            ultrachat_sample,
        ]
        if s is not None
    ]
    return {"arabic": base_arabic[:18], "replay": replay_pool}


def validate_sft_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify every normalized SFT sample satisfies Gemma 4 native token & message schema invariants."""
    thinking_true_count = 0
    thinking_false_count = 0

    for idx, sample in enumerate(samples):
        text = str(sample.get("text") or "")
        enable_thinking = bool(sample.get("enable_thinking", False))
        msgs = sample.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError(f"Sample {idx} missing valid `messages` list.")
        for m in msgs:
            if set(m.keys()) != {"role", "content", "reasoning_content"}:
                raise ValueError(
                    f"Sample {idx} message keys {set(m.keys())} do not match uniform Arrow schema."
                )
        if not verify_thinking_block_preserved(text, enable_thinking=enable_thinking):
            raise ValueError(
                f"Sample at index {idx} (source={sample.get('source')!r}) failed "
                f"Gemma 4 native chat/thinking verification (enable_thinking={enable_thinking})."
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
    """Parse CLI arguments for building or pulling the Stage 1 SFT dataset."""
    parser = argparse.ArgumentParser(
        description="Build, decontaminate, validate, or pull the Stage 1 SFT dataset (75% Arabic / 25% Replay)."
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
        "--total-samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Maximum total samples to build across Arabic and Replay sources.",
    )
    parser.add_argument(
        "--pull-from-hub",
        action="store_true",
        help=f"Pull pre-built verified dataset directly from `{PUBLISHED_SFT_DATASET_REPO}`.",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=PUBLISHED_SFT_DATASET_REPO,
        help="Hugging Face dataset repository ID when using --pull-from-hub.",
    )
    parser.add_argument(
        "--skip-decontamination",
        action="store_true",
        help="Skip live benchmark test-set loading for 13-gram decontamination (not recommended for production).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild or re-download even if output-dir already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast validation of all 15 SFT normalizers and 75/25 stratified mixer.",
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
    """Entry point for Stage 1 SFT dataset construction, decontamination, and verification."""
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

    arabic_ratio = float(ds_cfg.get("arabic_mix_ratio", 0.75))
    replay_ratio = float(ds_cfg.get("replay_mix_ratio", 0.25))
    total_target = int(
        args.max_samples
        if args.max_samples is not None
        else (24 if args.dry_run else ds_cfg.get("total_target_samples", 30000))
    )
    output_dir = pathlib.Path(
        args.output_dir or ds_cfg.get("output_dataset_dir", "data/sft_mixed")
    )

    if not args.dry_run and not args.force and (output_dir / "dataset_info.json").exists():
        logger.info("Stage 1 SFT dataset already exists at %s; skipping (pass --force to rebuild).", output_dir)
        return 0

    if args.pull_from_hub and not args.dry_run:
        from datasets import load_dataset

        logger.info("Pulling pre-built verified SFT dataset from Hugging Face Hub: %s ...", args.hub_repo)
        hf_ds = load_dataset(args.hub_repo, split="train")
        if args.max_samples is not None and args.max_samples < len(hf_ds):
            hf_ds = hf_ds.select(range(args.max_samples))
        rows_for_check = [hf_ds[i] for i in range(len(hf_ds))]
        validation_stats = validate_sft_samples(rows_for_check)
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        hf_ds.save_to_disk(str(output_dir))
        summary = {
            "mode": "pulled_from_hub",
            "hub_repo": args.hub_repo,
            "total_rows": len(hf_ds),
            "validation_stats": validation_stats,
            "output_dir": str(output_dir),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d verified SFT rows from '%s' to %s", len(hf_ds), args.hub_repo, output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    allocation = compute_mix_allocation(
        total_samples=total_target,
        arabic_ratio=arabic_ratio,
        replay_ratio=replay_ratio,
    )

    if args.dry_run and not args.live_stream:
        pools = build_synthetic_dry_run_pools()
        arabic_samples = pools["arabic"]
        replay_samples = pools["replay"]
        decontam_summary: Dict[str, int] = {}
    else:
        tokenizer = _load_optional_gemma4_tokenizer(PRIMARY_MODEL_ID)
        decontaminator = BenchmarkDecontaminator.from_eval_benchmarks(
            load_live=not (args.skip_decontamination or args.dry_run)
        )
        decontam_summary = dict(decontaminator.loaded_benchmarks)
        seen_prompts: Set[bytes] = set()
        step_cache_dir: Optional[pathlib.Path] = (
            (output_dir.parent / ".sft_build_cache") if not args.dry_run else None
        )
        if step_cache_dir is not None:
            step_cache_dir.mkdir(parents=True, exist_ok=True)

        def _fetch_with_retry_and_cache(
            step_id: str,
            expected_max: int,
            fetch_fn: Callable[[], List[Dict[str, Any]]],
        ) -> List[Dict[str, Any]]:
            cache_file = (
                step_cache_dir / f"{step_id}_{expected_max}.jsonl"
                if step_cache_dir is not None
                else None
            )
            if cache_file is not None and cache_file.exists() and cache_file.stat().st_size > 0:
                cached_rows: List[Dict[str, Any]] = []
                with open(cache_file, "r", encoding="utf-8") as cf:
                    for raw_line in cf:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        item = json.loads(raw_line)
                        first_user = next(
                            (m["content"] for m in item.get("messages", []) if m.get("role") == "user"),
                            "",
                        )
                        sig = hashlib.blake2b(
                            first_user.strip().lower().encode("utf-8", errors="ignore"),
                            digest_size=12,
                        ).digest()
                        if sig in seen_prompts:
                            continue
                        seen_prompts.add(sig)
                        cached_rows.append(item)
                logger.info(
                    "Loaded %d cached SFT rows for step '%s' from %s",
                    len(cached_rows),
                    step_id,
                    cache_file.name,
                )
                return cached_rows

            snapshot = set(seen_prompts)
            last_exc: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    rows = fetch_fn()
                    if cache_file is not None:
                        tmp_cache = cache_file.with_suffix(".jsonl.tmp")
                        with open(tmp_cache, "w", encoding="utf-8") as tf:
                            for r in rows:
                                tf.write(json.dumps(r, ensure_ascii=False) + "\n")
                        tmp_cache.replace(cache_file)
                    return rows
                except Exception as exc:
                    last_exc = exc
                    seen_prompts.clear()
                    seen_prompts.update(snapshot)
                    logger.warning(
                        "Step '%s' attempt %d/3 encountered transient error (%s); retrying...",
                        step_id,
                        attempt,
                        exc,
                    )
                    if attempt < 3:
                        time.sleep(2.5 * attempt)
            assert last_exc is not None
            raise last_exc

        arabic_samples: List[Dict[str, Any]] = []
        arabic_target_total = int(allocation["arabic_total"])
        for src_key, count in allocation["arabic_per_source"].items():
            if count <= 0:
                continue
            logger.info("Streaming %d samples from Arabic SFT source '%s'...", count, src_key)
            arabic_samples.extend(
                _fetch_with_retry_and_cache(
                    step_id=f"arabic_{src_key}",
                    expected_max=count,
                    fetch_fn=lambda sk=src_key, cnt=count: load_and_normalize_arabic_sft_source(
                        source_key=sk,
                        max_samples=cnt,
                        streaming=True,
                        tokenizer=tokenizer,
                        decontaminator=decontaminator,
                        seen_prompts=seen_prompts,
                    ),
                )
            )

        # Top up any strict-hygiene / deduplication deficit from high-purity Arabic pools
        # (thinking-first so any Arabic CoT shortfall is replenished by CoT rows, never noisy synth)
        for topup_key in (
            "smolkalam_arabic_reasoning_cot",
            "smolkalam_arabic_science",
            "smolkalam_arabic_openhermes",
            "alpaca_gpt4_arabic",
        ):
            deficit = arabic_target_total - len(arabic_samples)
            if deficit <= 0:
                break
            logger.info(
                "Topping up %d Arabic SFT deficit rows from '%s' (dedup enforced)...",
                deficit,
                topup_key,
            )
            arabic_samples.extend(
                _fetch_with_retry_and_cache(
                    step_id=f"arabic_topup_{topup_key}",
                    expected_max=deficit,
                    fetch_fn=lambda tk=topup_key, df=deficit: load_and_normalize_arabic_sft_source(
                        source_key=tk,
                        max_samples=df,
                        streaming=True,
                        tokenizer=tokenizer,
                        decontaminator=decontaminator,
                        seen_prompts=seen_prompts,
                    ),
                )
            )

        replay_samples: List[Dict[str, Any]] = []
        replay_target_total = int(allocation["replay_total"])
        for src_key, count in allocation["replay_per_source"].items():
            if count <= 0:
                continue
            logger.info("Streaming %d samples from Replay source '%s'...", count, src_key)
            replay_samples.extend(
                _fetch_with_retry_and_cache(
                    step_id=f"replay_{src_key}",
                    expected_max=count,
                    fetch_fn=lambda sk=src_key, cnt=count: load_and_normalize_replay_source(
                        source_key=sk,
                        max_samples=cnt,
                        streaming=True,
                        tokenizer=tokenizer,
                        decontaminator=decontaminator,
                        seen_prompts=seen_prompts,
                    ),
                )
            )

        for rep_topup_key in (
            "codefeedback_instruction",
            "openmath_instruct2_cot",
            "ultrachat_english_replay",
        ):
            rep_deficit = replay_target_total - len(replay_samples)
            if rep_deficit <= 0:
                break
            logger.info(
                "Topping up %d Replay SFT deficit rows from '%s' (dedup enforced)...",
                rep_deficit,
                rep_topup_key,
            )
            replay_samples.extend(
                _fetch_with_retry_and_cache(
                    step_id=f"replay_topup_{rep_topup_key}",
                    expected_max=rep_deficit,
                    fetch_fn=lambda rtk=rep_topup_key, rdf=rep_deficit: load_and_normalize_replay_source(
                        source_key=rtk,
                        max_samples=rdf,
                        streaming=True,
                        tokenizer=tokenizer,
                        decontaminator=decontaminator,
                        seen_prompts=seen_prompts,
                    ),
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

    per_source_counts: Dict[str, int] = {}
    for s in mixed_samples:
        src = str(s.get("source", "unknown"))
        per_source_counts[src] = per_source_counts.get(src, 0) + 1

    summary = {
        "mode": "dry_run" if args.dry_run else "production",
        "arabic_sources_configured": list(SFT_ARABIC_DATASETS.keys()),
        "replay_sources_configured": list(SFT_REPLAY_DATASETS.keys()),
        "decontaminated_benchmarks": decontam_summary,
        "allocation": allocation,
        "per_source_counts": per_source_counts,
        "mix_stats": mix_stats,
        "validation_stats": validation_stats,
        "output_dir": str(output_dir),
    }

    if not args.dry_run:
        from datasets import Dataset

        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        hf_dataset = Dataset.from_list(mixed_samples)
        hf_dataset.save_to_disk(str(output_dir))
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        shutil.rmtree(output_dir.parent / ".sft_build_cache", ignore_errors=True)
        logger.info("Saved %d mixed SFT examples to %s", len(mixed_samples), output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
