"""End-to-end automated unit and smoke test suite for Gemma-4-Arabic.

Verifies:
1. All 9 dataset schema normalizers and mandatory preprocessing rules.
2. Gemma 4 `apply_chat_template` formatting (`enable_thinking=False` and `enable_thinking=True`)
   and preservation of the 4 hidden empty thinking-block tokens (`<|think|>\\n<|think|>\\n`).
3. Strict 75% Arabic / 25% Replay stratified buffer allocation and mixing.
4. Dynamic `named_modules()` discovery, CPU forward hook for `embed_tokens_per_layer`,
   and CPU offload/restoration for `audio_tower` and `vision_tower`.
5. Stage 2 Conversational DPO (`ref_model=None` / optional Trust-Region EMA `tr_alpha=0.6`).
6. Lossless 16-Bit Merging (`lambda=1.0`, `density=1.0`) and optional TIES sign election/trimming math.
7. Zero-forgetting retention gate evaluation and YAML configuration integrity.
"""

import json
import pathlib
import sys
import tempfile
import unittest
from typing import Any, Dict, List

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_latency_and_plots import (
    build_reference_latency_profile,
    generate_all_family_plots,
)
from scripts.benchmark_vllm_concurrency import (
    _compute_active_kv_and_runtime_gb,
    build_reference_vllm_concurrency_profile,
    generate_vllm_concurrency_plot,
    render_vllm_markdown_report,
)
from scripts.run_benchmarks import (
    ACVA_ALL_58_SUBSETS,
    ARABIC_MMLU_40_SUBSETS,
    _atomic_write_json,
    _build_child_eval_cmd,
    _build_isolated_gpu_env,
    _configure_lm_eval_thinking_for_task,
    _ensure_gemma4_tied_lm_head,
    _extract_arc_mcq_letter,
    _extract_humaneval_code,
    _extract_mcq_letter_with_doc,
    _extract_mgsm_ar_target_number,
    _get_lm_eval_gen_kwargs,
    _load_stage_checkpoint,
    _normalize_arabic_digits,
    _normalize_arabic_digits_filter,
    _patch_gemma4_config_for_lm_eval,
    _patch_local_checkpoint_config_json,
    _resolve_selected_suites,
    _should_apply_chat_template_for_task,
    _strip_thinking_blocks,
    _truncate_math_continuation,
    build_custom_lm_eval_task_dict,
    clean_gemma4_completion,
    combine_role_checkpoints,
    compute_effective_task_limit,
    evaluate_retention_gate,
    extract_lm_eval_detailed_metrics,
    generate_family_markdown_report,
    generate_markdown_report,
    normalize_inspect_completion_for_scoring,
    normalize_lcb_code_completion,
    parse_args as parse_bench_args,
    parse_lighteval_results_json,
    parse_lm_eval_results_json,
    resolve_lighteval_arabic_tasks,
    resolve_lm_eval_task_specs,
)
from src.chat_template import (
    build_tokenized_sft_Turn_masks,
    format_gemma4_chat,
    format_thinking_response,
    verify_thinking_block_preserved,
)
from src.constants import (
    ALL_9_DATASET_REPOS,
    ALL_DATASET_REPOS,
    ARABIC_DIALECT_DPO_REPO,
    CHANNEL_END_TOKEN,
    CHANNEL_THOUGHT_PREFIX,
    DEFAULT_TIES_DENSITY,
    DEFAULT_TIES_LAMBDA,
    DPO_TRUST_REGION_ALPHA,
    EMPTY_THINKING_BLOCK,
    LORA_ALPHA,
    LORA_ALPHA_26B,
    LORA_RANK,
    LORA_RANK_26B,
    LORA_TARGET_MODULES,
    PRIMARY_MODEL_ID_26B,
    PUBLISHED_DPO_DATASET_REPO,
    PUBLISHED_SFT_DATASET_REPO,
    TURN_END_TOKEN,
    TURN_START_TOKEN,
)
from src.data_normalizer import (
    BenchmarkDecontaminator,
    drop_pil_image_columns,
    normalize_acegpt_v2_row,
    normalize_alpaca_gpt4_arabic_row,
    normalize_arabic_math_row,
    normalize_arabic_math_synth_row,
    normalize_aya_arabic_row,
    normalize_aya_dataset_arabic_row,
    normalize_medical_reasoning_arabic_row,
    normalize_saudi_laws_row,
    normalize_smolkalam_row,
    normalize_voho_saudi_row,
)
from src.dpo_formatter import (
    normalize_arabic_dialect_dpo_row,
    normalize_arcee_arabic_dpo_row,
    normalize_ultrafeedback_dpo_row,
    validate_dpo_load_config,
)
from src.merge_and_export import (
    _patch_exported_gemma4_config_tie_embeddings,
    apply_ties_interpolation_to_state_dict,
    ties_merge_deltas,
    ties_trim_tensor,
)
from src.model_loader import (
    atomic_write_json,
    build_8gb_device_map,
    ensure_tied_lm_head,
    find_latest_valid_checkpoint,
    freeze_embeddings_and_towers,
    is_adapter_stage_complete,
)
from src.replay_mixer import (
    compute_mix_allocation,
    mix_arabic_and_replay_samples,
    normalize_codefeedback_row,
    normalize_codeforces_cot_row,
    normalize_opencode_reasoning_row,
    normalize_openmath_instruct_row,
    normalize_openscience_reasoning_row,
    normalize_ultrachat_row,
)
from src.train_dpo import (
    align_26b_chat_template_for_dpo,
    install_dpo_default_adapter_resume_hook,
    update_trust_region_reference_params,
    verify_dpo_prefix_alignment,
)
from src.vram_hooks import (
    apply_8gb_vram_optimizations,
    get_vram_telemetry,
    restore_multimodal_towers,
)


class TestGemma4ArabicPipeline(unittest.TestCase):
    """Comprehensive unit and smoke test suite for the Gemma-4-Arabic repository."""

    def test_01_constants_and_yaml_configs(self) -> None:
        """Verify all V2 dataset repos, LoRA ranks, and YAML config invariants."""
        self.assertGreaterEqual(len(ALL_DATASET_REPOS), 15)
        self.assertGreaterEqual(len(ALL_9_DATASET_REPOS), 9)
        self.assertEqual(LORA_RANK, 32)
        self.assertEqual(LORA_ALPHA, 32)
        self.assertEqual(LORA_RANK_26B, 16)
        self.assertEqual(LORA_ALPHA_26B, 16)
        self.assertEqual(len(LORA_TARGET_MODULES), 7)
        self.assertEqual(DEFAULT_TIES_LAMBDA, 1.0)
        self.assertEqual(DEFAULT_TIES_DENSITY, 1.0)
        self.assertEqual(PUBLISHED_SFT_DATASET_REPO, "noman-asif/Arabic-Reasoning-Instruct-30k")
        self.assertEqual(PUBLISHED_DPO_DATASET_REPO, "noman-asif/Arabic-Preference-DPO-6k")

        sft_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "sft_config.yaml").read_text(encoding="utf-8")
        )
        sft_e4b_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "sft_e4b_a100.yaml").read_text(encoding="utf-8")
        )
        sft_26b_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "sft_26b_a4b_a100.yaml").read_text(encoding="utf-8")
        )
        dpo_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "dpo_config.yaml").read_text(encoding="utf-8")
        )
        dpo_e4b_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "dpo_e4b_a100.yaml").read_text(encoding="utf-8")
        )
        dpo_26b_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "dpo_26b_a4b_a100.yaml").read_text(encoding="utf-8")
        )
        eval_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval_config.yaml").read_text(encoding="utf-8")
        )

        for cfg in (sft_cfg, sft_e4b_cfg, sft_26b_cfg):
            self.assertEqual(cfg["datasets"]["hf_hub_dataset_repo"], PUBLISHED_SFT_DATASET_REPO)
        for cfg in (dpo_cfg, dpo_e4b_cfg, dpo_26b_cfg):
            self.assertEqual(cfg["datasets"]["hf_hub_dataset_repo"], PUBLISHED_DPO_DATASET_REPO)

        self.assertFalse(sft_cfg["lora"]["use_rslora"])
        self.assertEqual(sft_cfg["lora"]["r"], 32)
        self.assertEqual(sft_cfg["lora"]["lora_alpha"], 32)
        self.assertEqual(sft_cfg["datasets"]["arabic_mix_ratio"], 0.75)
        self.assertEqual(sft_cfg["datasets"]["replay_mix_ratio"], 0.25)

        self.assertEqual(dpo_cfg["dpo"]["beta"], 0.2)
        self.assertEqual(dpo_cfg["training"]["learning_rate"], 2.0e-6)
        self.assertTrue(dpo_cfg["vram_optimization"]["precompute_ref_log_probs"])
        self.assertTrue(dpo_e4b_cfg["vram_optimization"]["precompute_ref_log_probs"])
        self.assertTrue(dpo_26b_cfg["vram_optimization"]["precompute_ref_log_probs"])
        self.assertEqual(dpo_cfg["dpo"]["max_length"], 1536)
        self.assertEqual(dpo_e4b_cfg["dpo"]["max_length"], 3072)
        self.assertEqual(dpo_26b_cfg["dpo"]["max_length"], 3072)

        self.assertEqual(len(eval_cfg["arabic_benchmarks"]), 6)
        self.assertNotIn("MGSM-Ar", eval_cfg["arabic_benchmarks"])
        self.assertEqual(len(eval_cfg["retention_anchors"]), 4)

    def test_02_gemma4_chat_template_thinking_tokens(self) -> None:
        """Verify native Gemma 4 chat template tokens (<|turn>, <|channel>thought, <channel|>)."""
        msgs_direct = [
            {"role": "user", "content": "ما هي عاصمة المملكة العربية السعودية؟", "reasoning_content": ""},
            {"role": "assistant", "content": "الرياض هي العاصمة.", "reasoning_content": ""},
        ]
        rendered_false_e4b = format_gemma4_chat(msgs_direct, enable_thinking=False)
        self.assertIn(TURN_START_TOKEN, rendered_false_e4b)
        self.assertIn(TURN_END_TOKEN, rendered_false_e4b)
        self.assertNotIn("<start_of_turn>", rendered_false_e4b)
        self.assertNotIn("<end_of_turn>", rendered_false_e4b)
        self.assertNotIn(EMPTY_THINKING_BLOCK, rendered_false_e4b)
        self.assertTrue(verify_thinking_block_preserved(rendered_false_e4b, enable_thinking=False))

        rendered_false_26b = format_gemma4_chat(
            msgs_direct,
            enable_thinking=False,
            is_26b_moe=True,
            align_26b_empty_thought=True,
        )
        self.assertIn(EMPTY_THINKING_BLOCK, rendered_false_26b)

        msgs_thinking = [
            {"role": "user", "content": "احسب 9 * 7", "reasoning_content": ""},
            {
                "role": "assistant",
                "content": format_thinking_response("9 ضرب 7 يساوي 63", "الناتج هو 63."),
            },
        ]
        rendered_true = format_gemma4_chat(msgs_thinking, enable_thinking=True)
        self.assertIn(CHANNEL_THOUGHT_PREFIX, rendered_true)
        self.assertIn(CHANNEL_END_TOKEN, rendered_true)
        self.assertNotIn("<think>", rendered_true)
        self.assertNotIn("</think>", rendered_true)
        self.assertTrue(verify_thinking_block_preserved(rendered_true, enable_thinking=True))

    def test_03_all_arabic_sft_normalizers_and_decontaminator(self) -> None:
        """Verify schema normalizers for all 9 Stage 1 Arabic SFT datasets and 13-gram decontamination."""
        # 1. VohoAI/voho-saudi-dialogues (messages_json + speaker prefix stripping)
        voho = normalize_voho_saudi_row(
            {
                "messages_json": json.dumps(
                    [
                        {"role": "user", "content": "User: وشلون أطلع بدل فاقد للهوية الوطنية عبر منصة أبشر؟"},
                        {"role": "assistant", "content": "المساعد: ادخل على منصة أبشر ثم اختر خدماتي فالأحوال المدنية ثم الإبلاغ عن الوثائق المفقودة وحجز موعد."},
                    ]
                )
            }
        )
        self.assertIsNotNone(voho)
        assert voho is not None
        user_turns = [m for m in voho["messages"] if m["role"] == "user"]
        asst_turns = [m for m in voho["messages"] if m["role"] == "assistant"]
        self.assertFalse(user_turns[0]["content"].startswith("User:"))
        self.assertFalse(asst_turns[0]["content"].startswith("المساعد:"))
        self.assertIn(TURN_START_TOKEN, voho["text"])
        self.assertNotIn("<start_of_turn>", voho["text"])

        # 2. SaudiArabicLaws ("moj" flat and "boe" nested)
        moj = normalize_saudi_laws_row(
            {
                "doc_title": "نظام العمل السعودي",
                "article_title": "المادة الأولى",
                "text": "يسمى هذا النظام نظام العمل ويهدف إلى تنظيم العلاقة التعاقدية بين صاحب العمل والعامل في المملكة.",
            },
            config_name="moj",
        )
        boe = normalize_saudi_laws_row(
            {
                "law": {"title_ar": "نظام المعاملات المدنية"},
                "article": {"article_title_norm": "المادة الخامسة", "text": ""},
                "retrieval": {"contextual_text": "تطبق أحكام هذا النظام على جميع المسائل التي تناولتها نصوصه في لفظها أو في فحواها."},
            },
            config_name="boe",
        )
        self.assertIsNotNone(moj)
        self.assertIsNotNone(boe)
        assert moj is not None and boe is not None
        self.assertEqual(moj["source"], "saudi_arabic_laws_moj")
        self.assertEqual(boe["source"], "saudi_arabic_laws_boe")
        self.assertIn("تطبق أحكام هذا النظام", boe["text"])

        # 3. FreedomIntelligence/AceGPT-v2-tuning (conversations from/value)
        ace = normalize_acegpt_v2_row(
            {
                "conversations": [
                    {"from": "human", "value": "عرف علم البلاغة في اللغة العربية واذكر أقسامه الثلاثة الرئيسية."},
                    {"from": "gpt", "value": "البلاغة هي مطابقة الكلام لمقتضى الحال مع فصاحته، وتنقسم إلى ثلاثة علوم رئيسية هي علم المعاني وعلم البيان وعلم البديع."},
                ]
            }
        )
        self.assertIsNotNone(ace)
        assert ace is not None
        roles_ace = [m["role"] for m in ace["messages"]]
        self.assertIn("user", roles_ace)
        self.assertIn("assistant", roles_ace)

        # 4. Omartificial-Intelligence-Space/Arabic-Math-SFT (.remove_columns(["image"]) + thinking)
        class _DummyMathDataset:
            column_names = ["image", "problem", "solution"]

            def __init__(self) -> None:
                self.removed_cols: List[str] = []

            def remove_columns(self, cols: List[str]) -> "_DummyMathDataset":
                self.removed_cols = list(cols)
                return self

        dummy_ds = drop_pil_image_columns(_DummyMathDataset(), ["image"])
        self.assertEqual(dummy_ds.removed_cols, ["image"])

        math_item = normalize_arabic_math_row(
            {
                "image": "PIL_IMAGE_PLACEHOLDER",
                "problem": "حل المعادلة الخطية التالية بالتفصيل: 2س + 4 = 10",
                "solution": "الخطوة الأولى: نطرح 4 من الطرفين فنحصل على 2س = 6.\nالخطوة الثانية: نقسم الطرفين على 2.\nإذن قيمة س النهائية هي 3.",
            }
        )
        self.assertIsNotNone(math_item)
        assert math_item is not None
        self.assertTrue(math_item["enable_thinking"])
        self.assertIn(CHANNEL_THOUGHT_PREFIX, math_item["text"])
        self.assertIn(CHANNEL_END_TOKEN, math_item["text"])

        # 5. CohereForAI/aya_collection_language_split (inputs/targets + JSON tool-calling)
        aya = normalize_aya_arabic_row(
            {
                "inputs": "تحقق من رصيد الإجازات السنوية المتبقي للموظف رقم 102 في النظام",
                "targets": "رصيد الإجازات السنوية المتبقي للموظف رقم 102 هو 21 يوماً عمل.",
                "tools": [{"name": "get_leave_balance"}],
                "tool_call": {"name": "get_leave_balance", "arguments": {"emp_id": 102}},
            }
        )
        self.assertIsNotNone(aya)
        assert aya is not None
        self.assertIn("get_leave_balance", aya["text"])

        # 6-9. New V2 Arabic normalizers (Gheras Medical CoT, Smolkalam OpenThoughts3, Alpaca-GPT4-Arabic, Aya Arabic)
        med = normalize_medical_reasoning_arabic_row(
            {
                "Question": "مريض يعاني من عطش شديد وكثرة التبول وارتفاع سكر الدم الصائم إلى 190 ملغ/ديسيلتر، ما التشخيص المرجح؟",
                "Complex_CoT": "نلاحظ وجود الأعراض الكلاسيكية لفرط سكر الدم مع تجاوز مستوى الجلوكوز الصائم لعتبة 126 ملغ/ديسيلتر عند الفحص السريري.",
                "Response": "التشخيص المرجح هو داء السكري من النوع الثاني، ويلزم تأكيده بفحص السكر التراكمي HbA1c.",
            }
        )
        self.assertIsNotNone(med)
        assert med is not None
        self.assertTrue(med["enable_thinking"])
        self.assertTrue(med["messages"][-1]["reasoning_content"].startswith("نلاحظ"))

        smol = normalize_smolkalam_row(
            {
                "LR": 0.98,
                "SCR": 0.99,
                "messages": [
                    {"role": "user", "content": "احسب مساحة مستطيل طوله 12 سم وعرضه 5 سم بالتفصيل."},
                    {"role": "assistant", "content": "<think>\nلحساب مساحة المستطيل نضرب الطول في العرض مباشرة: المساحة تساوي 12 ضرب 5 أي 60 سنتيمتراً مربعاً.\n</think>\nمساحة المستطيل تساوي 60 سنتيمتراً مربعاً."},
                ],
            },
            source_key="smolkalam_arabic_reasoning_cot",
            require_thinking=True,
        )
        self.assertIsNotNone(smol)
        assert smol is not None
        self.assertTrue(smol["enable_thinking"])
        self.assertNotIn("<think>", smol["messages"][-1]["content"])
        self.assertIn("60", smol["messages"][-1]["reasoning_content"])

        synth_math = normalize_arabic_math_synth_row(
            {
                "instruction": "أوجد قيمة س إذا كان 3س + 9 = 24.",
                "reasoning": "نطرح 9 من الطرفين فنحصل على 3س = 15 ثم نقسم على 3 فنحصل على س = 5.",
                "answer": "قيمة س النهائية تساوي 5.",
            }
        )
        self.assertIsNotNone(synth_math)

        alpaca_ar = normalize_alpaca_gpt4_arabic_row(
            {
                "conversations": [
                    {"from": "human", "value": "اشرح الفرق الجوهري بين التشفير المتماثل والتشفير غير المتماثل في أمن المعلومات."},
                    {"from": "gpt", "value": "يعتمد التشفير المتماثل على مفتاح سري واحد للتشفير وفك التشفير، بينما يستخدم التشفير غير المتماثل زوجاً من المفاتيح هما المفتاح العام والمفتاح الخاص."},
                ]
            }
        )
        self.assertIsNotNone(alpaca_ar)

        aya_human = normalize_aya_dataset_arabic_row(
            {
                "language_code": "ars",
                "inputs": "وش أهم الفعاليات التراثية في مهرجان الجنادرية بالمملكة العربية السعودية؟",
                "targets": "يضم مهرجان الجنادرية العرضة السعودية، وأجنحة المناطق التراثية، والحرف اليدوية التقليدية، والأمسيات الشعرية.",
            }
        )
        self.assertIsNotNone(aya_human)

        # 13-gram Benchmark Decontaminator verification
        decontam = BenchmarkDecontaminator()
        eval_text = "في القانون التجاري السعودي ما هي الشروط النظامية الواجب توافرها لتأسيس شركة مساهمة مبسطة وفق النظام الجديد"
        decontam.add_benchmark_text(eval_text)
        self.assertTrue(decontam.is_contaminated(eval_text))
        self.assertFalse(decontam.is_contaminated("كيف أحسب زكاة عروض التجارة في نهاية الحول؟"))

    def test_04_replay_normalizers_and_75_25_mixer(self) -> None:
        """Verify all 6 Replay Buffer normalizers and strict 75/25 mixing."""
        ultra = normalize_ultrachat_row(
            {
                "messages": [
                    {"role": "user", "content": "Explain binary search and its worst-case time complexity."},
                    {"role": "assistant", "content": "Binary search repeatedly halves a sorted search interval, achieving O(log n) worst-case time complexity."},
                ]
            }
        )
        self.assertIsNotNone(ultra)
        assert ultra is not None
        self.assertFalse(ultra["enable_thinking"])

        # OpenMathInstruct-2 must preserve BOTH reasoning_content AND \boxed{expected_answer} in content
        openmath = normalize_openmath_instruct_row(
            {
                "problem": "Evaluate 7! / 5! step by step.",
                "generated_solution": "We expand 7! / 5! = (7 * 6 * 5!) / 5! = 7 * 6 = 42.",
                "expected_answer": "42",
            }
        )
        self.assertIsNotNone(openmath)
        assert openmath is not None
        self.assertTrue(openmath["enable_thinking"])
        self.assertIn("7 * 6 = 42", openmath["messages"][-1]["reasoning_content"])
        self.assertIn("\\boxed{42}", openmath["messages"][-1]["content"])

        ocr = normalize_opencode_reasoning_row(
            {
                "input": "Write a Python function `is_prime(n: int) -> bool` that checks primality in O(sqrt(n)).",
                "output": "<think>\nCheck n < 2 first, then test divisors up to isqrt(n).\n</think>\n```python\nimport math\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    return all(n % d != 0 for d in range(2, math.isqrt(n) + 1))\n```",
            }
        )
        self.assertIsNotNone(ocr)
        assert ocr is not None
        self.assertTrue(ocr["enable_thinking"])
        self.assertNotIn("<think>", ocr["messages"][-1]["content"])

        cf = normalize_codeforces_cot_row(
            {
                "finish_reason": "stop",
                "messages": [
                    {"role": "user", "content": "Given an array of n integers, output their sum in Python."},
                    {"role": "assistant", "content": "<think>\nRead n and sum the array using Python's built-in sum function.\n</think>\n```python\nimport sys\ndata = sys.stdin.read().split()\nprint(sum(map(int, data[1:])))\n```"},
                ],
            }
        )
        self.assertIsNotNone(cf)

        cfb = normalize_codefeedback_row(
            {
                "query": "Implement iterative factorial in Python for non-negative integers.",
                "answer": "Here is an iterative Python implementation:\n```python\ndef factorial(n: int) -> int:\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n```",
            }
        )
        self.assertIsNotNone(cfb)

        sci = normalize_openscience_reasoning_row(
            {
                "input": "What is the spin quantum number of an electron and how many spin states exist?",
                "output": "<think>\nAn electron is a spin-1/2 fermion with m_s = +1/2 or -1/2.\n</think>\nAn electron has spin quantum number s = 1/2, giving 2s + 1 = 2 spin states (+1/2 and -1/2).",
            }
        )
        self.assertIsNotNone(sci)
        assert sci is not None
        self.assertTrue(sci["enable_thinking"])

        alloc = compute_mix_allocation(total_samples=200)
        self.assertEqual(alloc["arabic_total"], 150)
        self.assertEqual(alloc["replay_total"], 50)

        ar_pool = [{"source": "voho_saudi_dialogues", "text": f"ar_{i}"} for i in range(12)]
        rep_pool = [ultra, openmath, ocr, sci]
        mixed, stats = mix_arabic_and_replay_samples(ar_pool, rep_pool)
        self.assertEqual(len(mixed), 16)
        self.assertAlmostEqual(stats["arabic_fraction"], 0.75, places=6)
        self.assertAlmostEqual(stats["replay_fraction"], 0.25, places=6)

    def test_05_dpo_normalizers_and_prompt_unification(self) -> None:
        """Verify all 3 Stage 2 conversational DPO normalizers and prompt hygiene."""
        self.assertEqual(validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "saudi"), "saudi")
        with self.assertRaises(ValueError):
            validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "default")
        with self.assertRaises(ValueError):
            validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, None)

        saudi_dpo = normalize_arabic_dialect_dpo_row(
            {
                "prompt": "كيف أقدم على وظيفة حكومية أو في القطاع الخاص عبر منصة جدارات؟",
                "chosen": "سجل الدخول عبر النفاذ الوطني الموحد في منصة جدارات، ثم أكمل ملفك المهني وارفع المؤهلات، وبعدها استعرض الوظائف المتاحة وقدم عليها مباشرة.",
                "rejected": "ادخل على الموقع القديم لوزارة الخدمة المدنية وأرسل أوراقك بالبريد العادي وانتظر الرد.",
            }
        )
        self.assertIsNotNone(saudi_dpo)
        assert saudi_dpo is not None
        self.assertIsInstance(saudi_dpo["prompt"], str)
        self.assertIsInstance(saudi_dpo["prompt_messages"], list)
        self.assertIsInstance(saudi_dpo["chosen_only_messages"], list)
        self.assertIsInstance(saudi_dpo["rejected_only_messages"], list)

        arcee_dpo = normalize_arcee_arabic_dpo_row(
            {
                "chosen": [
                    {"role": "user", "content": "الصياغة المعتمدة للسؤال باللغة العربية حول أهمية الأمن السيبراني"},
                    {"role": "assistant", "content": "يحمي الأمن السيبراني البنية التحتية الرقمية والبيانات الحساسة من الهجمات الإلكترونية والاختراقات."},
                ],
                "rejected": [
                    {"role": "user", "content": "صياغة مختلفة تماما يجب استبدالها"},
                    {"role": "assistant", "content": "الأمن السيبراني مجرد برنامج لتسريع شاشة الحاسوب فقط لا غير."},
                ],
            }
        )
        self.assertIsNotNone(arcee_dpo)
        assert arcee_dpo is not None
        self.assertEqual(
            arcee_dpo["chosen_messages"][0]["content"],
            "الصياغة المعتمدة للسؤال باللغة العربية حول أهمية الأمن السيبراني",
        )
        self.assertEqual(
            arcee_dpo["rejected_messages"][0]["content"],
            "الصياغة المعتمدة للسؤال باللغة العربية حول أهمية الأمن السيبراني",
        )

        uf_dpo = normalize_ultrafeedback_dpo_row(
            {
                "prompt": "Explain why hash tables achieve O(1) average lookup time.",
                "chosen": [
                    {"role": "user", "content": "Explain why hash tables achieve O(1) average lookup time."},
                    {"role": "assistant", "content": "A hash function maps keys uniformly across buckets, keeping the average chain length constant when the load factor is bounded."},
                ],
                "rejected": [
                    {"role": "user", "content": "Explain why hash tables achieve O(1) average lookup time."},
                    {"role": "assistant", "content": "Hash tables sort every key alphabetically before scanning them one by one."},
                ],
                "chosen_rating": 5.0,
                "rejected_rating": 2.0,
            }
        )
        self.assertIsNotNone(uf_dpo)

    def test_06_vram_cpu_ple_hook_and_multimodal_offload(self) -> None:
        """Verify dynamic named_modules() PLE CPU hook and audio/vision tower offload."""
        class _MockGemma4Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.audio_tower = nn.Linear(16, 16)
                self.vision_tower = nn.Linear(16, 16)
                self.embed_tokens = nn.Embedding(64, 32)
                self.language_model = nn.Module()
                self.language_model.embed_tokens_per_layer = nn.Embedding(64, 256)
                self.lm_head = nn.Linear(32, 64, bias=False)

        model = _MockGemma4Backbone()
        freeze_embeddings_and_towers(model)
        state = apply_8gb_vram_optimizations(
            model=model,
            target_device="cpu",
            target_dtype=torch.bfloat16,
            non_blocking=True,
        )
        self.assertIn("language_model.embed_tokens_per_layer", state.ple_module_names)
        self.assertIn("audio_tower", state.multimodal_module_names)
        self.assertIn("vision_tower", state.multimodal_module_names)

        input_ids = torch.randint(0, 64, (2, 10))
        ple_out = model.language_model.embed_tokens_per_layer(input_ids)
        self.assertEqual(tuple(ple_out.shape), (2, 10, 256))
        self.assertEqual(ple_out.dtype, torch.bfloat16)

        telemetry = get_vram_telemetry(model)
        self.assertGreater(telemetry["total_params"], 0)
        self.assertEqual(telemetry["trainable_params"], 0)

        restored = restore_multimodal_towers(
            model=model,
            state=state,
            target_device="cpu",
            restore_ple=True,
        )
        self.assertEqual(len(state.hook_handles), 0)
        self.assertGreaterEqual(len(restored), 3)
        self.assertIn("model.audio_tower", build_8gb_device_map(0) if torch.cuda.is_available() else {"model.audio_tower": "cpu"})

    def test_07_trust_region_dpo_and_ties_merging_math(self) -> None:
        """Verify Trust-Region EMA (alpha=0.6) and TIES-Merging (lambda=0.85)."""
        ema_state = update_trust_region_reference_params(
            policy_params={"w": torch.tensor([10.0, 20.0])},
            ref_state={"w": torch.tensor([0.0, 10.0])},
            tr_alpha=DPO_TRUST_REGION_ALPHA,
        )
        self.assertTrue(torch.allclose(ema_state["w"], torch.tensor([6.0, 16.0])))

        vec = torch.tensor([0.01, -0.50, 2.00, -3.00])
        trimmed = ties_trim_tensor(vec, density=0.50)
        self.assertEqual(float(trimmed[0].item()), 0.0)
        self.assertEqual(float(trimmed[1].item()), 0.0)
        self.assertEqual(float(trimmed[2].item()), 2.0)
        self.assertEqual(float(trimmed[3].item()), -3.0)

        d1 = torch.tensor([2.0, -1.0])
        d2 = torch.tensor([4.0, 3.0])
        merged_delta = ties_merge_deltas([d1, d2], weights=[0.5, 0.5], density=1.0)
        self.assertAlmostEqual(float(merged_delta[0].item()), 3.0, places=5)
        self.assertAlmostEqual(float(merged_delta[1].item()), 3.0, places=5)

        base = {"layer.weight": torch.tensor([10.0, 20.0])}
        interpolated = apply_ties_interpolation_to_state_dict(
            base_state=base,
            adapter_deltas=[{"layer.weight": torch.tensor([2.0, 4.0])}],
            lambda_scale=DEFAULT_TIES_LAMBDA,
            density=1.0,
        )
        self.assertAlmostEqual(float(interpolated["layer.weight"][0].item()), 12.00, places=4)
        self.assertAlmostEqual(float(interpolated["layer.weight"][1].item()), 24.00, places=4)

    def test_08_retention_gate_and_markdown_report(self) -> None:
        """Verify Zero-Forgetting Retention Gate pass/fail logic and Markdown report."""
        anchors: Dict[str, Dict[str, Any]] = {
            "MMLU-Pro": {
                "domain": "english_reasoning",
                "lm_eval_task": "mmlu_pro",
                "num_fewshot": 5,
                "baseline_score": 69.4,
                "evaluated_score": 69.4,
                "max_regression_pct": 0.20,
            }
        }
        gate = evaluate_retention_gate(anchors, min_retention_ratio=0.998)
        self.assertTrue(gate["gate_passed"])

        md = generate_markdown_report(
            {
                "model_path": "outputs/merged_gemma4_e4b_arabic",
                "mode": "unit_test",
                "retention_gate": gate,
                "arabic_benchmarks": {
                    "ArabicMMLU": {
                        "lm_eval_task": "arabicmmlu",
                        "metric": "acc_norm",
                        "num_fewshot": 0,
                        "baseline_score": 55.2,
                        "evaluated_score": 64.5,
                    }
                },
                "retention_anchors": anchors,
            }
        )
        self.assertIn("ArabicMMLU", md)
        self.assertIn("PASSED", md)
        self.assertIn("Base (%) | Arabic-Tuned (%)", md)
        self.assertNotIn("Google Official", md)

    def test_09_dual_model_a100_configs_and_router_freeze(self) -> None:
        """Verify A100 80 GB configs for E4B and 26B-A4B, MoE router freeze, and lm_eval JSON parser."""
        for fname in (
            "sft_e4b_a100.yaml",
            "dpo_e4b_a100.yaml",
            "sft_26b_a4b_a100.yaml",
            "dpo_26b_a4b_a100.yaml",
        ):
            cfg_path = PROJECT_ROOT / "configs" / fname
            self.assertTrue(cfg_path.exists(), f"Missing config: {cfg_path}")
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            self.assertFalse(cfg["vram_optimization"]["offload_ple_to_cpu"])
            self.assertFalse(cfg["vram_optimization"]["offload_multimodal_towers_to_cpu"])
            self.assertEqual(cfg["training"]["optim"], "adamw_bnb_8bit")

        sft_e4b = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "sft_e4b_a100.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(sft_e4b["lora"]["r"], 16)
        self.assertEqual(sft_e4b["lora"]["lora_alpha"], 16)
        self.assertEqual(sft_e4b["datasets"]["total_target_samples"], 30000)
        self.assertFalse(sft_e4b["lora"]["use_rslora"])

        sft_26b = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "sft_26b_a4b_a100.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(sft_26b["model"]["base_model_id"], PRIMARY_MODEL_ID_26B)
        self.assertEqual(sft_26b["datasets"]["total_target_samples"], 30000)
        self.assertEqual(sft_26b["lora"]["r"], 16)
        self.assertEqual(sft_26b["lora"]["lora_alpha"], 16)
        self.assertFalse(sft_26b["lora"]["use_rslora"])
        self.assertIsNone(sft_26b["lora"]["target_modules"])
        self.assertTrue(sft_26b["vram_optimization"]["freeze_moe_router"])

        class _MockMoEBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = nn.Linear(16, 8, bias=False)
                self.gate_proj = nn.Linear(16, 32, bias=False)

        moe_block = _MockMoEBlock()
        freeze_embeddings_and_towers(moe_block, freeze_moe_router=True)
        self.assertFalse(moe_block.router.weight.requires_grad)
        self.assertTrue(moe_block.gate_proj.weight.requires_grad)

        # Verify lm_eval JSON parser converts [0, 1] fraction to [0, 100] percentage
        sample_lm_eval = {
            "results": {
                "arabicmmlu": {
                    "acc_norm,none": 0.645,
                    "acc_norm_stderr,none": 0.012,
                }
            }
        }
        parsed_score = parse_lm_eval_results_json(
            sample_lm_eval, task_name="arabicmmlu", metric_hint="acc_norm"
        )
        self.assertAlmostEqual(parsed_score, 64.50, places=2)

        # Verify --model-family CLI alias, --no-reference-fallback default, and Gemma4Config text_config patch
        parsed_cli = parse_bench_args(
            ["--model-family", "both", "--eval-base", "--use-vllm", "--use-cached-eval"]
        )
        self.assertEqual(parsed_cli.model_key, "both")
        self.assertTrue(parsed_cli.eval_base)
        self.assertTrue(parsed_cli.no_reference_fallback)
        self.assertTrue(parsed_cli.use_vllm)
        self.assertTrue(parsed_cli.use_cached_eval)

        class _MockTextCfg:
            vocab_size = 262144
            hidden_size = 2560

        class _MockGemma4Cfg:
            text_config = _MockTextCfg()

        mock_cfg = _MockGemma4Cfg()
        _patch_gemma4_config_for_lm_eval(mock_cfg)
        self.assertEqual(getattr(mock_cfg, "vocab_size", None), 262144)

        # Verify custom lm_eval task specs and subtask limit scaling
        for custom_key in (
            "arabicmmlu",
            "madinahqa",
            "aratrust",
            "mgsm_ar",
            "mmlu_pro",
            "arc_challenge_chat",
            "humaneval_instruct",
        ):
            spec_dict = build_custom_lm_eval_task_dict(custom_key, num_fewshot=0)
            self.assertIn("task", spec_dict)
            self.assertIn("dataset_path", spec_dict)
            self.assertIn("output_type", spec_dict)

        mmlu_pro_spec = build_custom_lm_eval_task_dict("mmlu_pro", num_fewshot=5)
        self.assertEqual(mmlu_pro_spec["dataset_path"], "TIGER-Lab/MMLU-Pro")
        self.assertEqual(mmlu_pro_spec["doc_to_target"], "{{target_letter}}")
        he_spec = build_custom_lm_eval_task_dict("humaneval_instruct", num_fewshot=0)
        self.assertIn("```python\n{{prompt}}\n```", he_spec["doc_to_text"])
        self.assertNotIn("Here is the completed function:", he_spec["doc_to_text"])
        self.assertNotIn("\n```", he_spec["generation_kwargs"]["until"])
        self.assertEqual(he_spec["generation_kwargs"]["max_gen_toks"], 4096)

        mgsm_spec = build_custom_lm_eval_task_dict("mgsm_ar", num_fewshot=5)
        self.assertEqual(mgsm_spec["doc_to_target"], "{{target_number}}")
        self.assertEqual(mgsm_spec["generation_kwargs"]["max_gen_toks"], 4096)
        self.assertIn("<turn|>", mgsm_spec["generation_kwargs"]["until"])
        self.assertNotIn("<|turn>", mgsm_spec["generation_kwargs"]["until"])
        self.assertIn("\nالسؤال:", mgsm_spec["generation_kwargs"]["until"])
        self.assertIn("\nسؤال:", mgsm_spec["generation_kwargs"]["until"])
        self.assertIn("\nQuestion:", mgsm_spec["generation_kwargs"]["until"])
        self.assertNotIn("السؤال:", mgsm_spec["generation_kwargs"]["until"])
        self.assertNotIn("Question:", mgsm_spec["generation_kwargs"]["until"])
        self.assertIn("\nQuestion:", mmlu_pro_spec["generation_kwargs"]["until"])
        self.assertNotIn("Question:", mmlu_pro_spec["generation_kwargs"]["until"])
        arc_spec = build_custom_lm_eval_task_dict("arc_challenge_chat", num_fewshot=0)
        self.assertIn("\nQuestion:", arc_spec["generation_kwargs"]["until"])
        self.assertNotIn("Question:", arc_spec["generation_kwargs"]["until"])
        self.assertEqual(arc_spec["generation_kwargs"]["max_gen_toks"], 4096)
        self.assertIsNone(_get_lm_eval_gen_kwargs("gsm8k"))
        self.assertIsNone(_get_lm_eval_gen_kwargs("humaneval"))
        self.assertIsNone(_get_lm_eval_gen_kwargs("arc_challenge"))
        self.assertEqual(
            _get_lm_eval_gen_kwargs("mmlu_pro", use_vllm=True),
            {
                "max_gen_toks": 24576,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "do_sample": True,
                "until": ["<turn|>", "<end_of_turn>", "<eos>", "</s>", "<|im_end|>"],
                "skip_special_tokens": False,
            },
        )
        self.assertEqual(
            _get_lm_eval_gen_kwargs("mmlu_pro", model_key_prefix="26b_a4b", use_vllm=False),
            {
                "max_gen_toks": 24576,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "do_sample": True,
                "until": ["<turn|>", "<end_of_turn>", "<eos>", "</s>", "<|im_end|>"],
            },
        )
        self.assertFalse(_should_apply_chat_template_for_task("arc_challenge", True))
        self.assertFalse(_should_apply_chat_template_for_task("humaneval", True))
        self.assertFalse(_should_apply_chat_template_for_task("gsm8k", True))
        self.assertFalse(_should_apply_chat_template_for_task("mgsm_ar", True))
        self.assertFalse(_should_apply_chat_template_for_task("arc_challenge", True, model_key_prefix="26b_a4b"))
        self.assertFalse(_should_apply_chat_template_for_task("gsm8k", True, model_key_prefix="26b_a4b"))
        self.assertFalse(_should_apply_chat_template_for_task("mgsm_ar", True, model_key_prefix="26b_a4b"))
        self.assertFalse(_should_apply_chat_template_for_task("humaneval", True, model_key_prefix="26b_a4b"))
        self.assertTrue(_should_apply_chat_template_for_task("arc_challenge_chat", True))
        self.assertTrue(_should_apply_chat_template_for_task("mmlu_pro", True))
        self.assertTrue(_should_apply_chat_template_for_task("mmlu_pro", True, model_key_prefix="26b_a4b"))
        self.assertTrue(_should_apply_chat_template_for_task("humaneval_instruct", True))

        arc_mc_spec = build_custom_lm_eval_task_dict("arc_challenge", num_fewshot=25)
        self.assertEqual(arc_mc_spec["output_type"], "multiple_choice")
        self.assertEqual(arc_mc_spec["num_fewshot"], 25)
        self.assertEqual(arc_mc_spec["doc_to_choice"], "choice_letters")
        self.assertEqual(arc_mc_spec["doc_to_target"], "gold")

        he_base_spec = build_custom_lm_eval_task_dict("humaneval", num_fewshot=0)
        self.assertTrue(
            he_base_spec["doc_to_text"].endswith("Here is the completed function:\n```python\n{{prompt}}")
        )
        self.assertEqual(he_base_spec["generation_kwargs"]["max_gen_toks"], 4096)
        self.assertNotIn("\n```", he_base_spec["generation_kwargs"]["until"])
        self.assertNotIn("\ndef", he_base_spec["generation_kwargs"]["until"])

        class _MockTok:
            eos_token = "<turn|>"
            eos_token_id = 106

        class _MockLMObj:
            enable_thinking = True
            think_end_token: Any = None
            chat_template_args: Dict[str, Any] = {}
            tokenizer = _MockTok()

        mock_vllm = _MockLMObj()
        _configure_lm_eval_thinking_for_task(
            mock_vllm, effective_use_vllm=True, task_chat_tpl=True, task_name="mmlu_pro"
        )
        self.assertTrue(mock_vllm.enable_thinking)
        self.assertEqual(mock_vllm.think_end_token, "<channel|>")
        self.assertEqual(mock_vllm.chat_template_args, {})
        _configure_lm_eval_thinking_for_task(
            mock_vllm,
            effective_use_vllm=True,
            task_chat_tpl=True,
            task_name="mmlu_pro",
            model_key_prefix="26b_a4b",
        )
        self.assertTrue(mock_vllm.enable_thinking)
        self.assertEqual(mock_vllm.think_end_token, "<channel|>")
        self.assertEqual(mock_vllm.tokenizer.eos_token_id, 106)
        _configure_lm_eval_thinking_for_task(
            mock_vllm,
            effective_use_vllm=True,
            task_chat_tpl=False,
            task_name="gsm8k",
            model_key_prefix="26b_a4b",
        )
        self.assertFalse(mock_vllm.enable_thinking)
        self.assertIsNone(mock_vllm.think_end_token)
        self.assertEqual(mock_vllm.tokenizer.eos_token, "<eos>")
        self.assertEqual(mock_vllm.tokenizer.eos_token_id, 1)
        _configure_lm_eval_thinking_for_task(
            mock_vllm, effective_use_vllm=True, task_chat_tpl=False, task_name="arc_challenge"
        )
        self.assertFalse(mock_vllm.enable_thinking)
        self.assertIsNone(mock_vllm.think_end_token)
        self.assertEqual(mock_vllm.chat_template_args, {})

        mock_hflm = _MockLMObj()
        _configure_lm_eval_thinking_for_task(
            mock_hflm, effective_use_vllm=False, task_chat_tpl=True, task_name="mmlu_pro"
        )
        self.assertTrue(mock_hflm.enable_thinking)
        self.assertEqual(mock_hflm.think_end_token, "<channel|>")
        self.assertEqual(mock_hflm.chat_template_args, {"enable_thinking": True})
        _configure_lm_eval_thinking_for_task(
            mock_hflm, effective_use_vllm=False, task_chat_tpl=True, task_name="gsm8k"
        )
        self.assertFalse(mock_hflm.enable_thinking)
        self.assertIsNone(mock_hflm.think_end_token)
        self.assertEqual(mock_hflm.chat_template_args, {})
        self.assertEqual(
            mgsm_spec["metric_list"][0]["regexes_to_ignore"],
            [",", "\\$", "\\.$"],
        )
        self.assertEqual(
            _strip_thinking_blocks("<|think|>\nstep 1\n<|think|>\nFinal Answer: (C)"),
            "Final Answer: (C)",
        )
        self.assertEqual(
            _extract_arc_mcq_letter("<|think|>\nThinking...\n<|think|>\nFinal Answer: (D)", "ABCDEFGHIJ"),
            "D",
        )
        self.assertEqual(
            _extract_arc_mcq_letter("Let's see what the answer is. The answer is **C**.", "ABCDEFGHIJ"),
            "C",
        )
        self.assertEqual(
            _extract_arc_mcq_letter("The answer is based on Hooke's law. Final Answer: **D**", "ABCDEFGHIJ"),
            "D",
        )
        self.assertEqual(
            _extract_arc_mcq_letter(
                "Let's check (i) and f(a). The answer is option D. A great question, I agree.",
                "ABCDEFGHIJ",
            ),
            "D",
        )
        self.assertEqual(
            _extract_arc_mcq_letter("Therefore, \\boxed{G}.", "ABCDEFGHIJ"),
            "G",
        )
        cot_mcq = "Let's evaluate: (A) is 10, (B) is 20, which matches, whereas (C) and (D) are wrong. Therefore, **B**."
        self.assertEqual(_extract_arc_mcq_letter(cot_mcq, "ABCD"), "B")
        self.assertEqual(_extract_arc_mcq_letter("Option B: The sun heats the water.", "ABCD"), "B")
        self.assertEqual(
            _extract_arc_mcq_letter("The correct option is: **B. The sun heats the water.**", "ABCD"),
            "B",
        )
        self.assertEqual(
            _extract_arc_mcq_letter(
                "B\n\nExplanation: The sun heats the water, whereas (A) wind and (C) moon do not.",
                "ABCD",
            ),
            "B",
        )
        self.assertEqual(
            _extract_arc_mcq_letter("B\nBecause solar radiation warms the surface.", "ABCD"),
            "B",
        )
        self.assertEqual(
            _extract_arc_mcq_letter(
                "The correct answer is C. Note that answer A is incorrect because wind is secondary.",
                "ABCD",
            ),
            "C",
        )
        self.assertEqual(
            _extract_arc_mcq_letter(
                "Final Answer: B. Note that the answer D is wrong.",
                "ABCD",
            ),
            "B",
        )
        self.assertEqual(
            _strip_thinking_blocks("<turn|>\n<|turn>model\n<|channel>thought\nLet's think\n<channel|>    return a + b"),
            "    return a + b",
        )
        self.assertEqual(
            _strip_thinking_blocks("<|channel>thought\nUnclosed thought with The answer is (D)"),
            "Unclosed thought with The answer is (D)",
        )
        self.assertEqual(
            _extract_mcq_letter_with_doc(
                "The sun heats the water.",
                {"choices": {"text": ["Wind", "The sun heats the water.", "Moon", "Stars"]}},
                "ABCD",
            ),
            "B",
        )
        raw_he = (
            "We need to check if any two numbers are closer than threshold.\n"
            "    for idx, elem in enumerate(numbers):\n"
            "        if idx > 0:\n"
            "            return True\n"
            "    return False\n"
            "```\n"
            "Example usage:\n```python\nprint(foo([1, 2], 0.5))\n```"
        )
        extracted_he = _extract_humaneval_code(raw_he, "def foo(numbers, threshold):\n", "foo")
        self.assertIn("for idx, elem in enumerate(numbers):", extracted_he)
        self.assertIn("return False", extracted_he)
        self.assertNotIn("We need to check", extracted_he)
        self.assertNotIn("print(foo", extracted_he)
        self.assertEqual(
            _extract_humaneval_code("    x = 1\n    return x\n```", "def foo():\n", "foo"),
            "def foo():\n    x = 1\n    return x\n",
        )
        self.assertEqual(
            _extract_humaneval_code(
                "Thinking about the problem...\n```python\ndef foo(a, b):\n    return a + b\n```",
                "def foo(a, b):\n",
                "foo",
            ),
            "def foo(a, b):\n    return a + b\n",
        )
        self.assertEqual(
            _extract_humaneval_code(
                "    x = a + b\n    return x\n\nExplanation: This adds a and b.\nprint(foo(1, 2))",
                "def foo(a, b):\n",
                "foo",
            ),
            "def foo(a, b):\n    x = a + b\n    return x\n",
        )
        self.assertEqual(
            _extract_humaneval_code(
                "We need to add a and b together.\n    x = a + b\n    return x\n",
                "def foo(a, b):\n",
                "foo",
            ),
            "def foo(a, b):\n    x = a + b\n    return x\n",
        )
        he_code = _extract_humaneval_code(
            "<|think|>\n<|think|>\n```python\ndef add(a, b):\n    return a + b\n```",
            "def add(a, b):\n",
            "add",
        )
        self.assertIn("def add(a, b):", he_code)
        self.assertIn("return a + b", he_code)
        self.assertNotIn("```", he_code)
        self.assertEqual(_extract_mgsm_ar_target_number("خطوات الحل...\n### 70000"), "70000")
        self.assertEqual(_extract_mgsm_ar_target_number("خطوات الحل...\n##### 57,500"), "57500")
        self.assertEqual(_extract_mgsm_ar_target_number("المجموع هو 126+240 = 366"), "366")
        self.assertEqual(_extract_mgsm_ar_target_number("الناتج النهائي #### ٧٠٠٠٠"), "70000")
        self.assertEqual(_normalize_arabic_digits("٥٧٬٥٠٠٫٢٥"), "57,500.25")
        self.assertEqual(_extract_mgsm_ar_target_number("الإجابة هي ٥٧٬٥٠٠"), "57500")
        self.assertEqual(
            _truncate_math_continuation(
                "المعطيات في السؤال: اشترى أحمد 5 كتب بسعر 10 ريال.\n5 * 10 = 50\n### 50\nالسؤال: سؤال آخر"
            ),
            "المعطيات في السؤال: اشترى أحمد 5 كتب بسعر 10 ريال.\n5 * 10 = 50\n### 50",
        )
        mgsm_rescored = extract_lm_eval_detailed_metrics(
            {
                "results": {"mgsm_ar": {"exact_match,flexible-extract": 0.84}},
                "samples": {"mgsm_ar": [{"target": "57500"}, {"target": "70000"}]},
            },
            task_name="mgsm_ar",
            metric_hint="exact_match",
        )
        self.assertEqual(mgsm_rescored["eval_score_pct"], 84.0)
        gsm8k_strict = extract_lm_eval_detailed_metrics(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.763,
                        "exact_match,flexible-extract": 0.910,
                    }
                },
                "samples": {"gsm8k": [{"target": "42"}]},
            },
            task_name="gsm8k",
            metric_hint="exact_match,strict-match",
        )
        self.assertEqual(gsm8k_strict["eval_score_pct"], 76.3)
        self.assertEqual(
            _normalize_arabic_digits_filter([["<|think|>\n١٢٣\n<|think|>\nالإجابة هي ٥٧٥٠٠"]], [{}]),
            [["الإجابة هي 57500"]],
        )
        truncated_math = _normalize_arabic_digits_filter(
            [["الحل هو 10 + 5 = 15\n#### 15\nالسؤال: اشترى أحمد 3 تفاحات..."]],
            [{}],
        )[0][0]
        self.assertTrue(truncated_math.endswith("#### 15"))
        self.assertNotIn("اشترى", truncated_math)

        for raw_candidate in (
            "    return a + b\n\ndef foo(a, b):\n    return a + b\n",
            "Sure! Here is the function:\ndef foo(a, b):\n    return a + b\n",
            "    return a + b\n```\nHere is an example:\n```python\nprint(foo(1, 2))\n```",
            "def foo(a, b):\n    return _helper(a, b)\n\ndef _helper(x, y):\n    return x + y\n",
        ):
            code_out = _extract_humaneval_code(raw_candidate, "def foo(a, b):\n", "foo")
            compile(code_out, "<humaneval>", "exec")
            ns: Dict[str, Any] = {}
            exec(code_out, ns)
            self.assertEqual(ns["foo"](3, 4), 7)

        double_code = _extract_humaneval_code(
            "```python\nres = []\nfor x in numbers:\n    res.append(x * 2)\nreturn res\n```",
            'def double_list(numbers):\n    """Double all numbers."""\n',
            "double_list",
        )
        ns_double: Dict[str, Any] = {}
        exec(double_code, ns_double)
        self.assertEqual(ns_double["double_list"]([1, 2, 3]), [2, 4, 6])

        sort_code = _extract_humaneval_code(
            "numbers.sort()\nreturn numbers",
            "def sort_nums(numbers):\n",
            "sort_nums",
        )
        ns_sort: Dict[str, Any] = {}
        exec(sort_code, ns_sort)
        self.assertEqual(ns_sort["sort_nums"]([3, 1, 2]), [1, 2, 3])

        indented_comment_col0_body = _extract_humaneval_code(
            "    # Sort the list in ascending order\nnumbers.sort()\nreturn numbers",
            "def sort_nums(numbers):\n",
            "sort_nums",
        )
        ns_sort2: Dict[str, Any] = {}
        exec(indented_comment_col0_body, ns_sort2)
        self.assertEqual(ns_sort2["sort_nums"]([4, 1, 3]), [1, 3, 4])

        col0_prefix_before_closing_fence = _extract_humaneval_code(
            "res = a + b\nreturn res\n```\nExample:\n```python\nprint(foo(1, 2))\n```",
            "def foo(a, b):\n",
            "foo",
        )
        ns_col0_prefix: Dict[str, Any] = {}
        exec(col0_prefix_before_closing_fence, ns_col0_prefix)
        self.assertEqual(ns_col0_prefix["foo"](5, 6), 11)

        eval_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval_config.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(eval_cfg["retention_gate"]["min_retention_ratio"], 0.95)
        self.assertEqual(eval_cfg["retention_gate"]["max_allowed_drop_pct"], 5.00)
        for anchor_name in ("MMLU-Pro", "GPQA-Diamond", "AIME-2026", "LiveCodeBench-v6"):
            self.assertEqual(
                eval_cfg["retention_anchors"][anchor_name]["max_regression_pct"],
                5.00,
            )
            self.assertTrue(eval_cfg["retention_anchors"][anchor_name]["enable_thinking"])
        for removed_anchor in ("GSM8K", "HumanEval", "ARC-Challenge"):
            self.assertNotIn(removed_anchor, eval_cfg["retention_anchors"])
        self.assertNotIn("MGSM-Ar", eval_cfg["arabic_benchmarks"])
        self.assertEqual(eval_cfg["evaluation"]["base_model_id"], "unsloth/gemma-4-E4B-it")
        self.assertEqual(eval_cfg["evaluation"]["batch_size"], 32)

        # Verify _ensure_gemma4_tied_lm_head and ensure_tied_lm_head fix untied lm_head.weight
        class _MockUntiedGemma4(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = _MockGemma4Cfg()
                self.config.tie_word_embeddings = False
                self.config.text_config.tie_word_embeddings = False
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.embed_tokens = nn.Embedding(32, 16)
                self.lm_head = nn.Linear(16, 32, bias=False)

        untied_m1 = _MockUntiedGemma4()
        self.assertNotEqual(
            untied_m1.lm_head.weight.data_ptr(),
            untied_m1.model.language_model.embed_tokens.weight.data_ptr(),
        )
        _ensure_gemma4_tied_lm_head(untied_m1, tag="unit_test")
        self.assertEqual(
            untied_m1.lm_head.weight.data_ptr(),
            untied_m1.model.language_model.embed_tokens.weight.data_ptr(),
        )
        self.assertTrue(untied_m1.config.tie_word_embeddings)
        self.assertTrue(untied_m1.config.text_config.tie_word_embeddings)

        untied_m2 = _MockUntiedGemma4()
        self.assertTrue(ensure_tied_lm_head(untied_m2))
        self.assertEqual(
            untied_m2.lm_head.weight.data_ptr(),
            untied_m2.model.language_model.embed_tokens.weight.data_ptr(),
        )

        with tempfile.TemporaryDirectory(prefix="gemma4_tie_cfg_") as tmp_dir:
            tmp_p = pathlib.Path(tmp_dir)
            cfg_file = tmp_p / "config.json"
            cfg_file.write_text(
                json.dumps(
                    {
                        "tie_word_embeddings": False,
                        "use_cache": False,
                        "text_config": {"tie_word_embeddings": False, "use_cache": False},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_patch_exported_gemma4_config_tie_embeddings(tmp_p))
            _patch_local_checkpoint_config_json(tmp_p)
            patched_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            self.assertTrue(patched_cfg["tie_word_embeddings"])
            self.assertTrue(patched_cfg["use_cache"])
            self.assertTrue(patched_cfg["text_config"]["tie_word_embeddings"])
            self.assertTrue(patched_cfg["text_config"]["use_cache"])

        self.assertEqual(compute_effective_task_limit("arabicmmlu", 200), 5)
        self.assertEqual(compute_effective_task_limit("arabic_leaderboard_alghafa_light", 180), 20)
        self.assertEqual(compute_effective_task_limit("arabic_leaderboard_acva_light", 200), 40)
        self.assertEqual(compute_effective_task_limit("mmlu_pro", 200), 15)
        self.assertEqual(compute_effective_task_limit("leaderboard_mmlu_pro", 200), 200)
        self.assertEqual(
            resolve_lm_eval_task_specs("mmlu_pro", num_fewshot=5, limit=200),
            (["mmlu_pro"], "mmlu_pro", 15),
        )

        tasks_armmlu, primary_armmlu, eff_armmlu = resolve_lm_eval_task_specs(
            "arabicmmlu", num_fewshot=0, limit=200
        )
        self.assertEqual(tasks_armmlu, ["arabicmmlu"])
        self.assertEqual(primary_armmlu, "arabicmmlu")
        self.assertEqual(eff_armmlu, 5)

        self.assertEqual(eval_cfg["arabic_benchmarks"]["ArabicMMLU"]["runner"], "lighteval")
        self.assertEqual(eval_cfg["arabic_benchmarks"]["ArabicMMLU"]["num_fewshot"], 0)
        self.assertEqual(eval_cfg["arabic_benchmarks"]["AlGhafa"]["runner"], "lighteval")
        self.assertEqual(eval_cfg["arabic_benchmarks"]["MadinahQA"]["runner"], "lighteval")
        self.assertEqual(eval_cfg["arabic_benchmarks"]["MadinahQA"]["num_fewshot"], 0)
        self.assertEqual(eval_cfg["arabic_benchmarks"]["ACVA"]["runner"], "lighteval")
        self.assertEqual(eval_cfg["arabic_benchmarks"]["EXAMS-AR"]["runner"], "lighteval")
        self.assertEqual(eval_cfg["arabic_benchmarks"]["AraTrust"]["runner"], "lighteval")
        self.assertEqual(eval_cfg["arabic_benchmarks"]["ACVA"]["num_fewshot"], 5)

        self.assertEqual(eval_cfg["retention_anchors"]["MMLU-Pro"]["runner"], "lm_eval")
        self.assertEqual(eval_cfg["retention_anchors"]["MMLU-Pro"]["lm_eval_task"], "mmlu_pro")
        self.assertEqual(eval_cfg["retention_anchors"]["MMLU-Pro"]["num_fewshot"], 5)
        self.assertEqual(
            eval_cfg["retention_anchors"]["MMLU-Pro"]["metric"], "exact_match,custom-extract"
        )
        self.assertEqual(eval_cfg["retention_anchors"]["GPQA-Diamond"]["runner"], "inspect_evals")
        self.assertEqual(
            eval_cfg["retention_anchors"]["GPQA-Diamond"]["lm_eval_task"],
            "inspect_evals/gpqa_diamond",
        )
        self.assertEqual(eval_cfg["retention_anchors"]["GPQA-Diamond"]["epochs"], 1)
        self.assertEqual(eval_cfg["retention_anchors"]["AIME-2026"]["runner"], "inspect_evals")
        self.assertEqual(
            eval_cfg["retention_anchors"]["AIME-2026"]["lm_eval_task"],
            "inspect_evals/aime2026",
        )
        self.assertEqual(eval_cfg["retention_anchors"]["AIME-2026"]["epochs"], 4)
        self.assertEqual(
            eval_cfg["retention_anchors"]["LiveCodeBench-v6"]["runner"], "lighteval_lcb"
        )
        self.assertEqual(
            eval_cfg["retention_anchors"]["LiveCodeBench-v6"]["lighteval_task"],
            "lcb:codegeneration_v6|0",
        )
        self.assertEqual(eval_cfg["retention_anchors"]["LiveCodeBench-v6"]["num_samples"], 1)

        # Verify full 40-subject ArabicMMLU + 58-subtask ACVA expansion when limit=None vs smoke limit
        self.assertEqual(len(ARABIC_MMLU_40_SUBSETS), 40)
        self.assertEqual(len(ACVA_ALL_58_SUBSETS), 58)
        full_ar_tasks = resolve_lighteval_arabic_tasks(["ArabicMMLU", "ACVA"], limit=None)
        self.assertEqual(len(full_ar_tasks), 98)
        limited_ar_tasks = resolve_lighteval_arabic_tasks(["ArabicMMLU", "ACVA"], limit=200)
        self.assertEqual(len(limited_ar_tasks), 6)
        self.assertEqual(limited_ar_tasks[0], "arabic_mmlu:All|0")

        le_payload = {
            "results": {
                "community:arabic_mmlu:All:5": {
                    "acc:logprob_normalization=LogProbCharNorm(name='norm', ignore_first_space=False)": 0.678,
                    "acc:logprob_normalization=LogProbCharNorm(name='norm', ignore_first_space=False)_stderr": 0.014,
                },
                "community:alghafa:_average:0": {"acc_norm": 0.612},
                "community:acva:_average:5": {"acc_norm": 0.784},
                "community:arabic_exams:default:0": {"acc_norm": 0.565},
                "madinah_qa:_average|5": {"acc_norm": 0.725},
                "aratrust:_average|0": {"acc_norm": 0.815},
                "lcb:codegeneration_v6|0": {
                    "codegen_pass@1:16": 0.2857,
                    "codegen_pass@1:16_stderr": 0.034,
                },
            },
            "doc_counts": {
                "lcb:codegeneration_v6|0": 175,
                "arabic_exams|0": 537,
            },
        }
        self.assertAlmostEqual(
            parse_lighteval_results_json(le_payload, "ArabicMMLU")["eval_score_pct"],
            67.80,
            places=2,
        )
        self.assertAlmostEqual(
            parse_lighteval_results_json(le_payload, "AlGhafa")["eval_score_pct"],
            61.20,
            places=2,
        )
        self.assertAlmostEqual(
            parse_lighteval_results_json(le_payload, "ACVA")["eval_score_pct"],
            78.40,
            places=2,
        )
        exams_parsed = parse_lighteval_results_json(le_payload, "EXAMS-AR")
        self.assertAlmostEqual(exams_parsed["eval_score_pct"], 56.50, places=2)
        self.assertEqual(exams_parsed["n_samples"], 537)
        self.assertAlmostEqual(
            parse_lighteval_results_json(le_payload, "MadinahQA")["eval_score_pct"],
            72.50,
            places=2,
        )
        self.assertAlmostEqual(
            parse_lighteval_results_json(le_payload, "AraTrust")["eval_score_pct"],
            81.50,
            places=2,
        )
        lcb_parsed = parse_lighteval_results_json(le_payload, "LiveCodeBench-v6")
        self.assertAlmostEqual(lcb_parsed["eval_score_pct"], 28.57, places=2)
        self.assertEqual(lcb_parsed["n_samples"], 175)

        # Verify 58-subtask unweighted macro-average on ACVA when individual subtasks are returned
        acva_58_payload = {
            "results": {
                f"custom|acva:{sub}|5": {"acc_norm": 0.80} for sub in ACVA_ALL_58_SUBSETS
            },
            "doc_counts": {f"custom|acva:{sub}|5": 150 for sub in ACVA_ALL_58_SUBSETS},
        }
        acva_58_parsed = parse_lighteval_results_json(acva_58_payload, "ACVA")
        self.assertAlmostEqual(acva_58_parsed["eval_score_pct"], 80.00, places=2)
        self.assertEqual(acva_58_parsed["n_samples"], 58 * 150)

        # Verify extract_lm_eval_detailed_metrics reports raw unmodified lm_eval score (zero post-hoc max() override)
        mmlu_pro_zero_payload = {
            "results": {"mmlu_pro": {"exact_match,custom-extract": 0.0}},
            "samples": {
                "mmlu_pro_biology": [
                    {
                        "target": "(C)",
                        "resps": [["<|think|>\nLet's solve.\n<|think|>\nFinal Answer: **C**"]],
                    }
                ]
            },
        }
        mmlu_pro_zero = extract_lm_eval_detailed_metrics(
            mmlu_pro_zero_payload, task_name="mmlu_pro", metric_hint="exact_match,custom-extract"
        )
        self.assertAlmostEqual(mmlu_pro_zero["eval_score_pct"], 0.0, places=2)

        mmlu_pro_live_payload = {
            "results": {"mmlu_pro": {"exact_match,custom-extract": 0.725}},
            "samples": {"mmlu_pro_biology": [{"target": "(C)", "resps": [["The answer is (C)"]]}]},
        }
        mmlu_pro_live = extract_lm_eval_detailed_metrics(
            mmlu_pro_live_payload, task_name="mmlu_pro", metric_hint="exact_match,custom-extract"
        )
        self.assertAlmostEqual(mmlu_pro_live["eval_score_pct"], 72.50, places=2)

        # Verify clean_gemma4_completion handles closed thoughts and drops unclosed mid-thought truncations
        self.assertEqual(
            clean_gemma4_completion(
                "<|channel>thought\nStep 1: check options.\n<channel|>The answer is (D)<turn|>extra"
            ),
            "The answer is (D)",
        )
        self.assertEqual(
            clean_gemma4_completion(
                "<|channel>thought\nUnfinished reasoning hitting max_tokens with (D)",
                drop_unclosed_thought=True,
            ),
            "",
        )

        le_payload_colon = {
            "results": {
                "madinah_qa:_average:5": {"acc_norm": 0.742},
                "aratrust:_average:0": {"acc_norm": 0.831},
            }
        }
        self.assertAlmostEqual(
            parse_lighteval_results_json(le_payload_colon, "MadinahQA")["eval_score_pct"],
            74.20,
            places=2,
        )
        self.assertAlmostEqual(
            parse_lighteval_results_json(le_payload_colon, "AraTrust")["eval_score_pct"],
            83.10,
            places=2,
        )
        child_cmd = _build_child_eval_cmd(parsed_cli, "26b_a4b")
        self.assertIn("26b_a4b", child_cmd)
        self.assertIn("--use-vllm", child_cmd)
        self.assertIn("--apply-chat-template", child_cmd)

        # Verify interrupt-safe atomic stage checkpointing and resume invariants
        with tempfile.TemporaryDirectory(prefix="gemma4_stage_ckpt_") as tmp_cache_dir:
            ckpt_file = pathlib.Path(tmp_cache_dir) / "live_raw_e4b.json"
            _atomic_write_json(
                ckpt_file,
                {
                    "model_key": "e4b",
                    "base_model_id": "unsloth/gemma-4-E4B-it",
                    "tuned_model_path": "outputs/merged_gemma4_e4b_arabic/merged_16bit",
                    "limit": None,
                    "base": {
                        "arabic": {"ArabicMMLU": {"eval_score_pct": 66.40}},
                        "retention": {
                            "MMLU-Pro": {"eval_score_pct": 69.40},
                            "AIME-2026": {"eval_score_pct": 0.0},
                            "GPQA-Diamond": {"_skipped": True},
                        },
                    },
                    "tuned": {
                        "arabic": {},
                        "retention": {},
                    },
                },
            )
            b_ar, b_ret, t_ar, t_ret = _load_stage_checkpoint(
                cache_path=ckpt_file,
                limit=None,
                base_model_id="unsloth/gemma-4-E4B-it",
                tuned_model_path="outputs/merged_gemma4_e4b_arabic/merged_16bit",
            )
            self.assertEqual(set(b_ar.keys()), {"ArabicMMLU"})
            self.assertEqual(set(b_ret.keys()), {"MMLU-Pro", "AIME-2026"})
            self.assertEqual(b_ret["AIME-2026"]["eval_score_pct"], 0.0)
            self.assertEqual(t_ar, {})
            self.assertEqual(t_ret, {})

            # Limit mismatch must reject the checkpoint cleanly
            mismatch = _load_stage_checkpoint(
                cache_path=ckpt_file,
                limit=200,
                base_model_id="unsloth/gemma-4-E4B-it",
                tuned_model_path="outputs/merged_gemma4_e4b_arabic/merged_16bit",
            )
            self.assertEqual(mismatch, ({}, {}, {}, {}))

        sel_ar, sel_ret = _resolve_selected_suites(
            eval_cfg, "arabicmmlu,acva,mmlu,gpqa,aime,lcb"
        ) or ([], [])
        self.assertEqual(sel_ar, ["ArabicMMLU", "ACVA"])
        self.assertEqual(
            sel_ret, ["MMLU-Pro", "GPQA-Diamond", "AIME-2026", "LiveCodeBench-v6"]
        )

        # Verify family Markdown report uses strictly Base | Arabic-Tuned columns
        mock_family_item = {
            "model_path": "outputs/merged_gemma4_e4b_arabic/merged_16bit",
            "average_arabic_gain": 2.15,
            "retention_gate": {"gate_passed": True, "min_observed_retention_ratio": 0.992},
            "arabic_benchmarks": {
                "ArabicMMLU": {
                    "metric": "acc_norm",
                    "num_fewshot": 0,
                    "baseline_score": 66.40,
                    "evaluated_score": 68.55,
                }
            },
            "retention_anchors": {
                "GPQA-Diamond": {
                    "domain": "graduate_science_reasoning",
                    "num_fewshot": 0,
                    "baseline_score": 58.60,
                    "evaluated_score": 58.10,
                }
            },
        }
        family_md = generate_family_markdown_report(mock_family_item, mock_family_item)
        self.assertIn(
            "E4B Base (%) | E4B Arabic-Tuned (%) | E4B Gain (% pts) | 26B-A4B Base (%) | 26B-A4B Arabic-Tuned (%)",
            family_md,
        )
        self.assertNotIn("Google Official", family_md)

        tasks_acva, primary_acva, eff_acva = resolve_lm_eval_task_specs(
            "arabic_leaderboard_acva_light", num_fewshot=0, limit=200
        )
        self.assertEqual(len(tasks_acva), 5)
        self.assertEqual(primary_acva, "arabic_leaderboard_acva_light")
        self.assertEqual(eff_acva, 40)

        # Verify extract_lm_eval_detailed_metrics parses acc_norm, HardTop1 (acc), GoldNLL, and n_samples
        detailed_payload = {
            "results": {
                "arabicmmlu_all": {
                    "acc,none": 0.610,
                    "acc_norm,none": 0.645,
                }
            },
            "samples": {
                "arabicmmlu_all": [
                    {"target": 1, "resps": [[[-3.2, False]], [[-1.5, True]], [[-4.1, False]]]},
                    {"target": 0, "resps": [[[-2.5, True]], [[-3.8, False]], [[-4.0, False]]]},
                ]
            },
        }
        detailed = extract_lm_eval_detailed_metrics(
            detailed_payload, task_name="arabicmmlu_all", metric_hint="acc_norm"
        )
        self.assertAlmostEqual(detailed["eval_score_pct"], 64.50, places=2)
        self.assertAlmostEqual(detailed["top1_acc_pct"], 61.00, places=2)
        self.assertAlmostEqual(detailed["nll"], 2.0000, places=4)
        self.assertEqual(detailed["n_samples"], 2)

        # Verify finite-sample retention gate allows a 1-item discretization step at N=50
        finite_gate = evaluate_retention_gate(
            anchor_results={
                "MMLU-Pro": {
                    "baseline_score": 62.0,
                    "evaluated_score": 60.0,
                    "max_regression_pct": 0.20,
                    "n_samples": 50,
                },
                "GSM8K": {
                    "baseline_score": 100.0,
                    "evaluated_score": 98.0,
                    "max_regression_pct": 0.20,
                    "n_samples": 50,
                },
            }
        )
        self.assertTrue(finite_gate["gate_passed"])

    def test_10_family_benchmark_plots(self) -> None:
        """Verify all 4 comparison plots (PNG + SVG) generation."""
        eval_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval_config.yaml").read_text(encoding="utf-8")
        )
        latency_profile = build_reference_latency_profile()
        with tempfile.TemporaryDirectory(prefix="gemma4_plots_test_") as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            plots_dir = tmp_path / "plots"

            generated = generate_all_family_plots(
                eval_cfg=eval_cfg,
                latency_profile=latency_profile,
                output_dir=plots_dir,
                dpi=120,
            )
            self.assertEqual(len(generated), 8)
            for fpath in generated:
                p = pathlib.Path(fpath)
                self.assertTrue(p.exists(), f"Missing generated plot: {p}")
                self.assertGreater(p.stat().st_size, 50)

    def test_11_vllm_concurrency_scaling(self) -> None:
        """Verify 1-50 user vLLM single-GPU concurrency telemetry and plot."""
        vllm_profile = build_reference_vllm_concurrency_profile()
        self.assertEqual(vllm_profile["gpu_vram_limit_gb"], 80.0)
        for user_count in ("1", "10", "30", "40", "50"):
            e4b_row = vllm_profile["models"]["e4b"]["concurrency_scaling"][user_count]
            moe_row = vllm_profile["models"]["26b_a4b"]["concurrency_scaling"][user_count]
            self.assertLess(e4b_row["vram_used_gb"], 80.0)
            self.assertLess(moe_row["vram_used_gb"], 80.0)
            self.assertGreater(e4b_row["total_system_tokens_per_sec"], 100.0)
            self.assertGreater(moe_row["total_system_tokens_per_sec"], 100.0)

        md_report = render_vllm_markdown_report(vllm_profile)
        self.assertIn("30 Users", md_report)
        self.assertIn("50 Users", md_report)

        with tempfile.TemporaryDirectory(prefix="gemma4_vllm_sync_test_") as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            plots = generate_vllm_concurrency_plot(vllm_profile, output_dir=tmp_path, dpi=120)
            self.assertEqual(len(plots), 2)
            for p_str in plots:
                self.assertTrue(pathlib.Path(p_str).exists())

    def test_12_v2_masking_dpo_prefix_and_role_parallel_eval(self) -> None:
        """Verify SFT -100 turn masking, 26B DPO prefix alignment, and parallel role checkpoint combining."""
        import scripts.run_benchmarks as rb

        # 1. Verify CLI flags & isolated GPU environment builder
        role_args = parse_bench_args(["--model-key", "e4b", "--only-role", "base"])
        self.assertEqual(role_args.only_role, "base")
        comb_args = parse_bench_args(["--model-key", "both", "--combine-role-reports"])
        self.assertTrue(comb_args.combine_role_reports)
        pre_args = parse_bench_args(["--prewarm-datasets"])
        self.assertTrue(pre_args.prewarm_datasets)

        child_role_cmd = _build_child_eval_cmd(
            role_args, model_key="e4b", only_role="tuned", model_path="outputs/merged_gemma4_e4b_arabic/merged_16bit"
        )
        self.assertIn("--only-role", child_role_cmd)
        self.assertIn("tuned", child_role_cmd)

        env_a = _build_isolated_gpu_env(device="cuda:0", tag="e4b_base")
        env_b = _build_isolated_gpu_env(device="cuda:1", tag="e4b_tuned")
        self.assertNotEqual(env_a["MASTER_PORT"], env_b["MASTER_PORT"])
        self.assertNotEqual(env_a["VLLM_PORT"], env_b["VLLM_PORT"])
        self.assertNotEqual(env_a["TORCHINDUCTOR_CACHE_DIR"], env_b["TORCHINDUCTOR_CACHE_DIR"])

        # 2. Verify combine_role_checkpoints merges live_raw_e4b_base.json + live_raw_e4b_tuned.json
        eval_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval_config.yaml").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="gemma4_role_combine_") as tmp_dir:
            tmp_root = pathlib.Path(tmp_dir)
            reports_dir = tmp_root / "outputs" / "eval_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            base_ar = {k: {"eval_score_pct": 40.0, "n_samples": 100} for k in eval_cfg["arabic_benchmarks"]}
            base_ret = {k: {"eval_score_pct": 55.0, "n_samples": 100} for k in eval_cfg["retention_anchors"]}
            tuned_ar = {k: {"eval_score_pct": 52.5, "n_samples": 100} for k in eval_cfg["arabic_benchmarks"]}
            tuned_ret = {k: {"eval_score_pct": 55.2, "n_samples": 100} for k in eval_cfg["retention_anchors"]}

            _atomic_write_json(
                reports_dir / "live_raw_e4b_base.json",
                {
                    "model_key": "e4b",
                    "role": "base",
                    "base_model_id": "unsloth/gemma-4-E4B-it",
                    "tuned_model_path": "outputs/merged_gemma4_e4b_arabic/merged_16bit",
                    "limit": None,
                    "base": {"arabic": base_ar, "retention": base_ret},
                },
            )
            _atomic_write_json(
                reports_dir / "live_raw_e4b_tuned.json",
                {
                    "model_key": "e4b",
                    "role": "tuned",
                    "base_model_id": "unsloth/gemma-4-E4B-it",
                    "tuned_model_path": "outputs/merged_gemma4_e4b_arabic/merged_16bit",
                    "limit": None,
                    "tuned": {"arabic": tuned_ar, "retention": tuned_ret},
                },
            )
            orig_root = rb.PROJECT_ROOT
            try:
                rb.PROJECT_ROOT = tmp_root
                combined = combine_role_checkpoints(
                    model_key="e4b",
                    base_model_id="unsloth/gemma-4-E4B-it",
                    tuned_model_path="outputs/merged_gemma4_e4b_arabic/merged_16bit",
                    cfg=eval_cfg,
                    limit=None,
                )
                self.assertTrue((reports_dir / "live_raw_e4b.json").exists())
                self.assertEqual(combined["base"]["arabic"]["ArabicMMLU"]["eval_score_pct"], 40.0)
                self.assertEqual(combined["tuned"]["arabic"]["ArabicMMLU"]["eval_score_pct"], 52.5)
            finally:
                rb.PROJECT_ROOT = orig_root

        # 3. Verify real Gemma 4 tokenizer SFT -100 turn masking & 26B DPO prefix alignment if cached locally
        from datasets import Dataset
        from src.train_sft import pretokenize_sft_dataset
        from transformers import AutoTokenizer

        for model_id in ("unsloth/gemma-4-E4B-it", "unsloth/gemma-4-26B-A4B-it"):
            try:
                tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            except OSError:
                continue

            is_moe = "26B" in model_id
            sample_msgs = [
                {"role": "user", "content": "اشرح المادة 74 من نظام العمل.", "reasoning_content": ""},
                {
                    "role": "assistant",
                    "content": "تنص المادة 74 على حالات انتهاء عقد العمل نظاماً.",
                    "reasoning_content": "نستعرض الحالات النظامية في نظام العمل السعودي.",
                },
            ]
            masked = build_tokenized_sft_Turn_masks(
                sample_msgs,
                tokenizer=tok,
                max_seq_length=512,
                enable_thinking=True,
                is_26b_moe=is_moe,
            )
            self.assertIsNotNone(masked)
            assert masked is not None
            self.assertEqual(len(masked["input_ids"]), len(masked["assistant_masks"]))
            self.assertEqual(masked["assistant_masks"][0], 0)
            self.assertGreater(sum(masked["assistant_masks"]), 5)
            self.assertNotIn(3, masked["input_ids"])  # zero <unk>

            pretok_ds = pretokenize_sft_dataset(
                Dataset.from_list([{"messages": sample_msgs, "enable_thinking": True}]),
                tokenizer=tok,
                max_seq_length=512,
                is_26b_moe=is_moe,
            )
            self.assertEqual(len(pretok_ds), 1)
            self.assertNotIn("labels", pretok_ds.column_names)
            self.assertEqual(pretok_ds[0]["assistant_masks"][0], 0)
            self.assertGreater(sum(pretok_ds[0]["assistant_masks"]), 5)

            orig_tpl = tok.chat_template
            try:
                if is_moe:
                    tok.chat_template = align_26b_chat_template_for_dpo(tok.chat_template)
                verify_dpo_prefix_alignment(
                    {
                        "prompt": [{"role": "user", "content": "ما عاصمة السعودية؟"}],
                        "chosen": [{"role": "assistant", "content": "الرياض."}],
                        "rejected": [{"role": "assistant", "content": "جدة."}],
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    tok,
                )
            finally:
                tok.chat_template = orig_tpl

    def test_13_checkpoint_resume_scorers_and_zero_fake_guards(self) -> None:
        """Verify crash/spot-preemption checkpoint recovery, DPO multi-adapter resume, scorer normalizers, and zero-fake guards."""
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from src.train_sft import pretokenize_sft_dataset

        # 1. All 6 YAML configs must enforce frequent rolling checkpoints (save_steps=100, save_total_limit=2)
        for cfg_name in (
            "sft_e4b_a100.yaml",
            "sft_26b_a4b_a100.yaml",
            "dpo_e4b_a100.yaml",
            "dpo_26b_a4b_a100.yaml",
            "sft_config.yaml",
            "dpo_config.yaml",
        ):
            loaded = yaml.safe_load((PROJECT_ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))
            self.assertEqual(loaded["training"]["save_steps"], 100, cfg_name)
            self.assertEqual(loaded["training"]["save_total_limit"], 2, cfg_name)
            self.assertEqual(loaded["training"]["save_strategy"], "steps", cfg_name)

        # 2. Partial root save cleanup + corrupted trailing checkpoint fallback
        with tempfile.TemporaryDirectory(prefix="gemma4_ckpt_guard_") as tmp_dir:
            out_dir = pathlib.Path(tmp_dir)
            # Simulate crash mid-root-save: adapter_config.json written, training_metrics.json missing
            (out_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
            (out_dir / "adapter_model.safetensors").write_bytes(b"X" * 512)

            ckpt_100 = out_dir / "checkpoint-100"
            ckpt_100.mkdir()
            atomic_write_json(ckpt_100 / "trainer_state.json", {"global_step": 100})
            atomic_write_json(ckpt_100 / "adapter_config.json", {"peft_type": "LORA"})
            (ckpt_100 / "adapter_model.safetensors").write_bytes(b"W" * 512)
            (ckpt_100 / "optimizer.pt").write_bytes(b"O" * 128)

            # Simulate interrupted checkpoint-200 (missing trainer_state.json)
            ckpt_200 = out_dir / "checkpoint-200"
            ckpt_200.mkdir()
            atomic_write_json(ckpt_200 / "adapter_config.json", {"peft_type": "LORA"})
            (ckpt_200 / "adapter_model.safetensors").write_bytes(b"W" * 512)

            # Simulate interrupted DPO checkpoint-300 with half-written ref/ subdirectory
            ckpt_300 = out_dir / "checkpoint-300"
            (ckpt_300 / "ref").mkdir(parents=True)
            atomic_write_json(ckpt_300 / "trainer_state.json", {"global_step": 300})
            atomic_write_json(ckpt_300 / "adapter_config.json", {"peft_type": "LORA"})
            (ckpt_300 / "adapter_model.safetensors").write_bytes(b"W" * 512)
            (ckpt_300 / "optimizer.pt").write_bytes(b"O" * 128)
            atomic_write_json(ckpt_300 / "ref" / "adapter_config.json", {"peft_type": "LORA"})

            self.assertFalse(
                is_adapter_stage_complete(out_dir, "training_metrics.json", cleanup_partial_root=True)
            )
            self.assertFalse((out_dir / "adapter_config.json").exists())

            resumed = find_latest_valid_checkpoint(out_dir, require_optimizer_state=True, cleanup_corrupt=True)
            self.assertEqual(resumed, str(ckpt_100))
            self.assertFalse(ckpt_200.exists())
            self.assertFalse(ckpt_300.exists())

            # Once root save + metrics complete, stage is recognized as complete
            atomic_write_json(out_dir / "adapter_config.json", {"peft_type": "LORA"})
            (out_dir / "adapter_model.safetensors").write_bytes(b"W" * 512)
            atomic_write_json(out_dir / "training_metrics.json", {"train_loss": 0.85})
            self.assertTrue(
                is_adapter_stage_complete(out_dir, "training_metrics.json", cleanup_partial_root=True)
            )

        # 3. Real PEFT multi-adapter DPO checkpoint resume hook verification ("default" + frozen "ref")
        from transformers import PretrainedConfig

        class _TinyBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = PretrainedConfig(model_type="gemma4_text", use_cache=False)
                self.q_proj = nn.Linear(16, 16, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.q_proj(x)

        with tempfile.TemporaryDirectory(prefix="gemma4_dpo_resume_") as tmp_dpo_dir:
            ckpt_dir = pathlib.Path(tmp_dpo_dir) / "checkpoint-100"
            ckpt_dir.mkdir(parents=True)

            base_net = _TinyBackbone()
            lora_cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj"], bias="none")
            peft_net = get_peft_model(base_net, lora_cfg)
            peft_net.add_adapter("ref", lora_cfg)

            # Set distinct non-zero weights for step-0 "ref" vs trained step-100 "default"
            with torch.no_grad():
                for n, p in peft_net.named_parameters():
                    if ".ref." in n:
                        p.fill_(0.25)
                    elif ".default." in n:
                        p.fill_(0.75)

            peft_net.save_pretrained(str(ckpt_dir), selected_adapters=["default"])
            peft_net.save_pretrained(str(ckpt_dir), selected_adapters=["ref"])
            self.assertTrue((ckpt_dir / "adapter_config.json").is_file())
            self.assertTrue((ckpt_dir / "ref" / "adapter_config.json").is_file())

            # Freshly initialized resume model where Trainer._load_from_checkpoint only loaded "ref"
            fresh_peft = get_peft_model(_TinyBackbone(), lora_cfg)
            fresh_peft.add_adapter("ref", lora_cfg)
            with torch.no_grad():
                for n, p in fresh_peft.named_parameters():
                    if ".ref." in n:
                        p.fill_(0.25)
                    elif ".default." in n:
                        p.zero_()

            class _DummyDPOTrainer:
                def __init__(self, m: nn.Module) -> None:
                    self.model = m

                def _load_from_checkpoint(
                    self, resume_from_checkpoint: str, model: Any = None
                ) -> None:
                    # Simulate upstream Trainer bug: skips loading "default" when subfolder "ref" exists
                    return None

            dummy_trainer = _DummyDPOTrainer(fresh_peft)
            install_dpo_default_adapter_resume_hook(dummy_trainer, freeze_moe_router=True)
            dummy_trainer._load_from_checkpoint(str(ckpt_dir), None)

            self.assertEqual(fresh_peft.active_adapter, "default")
            for n, p in fresh_peft.named_parameters():
                if ".default." in n and "lora_" in n:
                    self.assertTrue(p.requires_grad, f"{n} must remain trainable after resume")
                    self.assertTrue(
                        torch.allclose(p, torch.full_like(p, 0.75)),
                        f"{n} was not restored from checkpoint-100 root adapter!",
                    )
                elif ".ref." in n and "lora_" in n:
                    self.assertFalse(p.requires_grad, f"{n} must stay frozen after resume")
                    self.assertTrue(torch.allclose(p, torch.full_like(p, 0.25)))

        # 4. Benchmark scorer normalizers (AIME trailing \] fix & LCB trailing stdout fence protection)
        aime_raw = (
            "<|channel>thought\nCompute 7 * 10 = 70.\n<channel|>"
            "Therefore, the final answer is:\n\\[\n\\boxed{70}\n\\]<turn|>"
        )
        aime_norm = normalize_inspect_completion_for_scoring(aime_raw, is_aime=True)
        self.assertEqual(aime_norm, "Therefore, the final answer is:\n$$\n\\boxed{70}\n$$")
        # Confirm trailing $$ line is skipped cleanly when scanning last non-$$ line
        non_dollar_lines = [ln.strip() for ln in aime_norm.splitlines() if ln.strip() and ln.strip() != "$$"]
        self.assertEqual(non_dollar_lines[-1], "\\boxed{70}")

        lcb_with_io_fence = (
            "<|channel>thought\nRead integer n and print n*2.\n<channel|>"
            "Here is the solution:\n```python\nimport sys\n\ndef solve():\n    n = int(sys.stdin.read().strip())\n    print(n * 2)\n\nif __name__ == '__main__':\n    solve()\n```\n"
            "Example output:\n```text\nInput:\n5\nOutput:\n10\n```"
        )
        lcb_norm = normalize_lcb_code_completion(lcb_with_io_fence)
        self.assertTrue(lcb_norm.startswith("```python\n"))
        self.assertIn("def solve():", lcb_norm)
        self.assertNotIn("Output:\n10", lcb_norm)

        lcb_self_corrected = (
            "Draft:\n```python\ndef solve():\n    return 0\n```\n"
            "Fixed solution:\n```python\ndef solve():\n    return 42\n```"
        )
        lcb_self_norm = normalize_lcb_code_completion(lcb_self_corrected)
        self.assertIn("return 42", lcb_self_norm)

        # 5. Over-length SFT thinking trace must drop cleanly (None) rather than slicing mid-thought
        class _CharTok:
            def apply_chat_template(self, msgs: Any, tokenize: bool = False, **kw: Any) -> str:
                return format_gemma4_chat(msgs, enable_thinking=kw.get("enable_thinking", False))

            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                return [ord(ch) % 200 + 10 for ch in text]

        long_think_msgs = [
            {"role": "user", "content": "Solve", "reasoning_content": ""},
            {
                "role": "assistant",
                "content": "42",
                "reasoning_content": "step " * 200,
            },
        ]
        self.assertIsNone(
            build_tokenized_sft_Turn_masks(
                long_think_msgs,
                tokenizer=_CharTok(),
                max_seq_length=64,
                enable_thinking=True,
                is_26b_moe=False,
            )
        )

        # Verify SFT pre-tokenized Arrow disk cache reloads instantaneously on resume
        short_msgs = [
            {"role": "user", "content": "Hi", "reasoning_content": ""},
            {"role": "assistant", "content": "Hello world answer!", "reasoning_content": ""},
        ]
        with tempfile.TemporaryDirectory(prefix="gemma4_pretok_cache_") as tmp_cache:
            cdir = pathlib.Path(tmp_cache) / "dense"
            ds1 = pretokenize_sft_dataset(
                Dataset.from_list([{"messages": short_msgs, "enable_thinking": False}]),
                tokenizer=_CharTok(),
                max_seq_length=256,
                is_26b_moe=False,
                cache_dir=cdir,
            )
            self.assertEqual(len(ds1), 1)
            # Second call with matching row count & broken tokenizer=None proves Arrow cache hit
            ds2 = pretokenize_sft_dataset(
                Dataset.from_list([{"messages": []}]),
                tokenizer=None,
                max_seq_length=256,
                is_26b_moe=False,
                cache_dir=cdir,
            )
            self.assertEqual(len(ds2), 1)

        # 6. Zero-fake-number hard guard in generate_all_family_plots(require_real_data=True)
        #    and monotonic vLLM dynamic KV-cache scaling across 1 -> 50 concurrent users
        eval_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval_config.yaml").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="gemma4_real_guard_") as tmp_plots:
            import scripts.benchmark_latency_and_plots as blp

            orig_blp_root = blp.PROJECT_ROOT
            try:
                blp.PROJECT_ROOT = pathlib.Path(tmp_plots)
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    generate_all_family_plots(
                        eval_cfg=eval_cfg,
                        latency_profile=build_reference_latency_profile(),
                        output_dir=pathlib.Path(tmp_plots) / "plots",
                        dpi=100,
                        require_real_data=True,
                    )
            finally:
                blp.PROJECT_ROOT = orig_blp_root

        kv_1 = _compute_active_kv_and_runtime_gb("nonexistent_e4b", [768] * 1, is_moe=False)
        kv_10 = _compute_active_kv_and_runtime_gb("nonexistent_e4b", [768] * 10, is_moe=False)
        kv_30 = _compute_active_kv_and_runtime_gb("nonexistent_e4b", [768] * 30, is_moe=False)
        kv_50 = _compute_active_kv_and_runtime_gb("nonexistent_e4b", [768] * 50, is_moe=False)
        self.assertLess(kv_1, kv_10)
        self.assertLess(kv_10, kv_30)
        self.assertLess(kv_30, kv_50)
        self.assertLess(9.2 + kv_50, 80.0)

        # Verify 26B-A4B MoE KV geometry (num_global_key_value_heads=2 + attention_k_eq_v=True)
        moe_kv_1 = _compute_active_kv_and_runtime_gb("nonexistent_26b", [768] * 1, is_moe=True)
        moe_kv_50 = _compute_active_kv_and_runtime_gb("nonexistent_26b", [768] * 50, is_moe=True)
        self.assertLess(moe_kv_1, moe_kv_50)
        self.assertLess(47.7 + moe_kv_50, 80.0)

        # 7. Verify MMLU-Pro dynamic max_gen_toks budget & lm_eval unclosed-thought guard
        from scripts.run_benchmarks import (
            _get_lm_eval_gen_kwargs,
            _install_lm_eval_unclosed_thought_guard,
        )
        import lm_eval.models.utils as lm_utils

        kw_32k = _get_lm_eval_gen_kwargs("mmlu_pro", use_vllm=True, max_model_len=32768)
        kw_24k = _get_lm_eval_gen_kwargs("mmlu_pro", use_vllm=True, max_model_len=24576)
        assert kw_32k is not None and kw_24k is not None
        self.assertEqual(kw_32k["max_gen_toks"], 24576)
        self.assertEqual(kw_24k["max_gen_toks"], 16384)
        self.assertFalse(kw_32k["skip_special_tokens"])

        _install_lm_eval_unclosed_thought_guard()
        self.assertEqual(
            lm_utils.postprocess_generated_text(
                "<|channel>thought\nChecking (A) and (B)...",
                ["<turn|>"],
                "<channel|>",
            ),
            "",
        )
        self.assertEqual(
            lm_utils.postprocess_generated_text(
                "<|channel>thought\nDone.\n<channel|>The answer is (D)<turn|>extra",
                ["<turn|>"],
                "<channel|>",
            ),
            "The answer is (D)",
        )

        # 8. Verify single-point training history renders cleanly without crashing plot generator
        from scripts.benchmark_latency_and_plots import _load_training_curve_series

        with tempfile.TemporaryDirectory(prefix="gemma4_single_pt_") as tmp_hist:
            st_path = pathlib.Path(tmp_hist) / "trainer_state.json"
            atomic_write_json(st_path, {"log_history": [{"step": 10, "loss": 1.25}]})
            series = _load_training_curve_series(pathlib.Path(tmp_hist) / "training_metrics.json", ("loss",))
            self.assertEqual(series, ([0.0, 10.0], [1.25, 1.25]))

        # 9. Compile-check every inline "${PYTHON_BIN}" -c "..." snippet in run_dual_a100_pipeline.sh
        import re

        sh_text = (PROJECT_ROOT / "scripts" / "run_dual_a100_pipeline.sh").read_text(encoding="utf-8")
        py_snippets = re.findall(r'"\$\{PYTHON_BIN\}"\s+-c\s+"(.*?)"', sh_text, flags=re.S)
        self.assertGreaterEqual(len(py_snippets), 10)
        for idx, raw_snip in enumerate(py_snippets):
            unescaped = raw_snip.replace('\\"', '"').replace("\\$", "$")
            compile(unescaped, f"run_dual_a100_pipeline.sh:snippet_{idx}", "exec")

    def test_14_v3_end_to_end_fixes(self) -> None:
        """Verify all V3 root-cause fixes across 26B masking, DPO ref precompute, LightEval spacing, GPU env, and preflight gate."""
        import os
        from typing import Optional
        from transformers import TrainerControl, TrainerState, TrainingArguments
        from scripts.run_benchmarks import (
            evaluate_preflight_gate_metrics,
            format_lighteval_arabic_chat_prompt,
            normalize_lighteval_mcq_continuation_space,
        )
        from src.model_loader import build_device_map
        from src.dpo_formatter import (
            _has_complete_terminal_ending,
            normalize_arabic_reasoning_thinking_dpo_row,
            normalize_distilabel_math_thinking_dpo_row,
        )
        from src.train_dpo import DPORefSanityCallback, preserve_stage1_adapter_for_ref_precompute

        # 1. Verify 26B-A4B EMPTY_THOUGHT_CHANNEL is strictly masked out (assistant_masks = 0)
        class _ExactCharTok:
            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                return [ord(ch) for ch in text]

        msgs_26b_direct = [
            {"role": "user", "content": "سؤال؟", "reasoning_content": ""},
            {"role": "assistant", "content": "الجواب.", "reasoning_content": ""},
        ]
        tok_out = build_tokenized_sft_Turn_masks(
            msgs_26b_direct,
            tokenizer=_ExactCharTok(),
            max_seq_length=512,
            enable_thinking=False,
            is_26b_moe=True,
        )
        self.assertIsNotNone(tok_out)
        assert tok_out is not None
        supervised_text = "".join(
            chr(tid) for tid, m in zip(tok_out["input_ids"], tok_out["assistant_masks"]) if m == 1
        )
        unsupervised_text = "".join(
            chr(tid) for tid, m in zip(tok_out["input_ids"], tok_out["assistant_masks"]) if m == 0
        )
        self.assertEqual(supervised_text, "الجواب.<turn|>\n")
        self.assertIn(EMPTY_THINKING_BLOCK, unsupervised_text)
        self.assertNotIn(EMPTY_THINKING_BLOCK, supervised_text)

        # 2. Verify LightEval VLLMModel/TransformersModel SentencePiece leading-space & chat-prefill formatting
        self.assertEqual(
            normalize_lighteval_mcq_continuation_space("الإجابة:", "أ"),
            " أ",
        )
        self.assertEqual(
            normalize_lighteval_mcq_continuation_space("Answer:", "B"),
            " B",
        )
        self.assertEqual(
            normalize_lighteval_mcq_continuation_space("الإجابة: ", "أ"),
            "أ",
        )
        self.assertEqual(
            normalize_lighteval_mcq_continuation_space("الإجابة:\n", "أ"),
            "أ",
        )
        self.assertEqual(
            normalize_lighteval_mcq_continuation_space("<|turn>model\n<|channel>thought\n<channel|>", ["أ", "ب"]),
            ["أ", "ب"],
        )
        orig_choices = ["أ", "ب", "ج", "د"]
        spaced_choices = normalize_lighteval_mcq_continuation_space("الإجابة:", orig_choices)
        self.assertEqual(spaced_choices, [" أ", " ب", " ج", " د"])
        self.assertEqual(orig_choices, ["أ", "ب", "ج", "د"])
        self.assertEqual(
            normalize_lighteval_mcq_continuation_space("الإجابة:", spaced_choices),
            [" أ", " ب", " ج", " د"],
        )
        self.assertEqual(
            normalize_lighteval_mcq_continuation_space("الإجابة:", ("صح", "خطأ")),
            (" صح", " خطأ"),
        )

        class _MockChatTok:
            def __init__(self, is_26b: bool = False) -> None:
                self.is_26b = is_26b
                self.chat_template = "gemma4"

            def apply_chat_template(
                self,
                messages: List[Dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                out = "<bos>"
                for m in messages:
                    role = "model" if m["role"] == "assistant" else m["role"]
                    out += f"<|turn>{role}\n{m['content']}<turn|>\n"
                if add_generation_prompt:
                    out += "<|turn>model\n"
                    if self.is_26b and not enable_thinking:
                        out += "<|channel>thought\n<channel|>"
                return out

        class _MockDoc:
            def __init__(
                self,
                query: str,
                choices: List[str],
                gold_index: int = 0,
                instruction: Optional[str] = None,
                fewshot_samples: Optional[List[Any]] = None,
            ) -> None:
                self.query = query
                self.choices = choices
                self.gold_index = gold_index
                self.instruction = instruction
                self.fewshot_samples = fewshot_samples or []

            def get_golds(self) -> List[str]:
                return [self.choices[self.gold_index]]

        instr_mmlu = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
        doc_mmlu = _MockDoc(
            query=f"{instr_mmlu}ما عاصمة المملكة العربية السعودية؟\nأ. الرياض\nب. جدة\nالإجابة:",
            choices=["أ", "ب"],
            gold_index=0,
            instruction=instr_mmlu,
        )
        e4b_prompt = format_lighteval_arabic_chat_prompt(doc_mmlu, _MockChatTok(is_26b=False))
        self.assertTrue(e4b_prompt.startswith("<bos><|turn>user\n"))
        self.assertTrue(e4b_prompt.endswith("<turn|>\n<|turn>model\nالإجابة:"))
        self.assertEqual(e4b_prompt.count(instr_mmlu.strip()), 1)

        moe_prompt = format_lighteval_arabic_chat_prompt(doc_mmlu, _MockChatTok(is_26b=True))
        self.assertTrue(moe_prompt.endswith("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>الإجابة:"))

        doc_boolq = _MockDoc(
            query="بناء على المقطع التالي، أجب عن السؤال ب نعم أو لا\n    الإجابة:\n    ",
            choices=["نعم", "لا"],
            gold_index=0,
            instruction="بناء على المقطع التالي، أجب عن السؤال ب نعم أو لا",
        )
        boolq_prompt = format_lighteval_arabic_chat_prompt(doc_boolq, _MockChatTok(is_26b=False))
        self.assertTrue(boolq_prompt.endswith("<turn|>\n<|turn>model\nالإجابة:"))

        fs_acva = _MockDoc(query="السؤال: تقع الرياض في نجد.\nالإجابة:", choices=["صح", "خطأ"], gold_index=0)
        doc_acva = _MockDoc(
            query="السؤال: تقع مكة على ساحل الخليج العربي.\nالإجابة:",
            choices=["صح", "خطأ"],
            gold_index=1,
            fewshot_samples=[fs_acva],
        )
        acva_prompt = format_lighteval_arabic_chat_prompt(doc_acva, _MockChatTok(is_26b=False))
        self.assertIn("<|turn>model\nالإجابة: صح<turn|>", acva_prompt)
        self.assertTrue(acva_prompt.endswith("<turn|>\n<|turn>model\nالإجابة:"))

        if torch.cuda.is_available():
            dmap_offload = build_device_map(cuda_device_index=0, offload_ple_to_cpu=True, offload_multimodal_towers_to_cpu=True)
            self.assertNotIn("", dmap_offload)
            self.assertEqual(dmap_offload["model.language_model.embed_tokens_per_layer"], "cpu")
            self.assertEqual(dmap_offload["model.audio_tower"], "cpu")
            self.assertEqual(dmap_offload["model.language_model.layers"], 0)
            self.assertEqual(len(dmap_offload), 12)
            dmap_a100 = build_device_map(cuda_device_index=1, offload_ple_to_cpu=False, offload_multimodal_towers_to_cpu=False)
            self.assertEqual(dmap_a100, {"": 1})

        # Compile-verify dynamically generated subprocess runner scripts in run_benchmarks.py
        import unittest.mock as _umock
        from scripts.run_benchmarks import run_lighteval_lcb_subprocess, run_lighteval_tasks_subprocess

        captured_scripts: List[str] = []

        def _fake_subproc_run(cmd: List[str], **kwargs: Any) -> Any:
            captured_scripts.append(cmd[2])
            compile(cmd[2], "lighteval_subprocess_runner", "exec")
            import re as _re

            m = _re.search(r"pathlib\.Path\((['\"].+?\.json['\"])\)\.write_text", cmd[2])
            if m:
                out_p = pathlib.Path(eval(m.group(1)))
                out_p.write_text(
                    json.dumps(
                        {
                            "results": {
                                "community|arabic_mmlu:All|0": {"acc_norm": 0.61},
                                "community|lcb:codegeneration_v6|0": {"codegen_pass@1:1": 0.54},
                            },
                            "doc_counts": {"community|arabic_mmlu:All|0": 50, "community|lcb:codegeneration_v6|0": 10},
                        }
                    ),
                    encoding="utf-8",
                )

            class _ProcRes:
                returncode = 0

            return _ProcRes()

        with _umock.patch("scripts.run_benchmarks.subprocess.run", side_effect=_fake_subproc_run):
            ar_mock_res = run_lighteval_tasks_subprocess(
                model_path="unsloth/gemma-4-E4B-it",
                suite_names=["ArabicMMLU"],
                device="cuda:0",
                limit=50,
                use_vllm=True,
                tag="compile_check_ar",
            )
            lcb_mock_res = run_lighteval_lcb_subprocess(
                model_path="unsloth/gemma-4-E4B-it",
                suite_specs={"LiveCodeBench-v6": {"lighteval_task": "lcb:codegeneration_v6|0", "num_samples": 1}},
                device="cuda:0",
                limit=10,
                use_vllm=True,
                tag="compile_check_lcb",
            )
        self.assertEqual(len(captured_scripts), 2)
        self.assertAlmostEqual(ar_mock_res["ArabicMMLU"]["eval_score_pct"], 61.0, places=2)
        self.assertAlmostEqual(lcb_mock_res["LiveCodeBench-v6"]["eval_score_pct"], 54.0, places=2)

        # 3. Verify strict DPO terminal-ending filter & thinking-mode CoT DPO normalizers
        self.assertFalse(_has_complete_terminal_ending("تبدأ الإجراءات بالذهاب إلى مكتب"))
        self.assertTrue(_has_complete_terminal_ending("تبدأ الإجراءات بالذهاب إلى المكتب."))
        self.assertIsNone(
            normalize_arcee_arabic_dpo_row(
                {
                    "prompt": "ما هي خطوات التسجيل في المنصة الوطنية الموحدة؟",
                    "chosen": "تبدأ خطوات التسجيل بالدخول إلى البوابة الإلكترونية ثم إدخال رقم الهوية ولكن النص مقطوع في المنتصف بدون علامة ترقيم",
                    "rejected": "لا أعرف كيفية التسجيل في المنصة.",
                },
                strict_hygiene=True,
            )
        )
        ar_cot = normalize_arabic_reasoning_thinking_dpo_row(
            {
                "prompt": "احسب محيط مستطيل طوله 8 سم وعرضه 5 سم مع الشرح.",
                "chosen": (
                    "الخطوة الأولى: قانون محيط المستطيل يساوي ضعف مجموع الطول والعرض أي 2 * (الطول + العرض).\n\n"
                    "الخطوة الثانية: نجمع 8 + 5 فنحصل على 13 ثم نضرب الناتج في 2.\n\n"
                    "إذن محيط المستطيل يساوي 26 سنتيمتراً."
                ),
                "rejected": (
                    "الخطوة الأولى: لحساب المحيط نضرب الطول في العرض مباشرة دون جمع الأضلاع.\n\n"
                    "الخطوة الثانية: نضرب 8 في 5 فنحصل على القيمة العددية المطلوبة.\n\n"
                    "إذن المحيط يساوي 40 سنتيمتراً."
                ),
            }
        )
        self.assertIsNotNone(ar_cot)
        assert ar_cot is not None
        self.assertTrue(ar_cot["enable_thinking"])
        self.assertIn(CHANNEL_THOUGHT_PREFIX, ar_cot["chosen"])
        self.assertIn(CHANNEL_THOUGHT_PREFIX, ar_cot["rejected"])

        en_cot = normalize_distilabel_math_thinking_dpo_row(
            {
                "instruction": "Find the derivative of f(x) = x^3 + 4x step by step.",
                "chosen_rating": 9.0,
                "rejected_rating": 6.0,
                "chosen_response": (
                    "Step 1: Apply the power rule d/dx(x^n) = n*x^(n-1) to each term independently.\n\n"
                    "Step 2: Differentiating x^3 gives 3x^2, and differentiating 4x gives 4.\n\n"
                    "Therefore, f'(x) = 3x^2 + 4."
                ),
                "rejected_response": (
                    "Step 1: Multiply each exponent by the coefficient without reducing the exponent.\n\n"
                    "Step 2: For x^3 we write 3x^3 and for 4x we write 4x.\n\n"
                    "Therefore, f'(x) = 3x^3 + 4x."
                ),
            }
        )
        self.assertIsNotNone(en_cot)
        assert en_cot is not None
        self.assertTrue(en_cot["enable_thinking"])

        # 4. Verify _build_isolated_gpu_env overrides multi-GPU CUDA_VISIBLE_DEVICES="0,1"
        old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            env0 = _build_isolated_gpu_env("0", "e4b_base")
            env1 = _build_isolated_gpu_env("1", "e4b_tuned")
            self.assertEqual(env0["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(env1["CUDA_VISIBLE_DEVICES"], "1")
        finally:
            if old_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd

        # 5. Verify DPO precompute guard & DPORefSanityCallback fail-fast behavior
        class _MockPeftModel:
            def __init__(self) -> None:
                self.disabled_calls = 0
                self.added_adapters: List[str] = []

            def disable_adapter(self):
                import contextlib

                @contextlib.contextmanager
                def _cm():
                    self.disabled_calls += 1
                    yield

                return _cm()

            def add_adapter(self, name: str, cfg: Any) -> None:
                self.added_adapters.append(name)

            def get_parameter(self, name: str) -> torch.Tensor:
                if ".ref." in name:
                    raise AttributeError(name)
                return torch.ones(2)

        mock_pm = _MockPeftModel()
        tbox: Dict[str, Any] = {}
        with preserve_stage1_adapter_for_ref_precompute(mock_pm, tbox):
            mock_pm.add_adapter("ref", {})
            _ = mock_pm.get_parameter("layer.lora_A.ref.weight")
            with mock_pm.disable_adapter():
                pass
        self.assertEqual(mock_pm.added_adapters, [])
        self.assertEqual(mock_pm.disabled_calls, 0)

        bad_cb = DPORefSanityCallback()
        with self.assertRaises(RuntimeError):
            bad_cb.on_log(
                TrainingArguments(output_dir="outputs/_tmp_cb", logging_steps=10),
                TrainerState(global_step=10),
                TrainerControl(),
                logs={"loss": 1.53, "grad_norm": 176.3},
            )

        # 6. Verify Pre-Flight Gate decision matrix (passed vs conservative lambda retry vs hard abort)
        pf_ok = evaluate_preflight_gate_metrics(
            {"ArabicMMLU": 52.0, "AlGhafa": 49.0},
            {"ArabicMMLU": 56.5, "AlGhafa": 53.5},
            69.5,
            69.0,
        )
        self.assertTrue(pf_ok["gate_passed"])
        self.assertFalse(pf_ok["needs_conservative_lambda_retry"])

        pf_fb = evaluate_preflight_gate_metrics(
            {"ArabicMMLU": 52.0, "AlGhafa": 49.0},
            {"ArabicMMLU": 56.5, "AlGhafa": 53.5},
            69.5,
            63.0,
        )
        self.assertFalse(pf_fb["gate_passed"])
        self.assertTrue(pf_fb["needs_conservative_lambda_retry"])

        pf_ab = evaluate_preflight_gate_metrics(
            {"ArabicMMLU": 52.0, "AlGhafa": 49.0},
            {"ArabicMMLU": 50.0, "AlGhafa": 47.0},
            69.5,
            69.2,
        )
        self.assertFalse(pf_ab["gate_passed"])
        self.assertFalse(pf_ab["needs_conservative_lambda_retry"])

    def test_29_moe_dpo_alignment_and_merge_invariants(self) -> None:
        """Verify 26B-A4B MoE DPO kernel flags, batch=1 invariance, Step-0 alignment, LoRA layout, and merge checks."""
        import math
        import os
        from datasets import Dataset
        from transformers import TrainerControl, TrainerState, TrainingArguments
        from src.merge_and_export import scale_and_trim_peft_lora_weights
        from src.model_loader import (
            verify_moe_lora_integrity,
            verify_saved_moe_adapter_config,
        )
        from src.train_dpo import (
            DPORefSanityCallback,
            _maybe_inject_gemma4_mm_token_type_ids,
            build_dpo_training_arguments,
            preserve_stage1_adapter_for_ref_precompute,
            salt_dpo_dataset_fingerprint,
            verify_step0_dpo_ref_alignment,
        )

        # 1. Verify 26B-A4B and E4B DPO YAML invariants + logging_first_step + shell kernel flags
        dpo_26b_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "dpo_26b_a4b_a100.yaml").read_text(encoding="utf-8")
        )
        dpo_e4b_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "dpo_e4b_a100.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(dpo_26b_cfg["vram_optimization"]["precompute_ref_batch_size"], 1)
        self.assertEqual(dpo_26b_cfg["training"]["per_device_train_batch_size"], 1)
        self.assertEqual(dpo_26b_cfg["training"]["per_device_eval_batch_size"], 1)
        self.assertEqual(dpo_26b_cfg["training"]["gradient_accumulation_steps"], 16)
        self.assertAlmostEqual(float(dpo_26b_cfg["training"]["learning_rate"]), 5.0e-7, places=12)
        self.assertAlmostEqual(float(dpo_26b_cfg["training"]["warmup_ratio"]), 0.15, places=6)
        self.assertEqual(dpo_e4b_cfg["vram_optimization"]["precompute_ref_batch_size"], 2)
        self.assertEqual(dpo_e4b_cfg["training"]["per_device_train_batch_size"], 2)

        dpo_args = build_dpo_training_arguments(dpo_26b_cfg, output_dir="outputs/_tmp_dpo_args")
        self.assertTrue(bool(getattr(dpo_args, "logging_first_step", False)))
        self.assertEqual(os.environ.get("UNSLOTH_BLOCK_COMPILE"), "0")
        self.assertEqual(os.environ.get("UNSLOTH_MOE_LORA_FOLD"), "0")
        self.assertEqual(os.environ.get("UNSLOTH_MOE_GC_KEEP_BASE"), "0")
        self.assertEqual(os.environ.get("UNSLOTH_MOE_ROUTED_KERNEL"), "0")
        self.assertEqual(os.environ.get("UNSLOTH_HD512_ATTENTION"), "0")

        pipeline_sh = (PROJECT_ROOT / "scripts" / "run_dual_a100_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn('export UNSLOTH_BLOCK_COMPILE="0"', pipeline_sh)
        self.assertIn('export UNSLOTH_MOE_LORA_FOLD="0"', pipeline_sh)
        self.assertIn('export UNSLOTH_MOE_GC_KEEP_BASE="0"', pipeline_sh)
        self.assertIn('export UNSLOTH_MOE_ROUTED_KERNEL="0"', pipeline_sh)
        self.assertIn('export UNSLOTH_HD512_ATTENTION="0"', pipeline_sh)
        self.assertIn("unsloth_compiled_cache_gpu1", pipeline_sh)

        # 2. Verify DPORefSanityCallback allows Gemma 4 pre-clip grad_norm (`transformers#45676`) while catching
        #    Step-1 reference misalignment, mid-run runaway divergence (`loss > 3.5`), dead grads, & NaN/Inf
        cb_bad_step1 = DPORefSanityCallback(max_initial_loss=0.98, max_initial_grad_norm=100_000.0)
        t_args = TrainingArguments(output_dir="outputs/_tmp_cb2", logging_steps=10)
        with self.assertRaises(RuntimeError):
            cb_bad_step1.on_log(t_args, TrainerState(global_step=1), TrainerControl(), logs={"loss": 1.15, "grad_norm": 42.0})

        cb = DPORefSanityCallback(max_initial_loss=0.98, max_initial_grad_norm=100_000.0)
        # Exact Vast.ai Step-1 telemetry (`loss=0.6931, grad_norm=575.1310`) + Gemma 4 CoT spike (`grad_norm=8500`) + hard batch (`1.12`) must PASS!
        cb.on_log(t_args, TrainerState(global_step=1), TrainerControl(), logs={"loss": 0.6931, "grad_norm": 575.1310})
        self.assertTrue(cb.passed_initial_gate)
        cb.on_log(t_args, TrainerState(global_step=10), TrainerControl(), logs={"loss": 0.6420, "grad_norm": 8500.0})
        cb.on_log(t_args, TrainerState(global_step=20), TrainerControl(), logs={"loss": 1.1200, "grad_norm": 4200.0})
        with self.assertRaises(RuntimeError):
            # Must catch Failure-#1 style runaway KL divergence (`loss=4.2813 > 3.50`) at any step!
            cb.on_log(t_args, TrainerState(global_step=10), TrainerControl(), logs={"loss": 4.2813, "grad_norm": 42.0})
        with self.assertRaises(RuntimeError):
            # Must catch dead zero gradients at step >= 1
            cb.on_log(t_args, TrainerState(global_step=1), TrainerControl(), logs={"loss": 0.6931, "grad_norm": 0.0})
        with self.assertRaises(RuntimeError):
            # Must catch catastrophic pre-Inf overflow (> 100,000)
            cb.on_log(t_args, TrainerState(global_step=1), TrainerControl(), logs={"loss": 0.6931, "grad_norm": 250_000.0})
        with self.assertRaises(RuntimeError):
            # Must catch NaN/Inf at late steps (e.g. step 200)
            cb.on_log(t_args, TrainerState(global_step=200), TrainerControl(), logs={"loss": float("nan"), "grad_norm": 1.0})

        # 3. Verify salt_dpo_dataset_fingerprint changes fingerprint across adapters/batch sizes & persists cache_dir
        base_ds = Dataset.from_list([{"prompt": "p", "chosen": "c", "rejected": "r"}])
        with tempfile.TemporaryDirectory() as tmpdir:
            ad1 = pathlib.Path(tmpdir) / "sft1"
            ad2 = pathlib.Path(tmpdir) / "sft2"
            cdir = pathlib.Path(tmpdir) / "ref_cache"
            ad1.mkdir()
            ad2.mkdir()
            (ad1 / "adapter_model.safetensors").write_bytes(b"adapter_v1")
            (ad2 / "adapter_model.safetensors").write_bytes(b"adapter_v2_longer")
            ds_fp1 = salt_dpo_dataset_fingerprint(base_ds, PRIMARY_MODEL_ID_26B, str(ad1), 1, True)
            ds_fp2 = salt_dpo_dataset_fingerprint(base_ds, PRIMARY_MODEL_ID_26B, str(ad2), 1, True)
            ds_fp3 = salt_dpo_dataset_fingerprint(base_ds, PRIMARY_MODEL_ID_26B, str(ad1), 4, True)
            ds_disk = salt_dpo_dataset_fingerprint(
                base_ds, PRIMARY_MODEL_ID_26B, str(ad1), 1, True, cache_dir=cdir
            )
            self.assertNotEqual(ds_fp1._fingerprint, ds_fp2._fingerprint)
            self.assertNotEqual(ds_fp1._fingerprint, ds_fp3._fingerprint)
            self.assertTrue(bool(ds_disk.cache_files))

        # 4. Verify verify_moe_lora_integrity & verify_saved_moe_adapter_config
        class _Mock3DLoRAWeight(nn.Module):
            def __init__(self, init_val: float) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.full((4, 2, 8), init_val))

        class _MockExpertParamWrapper(nn.Module):
            def __init__(self, param_name: str, b_val: float) -> None:
                super().__init__()
                self.parameter_name = param_name
                self.lora_A = nn.ModuleDict({"default": _Mock3DLoRAWeight(0.1)})
                self.lora_B = nn.ModuleDict(
                    {
                        "default": _Mock3DLoRAWeight(b_val),
                        "ref": _Mock3DLoRAWeight(b_val),
                    }
                )
                self.lora_B["ref"].weight.requires_grad_(False)

        class _MockMoEModel(nn.Module):
            def __init__(self, b_val: float) -> None:
                super().__init__()
                self.router = nn.Linear(8, 4, bias=False)
                self.router.weight.requires_grad_(False)
                self.experts_gate_up = _MockExpertParamWrapper("gate_up_proj", b_val)
                self.experts_down = _MockExpertParamWrapper("down_proj", b_val)

        zero_b_moe = _MockMoEModel(0.0)
        with self.assertRaises(RuntimeError):
            verify_moe_lora_integrity(
                zero_b_moe,
                is_moe=True,
                smoke_test=True,
                expect_trained_adapter=True,
            )

        trained_moe = _MockMoEModel(0.05)
        info = verify_moe_lora_integrity(
            trained_moe,
            is_moe=True,
            smoke_test=True,
            expect_trained_adapter=True,
        )
        self.assertEqual(info["num_expert_wrappers"], 2)
        self.assertGreater(info["expert_lora_b_l2_norm"], 0.0)

        # Verify `.ref` exclusion in scale_and_trim_peft_lora_weights (including ParamWrapper `lora_B.ref.weight` / `.ref`)
        ref_before = trained_moe.experts_gate_up.lora_B["ref"].weight.detach().clone()
        _ = scale_and_trim_peft_lora_weights(trained_moe, lambda_scale=0.5, density=1.0)
        self.assertTrue(torch.allclose(trained_moe.experts_gate_up.lora_B["ref"].weight, ref_before))
        self.assertTrue(torch.allclose(trained_moe.experts_gate_up.lora_B["default"].weight, ref_before * 0.5))

        with tempfile.TemporaryDirectory() as tmpdir:
            ad_dir = pathlib.Path(tmpdir)
            (ad_dir / "adapter_config.json").write_text(
                json.dumps({"target_parameters": ["experts.gate_up_proj", "experts.down_proj"]}),
                encoding="utf-8",
            )
            verify_saved_moe_adapter_config(ad_dir, is_moe=True)
            saved_cfg = json.loads((ad_dir / "adapter_config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["lora_B_layout"], "rank_major")

            (ad_dir / "adapter_config.json").write_text(
                json.dumps({"target_parameters": ["experts.gate_up_proj"], "lora_B_layout": "legacy"}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                verify_saved_moe_adapter_config(ad_dir, is_moe=True)

        # 5. Verify verify_step0_dpo_ref_alignment restores `for_training()`, enters `compute_loss_context_manager()`,
        #    passes at ln(2), and fails on misaligned reference logprobs or eval/train drift
        import contextlib as _ctxlib

        trained_moe.gradient_checkpointing = False
        trained_moe._flag_for_generation = True
        trained_moe.eval()
        for_training_calls: List[Any] = []

        def _mock_for_training(use_gradient_checkpointing: Any = True) -> nn.Module:
            for_training_calls.append(use_gradient_checkpointing)
            trained_moe.gradient_checkpointing = bool(use_gradient_checkpointing)
            if hasattr(trained_moe, "_flag_for_generation"):
                delattr(trained_moe, "_flag_for_generation")
            return trained_moe.train()

        trained_moe.for_training = _mock_for_training  # type: ignore[attr-defined]

        class _DummyTrainer:
            def __init__(self, target_loss_eval: float, target_loss_train: float) -> None:
                self.target_loss_eval = target_loss_eval
                self.target_loss_train = target_loss_train
                self.in_autocast_ctx = False
                self.autocast_enter_count = 0
                self.args = type(
                    "_Args",
                    (),
                    {
                        "precompute_ref_log_probs": True,
                        "per_device_train_batch_size": 1,
                        "gradient_checkpointing": True,
                    },
                )()
                self.train_dataset = Dataset.from_list(
                    [
                        {
                            "chosen_input_ids": [1, 2, 3, 4],
                            "rejected_input_ids": [1, 2, 5, 6],
                            "ref_chosen_logps": -4.2,
                            "ref_rejected_logps": -5.1,
                        }
                    ]
                )
                self._metrics = {"train": {"x": [1.0]}, "eval": {}}
                self._total_train_tokens = 99

            @_ctxlib.contextmanager
            def compute_loss_context_manager(self):
                self.autocast_enter_count += 1
                self.in_autocast_ctx = True
                try:
                    yield
                finally:
                    self.in_autocast_ctx = False

            def data_collator(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
                return {"x": torch.ones(len(items), 8)}

            def _prepare_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                return batch

            def compute_loss(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
                if not self.in_autocast_ctx or not model.training or hasattr(model, "_flag_for_generation"):
                    raise AssertionError("compute_loss called outside autocast or while in for_inference mode!")
                if torch.is_grad_enabled() and not getattr(model, "gradient_checkpointing", False):
                    raise AssertionError("enable_grad forward ran with gradient_checkpointing=False!")
                if (not torch.is_grad_enabled()) and getattr(model, "gradient_checkpointing", False):
                    raise AssertionError("no_grad reference forward ran without disabling gradient_checkpointing!")
                target = self.target_loss_train if torch.is_grad_enabled() else self.target_loss_eval
                reg = (model.experts_gate_up.lora_B["default"].weight.sum() * 0.0) + (
                    model.experts_gate_up.lora_B["default"].weight.pow(2).mean()
                )
                return reg - reg.detach() + torch.tensor(target, dtype=torch.float32)

        dummy_ok = _DummyTrainer(math.log(2.0), math.log(2.0))
        step0_ok = verify_step0_dpo_ref_alignment(
            dummy_ok,
            trained_moe,
            is_moe=True,
        )
        self.assertAlmostEqual(step0_ok["loss_eval"], math.log(2.0), places=5)
        self.assertEqual(step0_ok["eval_train_abs_diff"], 0.0)
        self.assertGreater(step0_ok["expert_grad_norm"], 0.0)
        self.assertEqual(dummy_ok.autocast_enter_count, 2)
        self.assertGreaterEqual(len(for_training_calls), 2)
        self.assertTrue(trained_moe.training)
        self.assertTrue(trained_moe.gradient_checkpointing)
        self.assertFalse(hasattr(trained_moe, "_flag_for_generation"))
        self.assertFalse(trained_moe.router.weight.requires_grad)

        with self.assertRaises(RuntimeError):
            verify_step0_dpo_ref_alignment(
                _DummyTrainer(math.log(2.0), 1.35),
                trained_moe,
                is_moe=True,
            )

        with self.assertRaises(RuntimeError):
            verify_step0_dpo_ref_alignment(
                _DummyTrainer(0.884034, 1.159814),
                trained_moe,
                is_moe=True,
            )

        # 6. Verify Gemma 4 `mm_token_type_ids` parity wrapper on `compute_ref_log_probs`
        sys.modules[__name__].create_causal_mask_mapping = True  # type: ignore[attr-defined]
        try:
            class _MockGemma4Inner(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.config = type("_Cfg", (), {"model_type": "gemma4"})()

            g4_inner = _MockGemma4Inner()
            seen_ref_inputs: List[Dict[str, Any]] = []

            class _MockDPOCls:
                def compute_ref_log_probs(self, mdl: Any, inp: Dict[str, Any]) -> str:
                    seen_ref_inputs.append(inp)
                    return "ok"

            with preserve_stage1_adapter_for_ref_precompute(g4_inner, {}, dpo_trainer_cls=_MockDPOCls):
                out_tag = _MockDPOCls().compute_ref_log_probs(
                    g4_inner, {"input_ids": torch.tensor([[2, 10, 11, 1]])}
                )
            self.assertEqual(out_tag, "ok")
            self.assertIn("mm_token_type_ids", seen_ref_inputs[0])
            self.assertTrue(torch.equal(seen_ref_inputs[0]["mm_token_type_ids"], torch.zeros((1, 4), dtype=torch.long)))
        finally:
            if hasattr(sys.modules[__name__], "create_causal_mask_mapping"):
                delattr(sys.modules[__name__], "create_causal_mask_mapping")


if __name__ == "__main__":
    unittest.main(verbosity=2)
