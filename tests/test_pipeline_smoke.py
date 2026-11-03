"""End-to-end automated unit and smoke test suite for Gemma-4-Arabic.

Verifies:
1. All 9 dataset schema normalizers and mandatory preprocessing rules.
2. Gemma 4 `apply_chat_template` formatting (`enable_thinking=False` and `enable_thinking=True`)
   and preservation of the 4 hidden empty thinking-block tokens (`<|think|>\\n<|think|>\\n`).
3. Strict 75% Arabic / 25% Replay stratified buffer allocation and mixing.
4. Dynamic `named_modules()` discovery, CPU forward hook for `embed_tokens_per_layer`,
   and CPU offload/restoration for `audio_tower` and `vision_tower`.
5. Stage 2 Trust-Region DPO reference EMA math (`tr_alpha=0.6`).
6. TIES-Merging (`lambda=0.85`, `density=0.70`) sign election, trimming, and state dict interpolation.
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
    build_reference_vllm_concurrency_profile,
    generate_vllm_concurrency_plot,
    render_vllm_markdown_report,
)
from scripts.run_benchmarks import (
    evaluate_retention_gate,
    generate_markdown_report,
    parse_lm_eval_results_json,
)
from src.chat_template import (
    format_gemma4_chat,
    format_thinking_response,
    verify_thinking_block_preserved,
)
from src.constants import (
    ALL_9_DATASET_REPOS,
    ARABIC_DIALECT_DPO_REPO,
    DEFAULT_TIES_DENSITY,
    DEFAULT_TIES_LAMBDA,
    DPO_TRUST_REGION_ALPHA,
    EMPTY_THINKING_BLOCK,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_MODULES,
    PRIMARY_MODEL_ID_26B,
    THINK_TOKEN,
)
from src.data_normalizer import (
    drop_pil_image_columns,
    normalize_acegpt_v2_row,
    normalize_arabic_math_row,
    normalize_aya_arabic_row,
    normalize_saudi_laws_row,
    normalize_voho_saudi_row,
)
from src.dpo_formatter import (
    normalize_arabic_dialect_dpo_row,
    normalize_arcee_arabic_dpo_row,
    validate_dpo_load_config,
)
from src.merge_and_export import (
    apply_ties_interpolation_to_state_dict,
    ties_merge_deltas,
    ties_trim_tensor,
)
from src.model_loader import build_8gb_device_map, freeze_embeddings_and_towers
from src.replay_mixer import (
    compute_mix_allocation,
    mix_arabic_and_replay_samples,
    normalize_openmath_instruct_row,
    normalize_ultrachat_row,
)
from src.train_dpo import update_trust_region_reference_params
from src.vram_hooks import (
    apply_8gb_vram_optimizations,
    get_vram_telemetry,
    restore_multimodal_towers,
)


class TestGemma4ArabicPipeline(unittest.TestCase):
    """Comprehensive unit and smoke test suite for the Gemma-4-Arabic repository."""

    def test_01_constants_and_yaml_configs(self) -> None:
        """Verify all 9 dataset repos and YAML config invariants."""
        self.assertEqual(len(ALL_9_DATASET_REPOS), 9)
        self.assertEqual(LORA_RANK, 32)
        self.assertEqual(LORA_ALPHA, 64)
        self.assertEqual(len(LORA_TARGET_MODULES), 7)

        sft_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "sft_config.yaml").read_text(encoding="utf-8")
        )
        dpo_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "dpo_config.yaml").read_text(encoding="utf-8")
        )
        eval_cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval_config.yaml").read_text(encoding="utf-8")
        )

        self.assertTrue(sft_cfg["lora"]["use_rslora"])
        self.assertEqual(sft_cfg["lora"]["r"], 32)
        self.assertEqual(sft_cfg["lora"]["lora_alpha"], 64)
        self.assertEqual(sft_cfg["datasets"]["arabic_mix_ratio"], 0.75)
        self.assertEqual(sft_cfg["datasets"]["replay_mix_ratio"], 0.25)

        self.assertEqual(dpo_cfg["dpo"]["beta"], 0.1)
        self.assertEqual(dpo_cfg["dpo"]["tr_alpha"], 0.6)
        self.assertEqual(dpo_cfg["training"]["learning_rate"], 5.0e-7)
        self.assertEqual(dpo_cfg["dpo"]["max_length"], 1536)

        self.assertEqual(len(eval_cfg["arabic_benchmarks"]), 7)
        self.assertIn("MGSM-Ar", eval_cfg["arabic_benchmarks"])
        self.assertEqual(len(eval_cfg["retention_anchors"]), 4)

    def test_02_gemma4_chat_template_thinking_tokens(self) -> None:
        """Verify Gemma 4 chat template preserves thinking-block tokens in both modes."""
        msgs_direct = [
            {"role": "user", "content": "ما هي عاصمة المملكة العربية السعودية؟"},
            {"role": "assistant", "content": "الرياض هي العاصمة."},
        ]
        rendered_false = format_gemma4_chat(msgs_direct, enable_thinking=False)
        self.assertIn(EMPTY_THINKING_BLOCK, rendered_false)
        self.assertTrue(verify_thinking_block_preserved(rendered_false, enable_thinking=False))

        msgs_thinking = [
            {"role": "user", "content": "احسب 9 * 7"},
            {
                "role": "assistant",
                "content": format_thinking_response("9 ضرب 7 يساوي 63", "الناتج هو 63."),
            },
        ]
        rendered_true = format_gemma4_chat(msgs_thinking, enable_thinking=True)
        self.assertGreaterEqual(rendered_true.count(THINK_TOKEN), 2)
        self.assertTrue(verify_thinking_block_preserved(rendered_true, enable_thinking=True))

    def test_03_all_5_arabic_sft_normalizers(self) -> None:
        """Verify schema normalizers for all 5 Stage 1 Arabic SFT datasets."""
        # 1. VohoAI/voho-saudi-dialogues (messages_json + speaker prefix stripping)
        voho = normalize_voho_saudi_row(
            {
                "messages_json": json.dumps(
                    [
                        {"role": "user", "content": "User: وشلون أطلع بدل فاقد للهوية؟"},
                        {"role": "assistant", "content": "المساعد: عن طريق منصة أبشر ثم خدماتي."},
                    ]
                )
            }
        )
        self.assertIsNotNone(voho)
        assert voho is not None
        self.assertFalse(voho["messages"][1]["content"].startswith("User:"))
        self.assertFalse(voho["messages"][2]["content"].startswith("المساعد:"))
        self.assertIn(EMPTY_THINKING_BLOCK, voho["text"])

        # 2. SaudiArabicLaws ("moj" flat and "boe" nested)
        moj = normalize_saudi_laws_row(
            {
                "doc_title": "نظام العمل",
                "article_title": "المادة الأولى",
                "text": "يسمى هذا النظام نظام العمل.",
            },
            config_name="moj",
        )
        boe = normalize_saudi_laws_row(
            {
                "law": {"title_ar": "نظام المعاملات المدنية"},
                "article": {"article_title_norm": "المادة الخامسة", "text": ""},
                "retrieval": {"contextual_text": "تطبق أحكام هذا النظام على جميع المسائل."},
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
                    {"from": "human", "value": "عرف البلاغة في اللغة العربية."},
                    {"from": "gpt", "value": "البلاغة هي مطابقة الكلام لمقتضى الحال."},
                ]
            }
        )
        self.assertIsNotNone(ace)
        assert ace is not None
        self.assertEqual(ace["messages"][1]["role"], "user")
        self.assertEqual(ace["messages"][2]["role"], "assistant")

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
                "problem": "حل المعادلة 2س + 4 = 10",
                "solution": "نطرح 4 من الطرفين: 2س = 6\nنقسم على 2: س = 3",
            }
        )
        self.assertIsNotNone(math_item)
        assert math_item is not None
        self.assertTrue(math_item["enable_thinking"])
        self.assertGreaterEqual(math_item["text"].count(THINK_TOKEN), 2)

        # 5. CohereForAI/aya_collection_language_split (inputs/targets + JSON tool-calling)
        aya = normalize_aya_arabic_row(
            {
                "inputs": "تحقق من رصيد الإجازات للموظف رقم 102",
                "targets": "رصيد الإجازات المتبقي هو 21 يوما.",
                "tools": [{"name": "get_leave_balance"}],
                "tool_call": {"name": "get_leave_balance", "arguments": {"emp_id": 102}},
            }
        )
        self.assertIsNotNone(aya)
        assert aya is not None
        self.assertIn("get_leave_balance", aya["text"])

    def test_04_replay_normalizers_and_75_25_mixer(self) -> None:
        """Verify Replay Buffer normalizers (Datasets 6 & 7) and strict 75/25 mixing."""
        # 6. HuggingFaceH4/ultrachat_200k
        ultra = normalize_ultrachat_row(
            {
                "messages": [
                    {"role": "user", "content": "Explain binary search."},
                    {"role": "assistant", "content": "Binary search halves the search interval each step."},
                ]
            }
        )
        self.assertIsNotNone(ultra)
        assert ultra is not None
        self.assertFalse(ultra["enable_thinking"])

        # 7. nvidia/OpenMathInstruct-2
        openmath = normalize_openmath_instruct_row(
            {
                "problem": "Evaluate 7! / 5!.",
                "generated_solution": "7! / 5! = 7 * 6 = 42.",
                "expected_answer": "42",
            }
        )
        self.assertIsNotNone(openmath)
        assert openmath is not None
        self.assertTrue(openmath["enable_thinking"])

        alloc = compute_mix_allocation(total_samples=200)
        self.assertEqual(alloc["arabic_total"], 150)
        self.assertEqual(alloc["replay_total"], 50)

        ar_pool = [{"source": "voho_saudi_dialogues", "text": f"ar_{i}"} for i in range(12)]
        rep_pool = [ultra, openmath, ultra, openmath]
        mixed, stats = mix_arabic_and_replay_samples(ar_pool, rep_pool)
        self.assertAlmostEqual(stats["arabic_fraction"], 0.75, places=6)
        self.assertAlmostEqual(stats["replay_fraction"], 0.25, places=6)

    def test_05_dpo_normalizers_and_prompt_unification(self) -> None:
        """Verify Stage 2 DPO normalizers (Datasets 8 & 9) and prompt unification."""
        # 8. HeshamHaroon/arabic-dialect-dpo (must require name="saudi")
        self.assertEqual(validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "saudi"), "saudi")
        with self.assertRaises(ValueError):
            validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, "default")
        with self.assertRaises(ValueError):
            validate_dpo_load_config(ARABIC_DIALECT_DPO_REPO, None)

        saudi_dpo = normalize_arabic_dialect_dpo_row(
            {
                "prompt": "كيف أقدم على وظيفة في جدارات؟",
                "chosen": "سجل الدخول عبر النفاذ الوطني في منصة جدارات وأكمل ملفك المهني ثم قدم على الوظائف المتاحة.",
                "rejected": "لا أعرف.",
            }
        )
        self.assertIsNotNone(saudi_dpo)

        # 9. arcee-globe/arabic-orpo-dpo-mix-40k-filtered (prompt unification)
        arcee_dpo = normalize_arcee_arabic_dpo_row(
            {
                "chosen": [
                    {"role": "user", "content": "الصياغة المعتمدة للسؤال باللغة العربية"},
                    {"role": "assistant", "content": "إجابة نموذجية مفصلة وصحيحة."},
                ],
                "rejected": [
                    {"role": "user", "content": "صياغة مختلفة تماما يجب استبدالها"},
                    {"role": "assistant", "content": "إجابة ضعيفة."},
                ],
            }
        )
        self.assertIsNotNone(arcee_dpo)
        assert arcee_dpo is not None
        self.assertEqual(
            arcee_dpo["chosen_messages"][0]["content"],
            "الصياغة المعتمدة للسؤال باللغة العربية",
        )
        self.assertEqual(
            arcee_dpo["rejected_messages"][0]["content"],
            "الصياغة المعتمدة للسؤال باللغة العربية",
        )

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
        self.assertAlmostEqual(float(interpolated["layer.weight"][0].item()), 11.70, places=4)
        self.assertAlmostEqual(float(interpolated["layer.weight"][1].item()), 23.40, places=4)

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

        sft_26b = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "sft_26b_a4b_a100.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(sft_26b["model"]["base_model_id"], PRIMARY_MODEL_ID_26B)
        self.assertEqual(sft_26b["datasets"]["total_target_samples"], 120000)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
