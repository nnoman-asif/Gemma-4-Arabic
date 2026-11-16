# Gemma-4-Arabic Iso-Active Model Family (`E4B` Dense + `26B-A4B` Sparse MoE)

End-to-end continual instruction tuning (`QLoRA` across dense + 128 3D MoE experts), Conversational Direct Preference Optimization (`DPO`), Lossless 16-Bit Adapter Merging (`lambda=1.0`), 10-Suite Full-Dataset Evaluation, and Iso-Active Latency/vLLM Profiling pipeline for adapting Google's **Gemma 4** family to **Modern Standard Arabic (MSA)** and **Saudi dialects (Najdi, Hijazi, Khaliji)** with **Zero Catastrophic Forgetting**.

---

## Published Hugging Face Hub Artifacts

| Artifact Type | Hugging Face Repository | Architecture / Size | Description |
| :--- | :--- | :---: | :--- |
| **Edge / Workstation Model** | [`noman-asif/Gemma-4-E4B-Arabic-It`](https://huggingface.co/noman-asif/Gemma-4-E4B-Arabic-It) | **4.5B Total / 4.0B Active** (Dense) | Standalone 16-bit (`bfloat16`) `safetensors` checkpoint (`r=32, alpha=32`) for laptops & single-GPU workstations |
| **Enterprise MoE Flagship** | [`noman-asif/Gemma-4-26B-A4B-Arabic-It`](https://huggingface.co/noman-asif/Gemma-4-26B-A4B-Arabic-It) | **25.2B Total / 3.8B Active** (128 Experts, Top-8 + 1 Shared) | Standalone 16-bit (`bfloat16`) `safetensors` MoE flagship (`r=16, alpha=16` across attention + shared MLP + all 128 3D experts) |
| **Stage 1 SFT Dataset** | [`noman-asif/Arabic-Reasoning-Instruct-120k`](https://huggingface.co/datasets/noman-asif/Arabic-Reasoning-Instruct-120k) | **120,000 rows** (75% Arabic / 25% Replay) | 15-source 13-gram de-contaminated Arabic, Saudi dialect, legal (`moj` + `boe`), and native Gemma 4 reasoning curriculum |
| **Stage 2 DPO Dataset** | [`noman-asif/Arabic-Preference-DPO-20k`](https://huggingface.co/datasets/noman-asif/Arabic-Preference-DPO-20k) | **20,000 pairs** (85% Arabic/Saudi / 15% English Replay) | Conversational preference alignment dataset with unified prompt verification and length-ratio hygiene |

---

## The Iso-Active Compute Thesis: Why `Gemma-4-E4B` + `Gemma-4-26B-A4B`?

By pairing **`Gemma-4-E4B-it` (`4.0B` active / `4.5B` total)** with **`Gemma-4-26B-A4B-it` (`3.8B` active / `25.2B` total)**, both models activate **~4 Billion parameters per token (`3.8B-4.0B` FLOPs/token)**:
- **Same 4B-Class Inference Speed (~11-13 tok/s in unbatched PyTorch `generate()`, ~111-130 tok/s in vLLM):** Because `26B-A4B` routes each token through only **8 out of 128 fine-grained experts + 1 shared expert (`3.8B` active)**, its per-token FLOPs and generation latency match the `4B` dense model.
- **Full 128-Expert 3D Parameter Adaptation (`~498.5M` Trainable Params at `r=16`):** On `Gemma-4-26B-A4B-it`, Unsloth auto-discovers attention projections, shared dense MLPs, and all **128 3D fused expert tensors** (`experts.gate_up_proj`, `experts.down_proj`) via PEFT `target_parameters` so all `21B+` sparse knowledge weights absorb Arabic morphology, Saudi statutory law, and reasoning.
- **Frozen MoE Router Gate (`freeze_moe_router=True`):** The router gate (`router` / `.gate.`) is strictly frozen during both SFT and DPO so token load-balancing across all 128 experts remains mathematically identical to Google's pretrained distribution (zero expert collapse).

---

## Visual Benchmark & Efficiency Comparisons

1. **6-Suite Arabic & Saudi Benchmark Progression**
   ![Arabic Benchmark Comparison](assets/plots/oall_v2_arabic_comparison.png)

2. **Zero-Forgetting Retention Shield**
   ![Zero-Forgetting Retention Shield](assets/plots/zero_forgetting_retention_shield.png)

3. **Iso-Active Latency, Throughput & VRAM Pareto Curve**
   ![Iso-Active Latency & VRAM Pareto](assets/plots/iso_active_latency_vram_pareto.png)

4. **Simultaneous Dual-GPU SFT & DPO Training Convergence**
   ![SFT & DPO Training Curves](assets/plots/sft_dpo_training_curves.png)

---

## Dual-GPU (`2x A100 80 GB`) Pipeline Orchestration (`scripts/run_dual_a100_pipeline.sh`)

Running `bash scripts/run_dual_a100_pipeline.sh` executes a 4-phase, barrier-synchronized workflow designed for maximum 2-GPU efficiency and zero race conditions:

1. **Phase 0 (Upfront Environment & Dependency Verification):** Loads `.env` if present, verifies `>= 2` CUDA GPUs and `>= 110 GB` free disk space, and verifies/installs `transformers==5.17.0`, `peft>=0.21.3`, `trl==1.15.0`, `unsloth`, `vllm`, `lm-eval>=0.4.13`, `inspect-evals>=0.29.0`, and `lighteval@main` before training starts.
2. **Phase 1 & Phase 2A/2B (Parallel Dual-GPU Training + Hard Merge Barrier):**
   - Pulls the pre-built `120,000` SFT and `20,000` DPO datasets onto local disk once.
   - **`GPU 0`** trains **`Gemma-4-E4B-Arabic-It`** (`Stage 1 SFT` -> `Stage 2 DPO`, `r=32, alpha=32`) while **`GPU 1`** simultaneously trains **`Gemma-4-26B-A4B-Arabic-It`** (`Stage 1 SFT` -> `Stage 2 DPO`, `r=16, alpha=16`).
   - **Hard Barrier:** Benchmarking does **not** start until **both** models finish training and **both** adapters are merged into standalone `merged_16bit` (`bfloat16`, `lambda=1.0`) checkpoints on disk.
3. **Phase 3 (Parallel Base-vs-Tuned Benchmarking per Model):**
   - Prewarms all 10 benchmark datasets in one foreground process.
   - **Step 3A (`E4B`):** Loads **`Base E4B`** on **`GPU 0`** (`--only-role base`) and **`Tuned E4B`** on **`GPU 1`** (`--only-role tuned`) simultaneously, waits for both, and combines `live_raw_e4b.json` + `benchmark_comparison_e4b.md`.
   - **Step 3B (`26B-A4B`):** Once `E4B` finishes, loads **`Base 26B-A4B`** on **`GPU 0`** (`--only-role base`) and **`Tuned 26B-A4B`** on **`GPU 1`** (`--only-role tuned`) simultaneously, waits for both, and combines `live_raw_26b_a4b.json` + `family_benchmark_comparison.md`.
4. **Phase 4 (Latency, 50-User vLLM Concurrency & Sync Bundle):** Profiles single-stream TTFT/decode speed, 1–50 concurrent user vLLM scaling, and packages `outputs/gemma4_arabic_local_sync_bundle.tar.gz`.
5. **Automatic Interruption & Spot-Preemption Resume:** Re-running `bash scripts/run_dual_a100_pipeline.sh` automatically resumes from the exact interrupted stage — Stage 1 SFT and Stage 2 DPO save rolling checkpoints (`save_steps=100, save_total_limit=2`) with multi-adapter policy/reference verification and disk-cached pre-tokenization, Phase 2B auto-recovers incomplete `merged_16bit` exports, and Phase 3/4 atomically checkpoint completed benchmark suites into `outputs/eval_reports/live_raw_*.json` with zero synthetic fallbacks.

---

## Verified V2 Curriculum (`120,000` SFT + `20,000` DPO — 13-Gram Decontaminated)

### Stage 1: Continual Instruction Tuning (`120,000` Samples | 75% Arabic / 25% Replay)

| Bucket | Curated Hugging Face Sources | Target Rows (`Share`) | Key Engineering Invariants |
| :--- | :--- | :---: | :--- |
| **Arabic Reasoning CoT** | `AdaMLLab/smolkalam-arabic-conversational-sft` (`OpenThoughts3_50K`), `oddadmix/arabic-math-reasoning-synth`, `Gheras/Medical_Reasoning_Arabic_Dataset` | `30,000` (`25.00%`) | Explicit `reasoning_content` separated into Gemma 4 `<\|channel>thought\n...<channel\|>`, zero raw `<think>` tags |
| **Saudi Dialect & Legal** | `VohoAI/voho-saudi-dialogues`, `SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus` (`moj` + `boe`), `CohereLabs/aya_dataset` (`ars` / `arb`) | `18,000` (`15.00%`) | Najdi/Hijazi/Khaliji dialogues, active Saudi statutory law QA, human-annotated Arabic |
| **MSA Science & Instruction** | `AdaMLLab/smolkalam-arabic-conversational-sft` (`Mixture_of_Thoughts_science_no_think` + `OpenHermes_2.5_no_think`), `FreedomIntelligence/alpaca-gpt4-arabic` | `42,000` (`35.00%`) | High-complexity MSA STEM & instruction following, 70% zero-system-prompt alignment for 0-shot OALL v2 |
| **Code Generation Replay** | `nvidia/OpenCodeReasoning` (`split_0`), `open-r1/codeforces-cots` (`solutions_py_decontaminated`), `m-a-p/CodeFeedback-Filtered-Instruction` | `12,000` (`10.00%`) | Competitive programming CoT (`LiveCodeBench-v6` retention shield) |
| **Competition Math & Science Replay** | `nvidia/OpenMathInstruct-2` (`train_1M`), `nvidia/OpenScienceReasoning-2` | `13,500` (`11.25%`) | Preserves full reasoning trace **and** final `\boxed{answer}` (`AIME-2026` & `GPQA-Diamond` shield) |
| **General English Multi-Turn Replay** | `HuggingFaceH4/ultrachat_200k` (`train_sft`) | `4,500` (`3.75%`) | Multi-turn English instruction following (`MMLU-Pro` shield) |

### Stage 2: Conversational DPO Preference Alignment (`20,000` Pairs)

| Subset | Hugging Face Source | Share | Hygiene & Formatting (`src/dpo_formatter.py`) |
| :--- | :--- | :---: | :--- |
| **Saudi Dialect Preferences** | `HeshamHaroon/arabic-dialect-dpo` (`saudi`) | `20%` | Conversational `{prompt, chosen, rejected}`, strips prompt leakage & `<start_of_turn>` artifacts |
| **MSA & Cultural Safety** | `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` | `65%` | Unifies user prompts across `chosen`/`rejected`, enforces length-ratio hygiene (`<= 2.5x`) |
| **English Preference Replay** | `argilla/ultrafeedback-binarized-preferences-cleaned` | `15%` | High-margin (`>= 1.5`, `chosen_rating >= 4.5`) English anchor preventing Stage 2 English drift |

---

## Benchmark Evaluation Suite (6 OALL v2 Arabic + 4 Official Gemma 4 Retention Benchmarks)

All 10 benchmarks run on **100% full datasets** (zero `--limit` subsampling, zero YAML reference fallbacks) and report **`Base | Arabic-Tuned`** in `outputs/eval_reports/family_benchmark_comparison.md`:

| Category | Benchmark | Upstream Harness & Task | Metric | Shots / Mode | Full Dataset Size |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Arabic (OALL v2)** | **ArabicMMLU** | `lighteval` (`arabic_mmlu:*\|0`, all 40 subjects) | `acc_norm` | 0-shot MCQ | `14,455` |
| **Arabic (OALL v2)** | **MadinahQA** | `lighteval` (`madinah_qa:*\|0`, General + Grammar) | `acc_norm` | 0-shot MCQ | `977` |
| **Arabic (OALL v2)** | **AlGhafa** | `lighteval` (`alghafa:*\|0`, 9 native subsets) | `acc_norm` | 0-shot MCQ | `22,932` |
| **Arabic (OALL v2)** | **ACVA** | `lighteval` (`acva:*\|5`, all 58 subtasks) | `acc_norm` | 5-shot MCQ | `8,710` |
| **Arabic (OALL v2)** | **EXAMS-AR** | `lighteval` (`arabic_exams\|0`) | `acc_norm` | 0-shot MCQ | `537` |
| **Arabic (OALL v2)** | **AraTrust** | `lighteval` (`aratrust:*\|0`, 8 categories) | `acc_norm` | 0-shot MCQ | `521` |
| **Retention (Gemma 4)** | **MMLU-Pro** | `lm-eval>=0.4.13` (`mmlu_pro`) | `exact_match,custom-extract` | 5-shot Multiturn CoT, Thinking ON | `12,032` |
| **Retention (Gemma 4)** | **GPQA-Diamond** | `inspect-evals>=0.29.0` (`inspect_evals/gpqa_diamond`) | `accuracy` | 0-shot CoT (`epochs=1`), Thinking ON | `198` |
| **Retention (Gemma 4)** | **AIME-2026** | `inspect-evals>=0.29.0` (`inspect_evals/aime2026`, no tools) | `accuracy` | 0-shot CoT (`epochs=4`, `avg@4`), Thinking ON | `30 x 4 = 120` |
| **Retention (Gemma 4)** | **LiveCodeBench-v6** | `lighteval@main` (`lcb:codegeneration_v6\|0`) | `pass@1` | 0-shot CoT (`n=1`), Thinking ON | `175` |

---

## Quickstart & CLI Usage

### 1. Run the Automated Verification Suite

```bash
pytest tests/test_pipeline_smoke.py -v
```

### 2. `2x A100 80 GB` One-Command End-to-End Pipeline

```bash
bash scripts/run_dual_a100_pipeline.sh
```

### 3. Modular CLI Execution

```bash
# Pull pre-built V2 datasets (or build locally on CPU):
python scripts/build_sft_dataset.py --config configs/sft_26b_a4b_a100.yaml --output-dir data/sft_mixed --pull-from-hub
python scripts/build_dpo_dataset.py --config configs/dpo_26b_a4b_a100.yaml --output-dir data/dpo_mixed --pull-from-hub

# Stage 1 QLoRA SFT + Stage 2 DPO + Lossless 16-bit Merge:
python src/train_sft.py --config configs/sft_e4b_a100.yaml --dataset-dir data/sft_mixed --output-dir outputs/sft_e4b_a100
python src/train_dpo.py --config configs/dpo_e4b_a100.yaml --dataset-dir data/dpo_mixed --sft-adapter-dir outputs/sft_e4b_a100 --output-dir outputs/dpo_e4b_a100
python src/merge_and_export.py --base-model unsloth/gemma-4-E4B-it --adapter-dir outputs/dpo_e4b_a100 --output-dir outputs/merged_gemma4_e4b_arabic --lambda-scale 1.0 --export-format merged_16bit

# Parallel Base-vs-Tuned Benchmarking for E4B across 2 GPUs:
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmarks.py --config configs/eval_config.yaml --model-key e4b --only-role base --use-vllm &
CUDA_VISIBLE_DEVICES=1 python scripts/run_benchmarks.py --config configs/eval_config.yaml --model-key e4b --only-role tuned --use-vllm &
wait
python scripts/run_benchmarks.py --config configs/eval_config.yaml --model-key e4b --combine-role-reports
```

---

## License

This project, including all training scripts, evaluation suites, datasets, and merged model weights, is licensed under the [Apache License 2.0](LICENSE) ([`apache-2.0`](https://www.apache.org/licenses/LICENSE-2.0)).
