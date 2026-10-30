# Gemma-4-Arabic: Saudi & MSA Fine-Tuning, Alignment, and Evaluation Pipeline

End-to-end continual instruction tuning, Direct Preference Optimization (DPO), TIES model merging, and benchmark evaluation pipeline for adapting **`unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`** (`google/gemma-4-E4B-it`) to **Modern Standard Arabic (MSA)** and **Saudi dialects (Najdi, Hijazi, Khaliji)** on a single **8.0 GB VRAM** consumer GPU (single 8.0 GB GPU) without catastrophic forgetting.

---

## Key Engineering Highlights

1. **8.0 GB VRAM Breakthrough (10.95 GB -> 4.36 GB Active GPU Footprint):**
   - **Per-Layer Embeddings (`embed_tokens_per_layer`) CPU RAM Forward Hook:** In Gemma 4 E4B, `embed_tokens_per_layer` (`[262144, 10752]` in `bfloat16`) is skipped by 4-bit quantization and consumes **5.637 GB** of VRAM. [`src/vram_hooks.py`](src/vram_hooks.py) dynamically discovers this module via `named_modules()`, pins its weights in system RAM, moves look-up token IDs to CPU in a `register_forward_pre_hook`, and transfers only the looked-up activations (`[batch, seq_len, 10752]`, ~42 MB at `seq_len=2048`) back to CUDA via `non_blocking=True`.
   - **Multimodal Tower CPU Offloading:** Offloads `audio_tower` (**0.618 GB**) and `vision_tower` (**0.339 GB**) to CPU RAM during text-only SFT and DPO, then restores them cleanly prior to final checkpoint export.
2. **Gemma 4 Native Chat Template & Thinking Token Preservation:**
   - [`src/chat_template.py`](src/chat_template.py) enforces `tokenizer.apply_chat_template` using Gemma 4 control tokens (`<|turn|>`, `<|think|>`, `<turn|>`) so that the mandatory empty thinking block `<|think|>\n<|think|>\n` is preserved for direct answers (`enable_thinking=False`) and populated for reasoning traces (`enable_thinking=True`).
3. **Zero-Forgetting 3:1 Experience Replay & Rank-Stabilized LoRA (rsLoRA):**
   - Mixes **75% Arabic & Saudi domain data** with **25% general capability replay data** (English instruction following, coding, and math reasoning) using `r=32`, `lora_alpha=64`, `use_rslora=True` ($\alpha / \sqrt{r} = 11.31$), `NEFTune` ($\alpha=5$), and completion-only loss masking.
4. **Trust-Region DPO (TR-DPO) & TIES-Merging:**
   - Stage 2 aligns dialect authenticity and cultural safety with `beta=0.1`, `lr=5.0e-7`, and an Exponential Moving Average (EMA) reference anchor (`tr_alpha=0.6`).
   - Post-training [`src/merge_and_export.py`](src/merge_and_export.py) applies **TIES-Merging** (`density=0.85`, `lambda=0.85`) in `bfloat16` on CPU RAM to resolve sign conflicts and preserve base model capabilities before exporting to 16-bit SafeTensors and GGUF (`q4_k_m`, `q8_0`).

---

## 8.0 GB VRAM Budget Breakdown

| Component | Default `bnb-4bit` Load | With `src/vram_hooks.py` Offload | Notes |
| :--- | :---: | :---: | :--- |
| **4-bit NF4 Quantized Backbone & LM Head** | 4.360 GB | **4.360 GB** | Resident on GPU (`cuda:0`) |
| **`embed_tokens_per_layer` (`[262144, 10752]`, BF16)** | 5.637 GB | **0.000 GB (CPU RAM)** | Forward hook streams ~42 MB activation slice |
| **`audio_tower` + `vision_tower` (BF16)** | 0.956 GB | **0.000 GB (CPU RAM)** | Offloaded during text training; restored at export |
| **rsLoRA Adapters (`r=32`) + 8-bit AdamW States** | ~0.850 GB | **~0.850 GB** | All 7 linear projections (`q,k,v,o,gate,up,down`) |
| **Activations (`bs=1`, `seq=2048`, Unsloth GC)** | ~1.450 GB | **~1.450 GB** | `gradient_checkpointing="unsloth"`, `max_split_size_mb:128` |
| **Total Peak GPU VRAM** | **> 13.25 GB (OOM)** | **~6.66 GB (Fits 8.0 GB)** | Leaves > 1.3 GB safety headroom on 8.0 GB GPU |

---

## Verified 9-Dataset Curriculum

### Stage 1: Continual Instruction Tuning (75% Arabic / 25% Replay)

| # | HuggingFace Repository ID | Config / Split | Role & Mix Share | Schema Normalization (`src/data_normalizer.py` & `src/replay_mixer.py`) |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `VohoAI/voho-saudi-dialogues` | `default` / `train` (`868` rows) | Saudi Multi-Turn Dialogues (`15%` Arabic / `11.25%` Total) | Parses bilingual JSON string `topic`, keeps `messages`, filters `num_turns >= 2`. |
| **2** | `riotu-lab/SaudiArabicLaws` / [`saudi-arabic-laws-and-regulations-corpus`](https://huggingface.co/datasets/SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus) | `"moj"` (`7,090`) & `"boe"` (`15,612`) / `train` | Saudi Legal & Civic Grounding (`15%` Arabic / `11.25%` Total) | Concatenates both configs, converts `{prompt, response}` to `messages`, prepends Saudi legal system prompt. |
| **3** | `FreedomIntelligence/AceGPT-v2-tuning` | `default` / `train` (`272,499` rows) | Large-Scale Arabic Instruction (`35%` Arabic / `26.25%` Total) | Converts ShareGPT `conversations` (`from`/`value`) to standard `messages` (`role`/`content`). |
| **4** | `Omartificial-Intelligence-Space/Arabic-Math-SFT` | `default` / `train` (`49,645` rows) | Arabic STEM & Math CoT (`15%` Arabic / `11.25%` Total) | Calls `.remove_columns(["image"])` first to save RAM, maps `{problem, solution}` to `messages` with `enable_thinking=True`. |
| **5** | `CohereForAI/aya_collection_language_split` | `"arabic"` / `train` (streaming) | Broad MSA Foundations (`20%` Arabic / `15.00%` Total) | Streams `"arabic"` config, filters high-priority task types, maps `{inputs, targets}` to `messages`. |
| **6** | `HuggingFaceH4/ultrachat_200k` | `default` / `train_sft` (`207,865` rows) | English Multi-Turn Replay (`60%` Replay / `15.00%` Total) | Retains native `messages` (`role`, `content`) for multi-turn instruction retention. |
| **7** | `nvidia/OpenMathInstruct-2` | `default` / `train_1M` (`1,000,000` rows) | Math & Logic CoT Replay (`40%` Replay / `10.00%` Total) | Maps `{problem, generated_solution}` to `messages` with `enable_thinking=True`. |

### Stage 2: Trust-Region DPO Preference Alignment

| # | HuggingFace Repository ID | Config / Split | Mix Share | Schema Normalization (`src/dpo_formatter.py`) |
| :-: | :--- | :--- | :---: | :--- |
| **8** | `HeshamHaroon/arabic-dialect-dpo` | `name="saudi"` / `train` (`1,160` rows) | `35%` | Maps `{question, chosen, rejected}` to TRL DPO message lists with Saudi dialect system prompt. |
| **9** | `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` | `default` / `train` (`29,984` rows) | `65%` | Copies `chosen[0]["content"]` into `rejected[0]["content"]` before prompt extraction to fix English/Arabic prompt mismatch in `rejected`. |

---

## Repository Structure

```text
Gemma-4-Arabic/
|-- configs/
|   |-- sft_config.yaml          # Stage 1 rsLoRA, VRAM flags, and 75/25 mixture config
|   |-- dpo_config.yaml          # Stage 2 Trust-Region DPO config (beta=0.1, tr_alpha=0.6)
|   `-- eval_config.yaml         # 6 Arabic benchmarks + 4 Retention Gate anchors
|-- src/
|   |-- constants.py             # Verified HF dataset IDs, prompts, and Gemma 4 special tokens
|   |-- vram_hooks.py            # Dynamic PLE CPU forward hook & multimodal tower offloading
|   |-- model_loader.py          # Unsloth 4-bit NF4 loader + rsLoRA initialization
|   |-- chat_template.py         # Gemma 4 chat template wrapper (<|think|> token handling)
|   |-- data_normalizer.py       # Normalizers for all 5 Arabic/Saudi SFT datasets
|   |-- replay_mixer.py          # Normalizers for Replay datasets + strict 75/25 mixer
|   |-- dpo_formatter.py         # Normalizers for Saudi & MSA DPO preference datasets
|   |-- train_sft.py             # Stage 1 SFT trainer with completion-only loss masking
|   |-- train_dpo.py             # Stage 2 Trust-Region DPO trainer with EMA reference anchor
|   `-- merge_and_export.py      # TIES-Merging (lambda=0.85, density=0.85), tower restore, GGUF export
|-- scripts/
|   |-- build_sft_dataset.py     # CLI builder for Stage 1 SFT dataset
|   |-- build_dpo_dataset.py     # CLI builder for Stage 2 DPO dataset
|   `-- run_benchmarks.py        # CLI runner for lm-evaluation-harness & Retention Gate report
|-- tests/
|   `-- test_pipeline_smoke.py   # 9-test end-to-end smoke & invariant verification suite
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

---

## Quickstart & CLI Usage

### 1. Environment Verification & Smoke Tests

Run the automated unit and smoke test suite to verify all dataset normalizers, VRAM hooks, chat templates, and mathematical utilities:

```powershell
python -m unittest discover -s "D:\PERSONAL\Projects\Gemma-4-Arabic\tests" -v
```

### 2. Build Stage 1 SFT Dataset (75% Arabic / 25% Replay)

Dry-run verification with synthetic samples:

```powershell
python scripts/build_sft_dataset.py --config configs/sft_config.yaml
```

Full production dataset build (saves to `data/sft_mix_80k`):

```powershell
python scripts/build_sft_dataset.py --config configs/sft_config.yaml --output-dir data/sft_mix_80k
```

### 3. Build Stage 2 DPO Preference Dataset

Dry-run verification:

```powershell
python scripts/build_dpo_dataset.py --config configs/dpo_config.yaml
```

Full production DPO dataset build (saves to `data/dpo_mix_25k`):

```powershell
python scripts/build_dpo_dataset.py --config configs/dpo_config.yaml --output-dir data/dpo_mix_25k
```

### 4. Run Stage 1 SFT Training (`src/train_sft.py`)

2-step smoke test on GPU (verifies VRAM hooks stay well below 8.0 GB):

```powershell
python src/train_sft.py --config configs/sft_config.yaml --smoke-test
```

Full Stage 1 rsLoRA SFT training run:

```powershell
python src/train_sft.py --config configs/sft_config.yaml --dataset-dir data/sft_mix_80k --output-dir outputs/gemma4-e4b-arabic-sft
```

### 5. Run Stage 2 Trust-Region DPO Alignment (`src/train_dpo.py`)

Full Stage 2 TR-DPO alignment run:

```powershell
python src/train_dpo.py --config configs/dpo_config.yaml --sft-adapter-dir outputs/gemma4-e4b-arabic-sft/final_adapter --dataset-dir data/dpo_mix_25k --output-dir outputs/gemma4-e4b-arabic-dpo
```

### 6. Post-Training TIES-Merging & GGUF Export (`src/merge_and_export.py`)

Merge the aligned LoRA delta using TIES-Merging (`lambda=0.85`, `density=0.85`), restore `audio_tower` and `vision_tower`, and export 16-bit SafeTensors + GGUF (`q4_k_m`, `q8_0`):

```powershell
python src/merge_and_export.py --base-model unsloth/gemma-4-E4B-it-unsloth-bnb-4bit --adapter-dir outputs/gemma4-e4b-arabic-dpo/final_dpo_adapter --output-dir outputs/gemma4-e4b-arabic-merged --lambda-scale 0.85 --density 0.85 --export-gguf
```

### 7. Benchmark Evaluation & Zero-Forgetting Retention Gate (`scripts/run_benchmarks.py`)

Dry-run evaluation pipeline check:

```powershell
python scripts/run_benchmarks.py --config configs/eval_config.yaml
```

Full evaluation across all 6 Arabic/Saudi benchmarks (`ArabicMMLU`, `MadinahQA`, `AlGhafa`, `ACVA`, `EXAMS-AR`, `AraTrust`) and 4 Retention Gate anchors (`MMLU-Pro`, `GSM8K`, `HumanEval`, `ARC-Challenge`):

```powershell
python scripts/run_benchmarks.py --config configs/eval_config.yaml --candidate-model outputs/gemma4-e4b-arabic-merged --suite all
```
