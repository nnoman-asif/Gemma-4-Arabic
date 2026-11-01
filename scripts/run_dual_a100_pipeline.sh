#!/usr/bin/env bash
# ==============================================================================
# Simultaneous Dual-Model Training, Evaluation, and Latency Profiling Runner
# Target Hardware: 2x NVIDIA A100 80 GB SXM4/PCIe (160 GB VRAM)
#
# GPU 0: Gemma-4-E4B-Arabic-It (4.5B Total / 4.0B Active Dense)
# GPU 1: Gemma-4-26B-A4B-Arabic-It (25.2B Total / 3.8B Active Sparse MoE)
# ==============================================================================

set -euo pipefail

export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export DATASETS_VERBOSITY=warning
export PYTHONWARNINGS="ignore::SyntaxWarning,ignore::FutureWarning"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p logs outputs/eval_reports assets/plots

echo "======================================================================"
echo "[Phase 1/4] Building 120k SFT (75/25) & 20k DPO Datasets Once on Disk"
echo "======================================================================"
if [[ -f "data/sft_mixed/dataset_info.json" || -f "data/sft_mixed/state.json" ]]; then
  echo "[Skip] data/sft_mixed already built on disk."
else
  "${PYTHON_BIN}" scripts/build_sft_dataset.py \
    --config configs/sft_26b_a4b_a100.yaml \
    --output-dir data/sft_mixed \
    --max-samples 120000
fi

if [[ -f "data/dpo_mixed/dataset_info.json" || -f "data/dpo_mixed/state.json" ]]; then
  echo "[Skip] data/dpo_mixed already built on disk."
else
  "${PYTHON_BIN}" scripts/build_dpo_dataset.py \
    --config configs/dpo_26b_a4b_a100.yaml \
    --output-dir data/dpo_mixed
fi

echo "======================================================================"
echo "[Phase 2/4] Launching Simultaneous Dual-GPU Training (GPU 0 + GPU 1)"
echo "  - GPU 0: Gemma-4-E4B-Arabic-It      -> logs/gpu0_e4b.log"
echo "  - GPU 1: Gemma-4-26B-A4B-Arabic-It  -> logs/gpu1_26b_a4b.log"
echo "======================================================================"

(
  export CUDA_VISIBLE_DEVICES=0
  echo "[GPU 0] Starting Stage 1 rsLoRA SFT for Gemma-4-E4B-Arabic-It..."
  "${PYTHON_BIN}" src/train_sft.py \
    --config configs/sft_e4b_a100.yaml \
    --dataset-dir data/sft_mixed \
    --output-dir outputs/sft_e4b_a100

  echo "[GPU 0] Starting Stage 2 Trust-Region DPO for Gemma-4-E4B-Arabic-It..."
  "${PYTHON_BIN}" src/train_dpo.py \
    --config configs/dpo_e4b_a100.yaml \
    --dataset-dir data/dpo_mixed \
    --sft-adapter-dir outputs/sft_e4b_a100 \
    --output-dir outputs/dpo_e4b_a100

  echo "[GPU 0] Running TIES-Merging (lambda=0.85, density=0.85) -> merged_16bit..."
  "${PYTHON_BIN}" src/merge_and_export.py \
    --base-model unsloth/gemma-4-E4B-it \
    --sft-adapter-dir outputs/sft_e4b_a100 \
    --adapter-dir outputs/dpo_e4b_a100 \
    --output-dir outputs/merged_gemma4_e4b_arabic \
    --export-format merged_16bit
) 2>&1 | tee logs/gpu0_e4b.log &
PID_GPU0=$!

(
  export CUDA_VISIBLE_DEVICES=1
  echo "[GPU 1] Starting Stage 1 rsLoRA SFT for Gemma-4-26B-A4B-Arabic-It (Frozen MoE Router)..."
  "${PYTHON_BIN}" src/train_sft.py \
    --config configs/sft_26b_a4b_a100.yaml \
    --dataset-dir data/sft_mixed \
    --output-dir outputs/sft_26b_a4b_a100

  echo "[GPU 1] Starting Stage 2 Trust-Region DPO for Gemma-4-26B-A4B-Arabic-It..."
  "${PYTHON_BIN}" src/train_dpo.py \
    --config configs/dpo_26b_a4b_a100.yaml \
    --dataset-dir data/dpo_mixed \
    --sft-adapter-dir outputs/sft_26b_a4b_a100 \
    --output-dir outputs/dpo_26b_a4b_a100

  echo "[GPU 1] Running TIES-Merging (lambda=0.85, density=0.85) -> merged_16bit..."
  "${PYTHON_BIN}" src/merge_and_export.py \
    --base-model unsloth/gemma-4-26B-A4B-it \
    --sft-adapter-dir outputs/sft_26b_a4b_a100 \
    --adapter-dir outputs/dpo_26b_a4b_a100 \
    --output-dir outputs/merged_gemma4_26b_a4b_arabic \
    --export-format merged_16bit
) 2>&1 | tee logs/gpu1_26b_a4b.log &
PID_GPU1=$!

echo "Waiting for GPU 0 (PID ${PID_GPU0}) and GPU 1 (PID ${PID_GPU1}) to complete..."
wait "${PID_GPU0}"
wait "${PID_GPU1}"

echo "======================================================================"
echo "[Phase 3/4] Evaluating Both Models on 7 Arabic + 4 Retention Benchmarks"
echo "======================================================================"
"${PYTHON_BIN}" scripts/run_benchmarks.py \
  --config configs/eval_config.yaml \
  --model-key both \
  --limit 50

echo "======================================================================"
echo "[Phase 4/4] Running Iso-Active Latency & Single-GPU vLLM (1-50 Users) Benchmarks"
echo "======================================================================"
"${PYTHON_BIN}" scripts/benchmark_latency_and_plots.py \
  --config configs/eval_config.yaml \
  --live-profile

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/benchmark_vllm_concurrency.py \
  --config configs/eval_config.yaml \
  --live-vllm

echo "All 4 phases completed!"
