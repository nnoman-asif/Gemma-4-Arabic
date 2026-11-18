#!/usr/bin/env bash
# ==============================================================================
# End-to-End Dual-GPU Training, Lossless Merge, Parallel Base-vs-Tuned Eval,
# Latency Profiling & Sync Bundle Orchestrator
#
# Target Hardware: 2x NVIDIA A100 80 GB SXM4/PCIe (or 2x H100 80 GB)
#
# Execution Topology (Zero Race Conditions, Maximum 2-GPU Efficiency):
#   Phase 0: Upfront CUDA, Disk, .env & Dependency Verification (peft>=0.21.3,
#            unsloth, trl==1.15.0, transformers==5.17.0, vllm, lm-eval,
#            inspect-evals, lighteval@main) BEFORE spending GPU training hours.
#   Phase 1: Pull V3 High-Purity 30k SFT + 6k DPO Datasets from HF Hub.
#   Phase 2A: Simultaneous Dual-GPU Training:
#               - GPU 0: Gemma-4-E4B-Arabic-It      (Stage 1 SFT -> Stage 2 DPO)
#               - GPU 1: Gemma-4-26B-A4B-Arabic-It  (Stage 1 SFT -> Stage 2 DPO)
#             Hard barrier waits until BOTH models complete training.
#   Phase 2B: Lossless 16-bit Adapter Merging (lambda=1.0) -> merged_16bit.
#             Hard barrier verifies BOTH merged_16bit checkpoints on disk
#             before ANY benchmark begins.
#   Phase 3: Parallel Base-vs-Tuned Full-Dataset Benchmarking (10 Suites):
#               - Prewarm all 10 benchmark datasets in one foreground process.
#               - Step 3A (E4B):     GPU 0 = Base E4B     || GPU 1 = Tuned E4B
#                                    -> Hard wait -> Combine E4B report.
#               - Step 3B (26B-A4B): GPU 0 = Base 26B-A4B || GPU 1 = Tuned 26B-A4B
#                                    -> Hard wait -> Combine 26B-A4B & Family reports.
#   Phase 4: Iso-Active Latency Profiling, 50-User vLLM Concurrency & Sync Tarball.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Auto-load .env (HF_TOKEN, etc.) if present in repository root
if [[ -f ".env" ]]; then
  echo "[Init] Loading environment variables from .env..."
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export DATASETS_VERBOSITY=warning
export PYTHONWARNINGS="ignore::SyntaxWarning,ignore::FutureWarning"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UNSLOTH_BLOCK_COMPILE="0"
export UNSLOTH_MOE_LORA_FOLD="0"
export UNSLOTH_MOE_GC_KEEP_BASE="0"
export HF_ALLOW_CODE_EVAL="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export PYTHONHASHSEED="0"

PYTHON_BIN="${PYTHON_BIN:-python3}"

ACTIVE_PIDS=()

terminate_pid_tree() {
  local target_pid="$1"
  if [[ -n "${target_pid}" ]] && kill -0 "${target_pid}" 2>/dev/null; then
    local proc_comm
    proc_comm="$(ps -o comm= -p "${target_pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "${proc_comm}" == "tee" ]]; then
      return 0
    fi
    local child_pids
    child_pids="$(ps -o pid= --ppid "${target_pid}" 2>/dev/null || pgrep -P "${target_pid}" 2>/dev/null || true)"
    for cpid in ${child_pids}; do
      terminate_pid_tree "${cpid}"
    done
    kill -TERM "${target_pid}" 2>/dev/null || true
  fi
}

cleanup_on_signal() {
  echo ""
  echo "[Trap] Interrupt/termination signal received! Stopping active GPU workers cleanly so saved checkpoints stay intact..."
  if [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; then
    for pid in "${ACTIVE_PIDS[@]}"; do
      terminate_pid_tree "${pid}"
    done
  fi
  sleep 2
  exit 130
}
trap cleanup_on_signal INT TERM

mkdir -p logs outputs/eval_reports assets/plots

if [[ ! -f "outputs/.v3_pipeline_migrated" ]]; then
  echo "======================================================================"
  echo "[V3 Migration] Cleaning stale V2 checkpoints/merges while preserving verified E4B Base retention scores"
  echo "======================================================================"
  rm -rf \
    data/sft_mixed \
    data/dpo_mixed \
    outputs/sft_e4b_a100 \
    outputs/dpo_e4b_a100 \
    outputs/sft_26b_a4b_a100 \
    outputs/dpo_26b_a4b_a100 \
    outputs/merged_gemma4_e4b_arabic \
    outputs/merged_gemma4_26b_a4b_arabic \
    unsloth_compiled_cache \
    unsloth_compiled_cache_gpu0 \
    unsloth_compiled_cache_gpu1
  "${PYTHON_BIN}" -c "import json, os, pathlib; rep = pathlib.Path('outputs/eval_reports'); e4b_base = rep / 'live_raw_e4b_base.json';
if e4b_base.is_file():
    raw = json.loads(e4b_base.read_text(encoding='utf-8'))
    if isinstance(raw.get('base'), dict):
        kept_ret = list((raw['base'].get('retention') or {}).keys())
        raw['base']['arabic'] = {}
        tmp = e4b_base.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(str(tmp), str(e4b_base))
        print(f'[V3 Migration] Preserved E4B Base retention suites {kept_ret}; cleared stale V2 E4B Base Arabic cache.')
for stale in ('live_raw_e4b_tuned.json', 'live_raw_e4b.json', 'live_raw_26b_a4b_base.json', 'live_raw_26b_a4b_tuned.json', 'live_raw_26b_a4b.json', 'benchmark_results.json', 'benchmark_results_e4b.json', 'benchmark_results_26b_a4b.json', 'benchmark_comparison.md', 'benchmark_comparison_e4b.md', 'benchmark_comparison_26b_a4b.md', 'family_benchmark_comparison.json', 'family_benchmark_comparison.md', 'preflight_gate_report.json'):
    fp = rep / stale
    if fp.is_file():
        fp.unlink()"
  touch outputs/.v3_pipeline_migrated
fi

echo "======================================================================"
echo "[Phase 0/4] Upfront Hardware, CUDA Driver, Disk & Dependency Verification"
echo "======================================================================"

if ! command -v git >/dev/null 2>&1; then
  echo "[FATAL] 'git' binary is required (for lighteval@main installation) but was not found in PATH."
  exit 1
fi

"${PYTHON_BIN}" -c "import sys; print(f'[Python] {sys.version.split()[0]}'); assert sys.version_info >= (3, 11), f'Python >= 3.11 is required by vllm>=0.30.0 (found {sys.version.split()[0]}). Please select a Python 3.11/3.12 container.'"

"${PYTHON_BIN}" -c "import ctypes, re, subprocess, torch; n = torch.cuda.device_count() if torch.cuda.is_available() else 0; print(f'[Hardware] Detected {n} CUDA GPUs: {[torch.cuda.get_device_name(i) for i in range(n)]}'); assert n >= 2, f'Expected >= 2 CUDA GPUs for dual-GPU pipeline, found {n}.'; drv = 0;
try:
    v = ctypes.c_int(); ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(v)); drv = v.value
except Exception:
    out = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT); m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', out); drv = (int(m.group(1)) * 1000 + int(m.group(2)) * 10) if m else 0
maj, minr = drv // 1000, (drv % 1000) // 10; print(f'[CUDA Driver] Host NVIDIA Driver CUDA capability: {maj}.{minr} (raw={drv})'); has_vllm = False
try:
    import vllm; from packaging.version import Version; has_vllm = Version(vllm.__version__) >= Version('0.30.0')
except Exception:
    has_vllm = False
assert has_vllm or drv >= 13000, f'[FATAL] Host NVIDIA driver only supports CUDA {maj}.{minr} (< 13.0), whereas PyPI vllm>=0.30.0 (required for Gemma 4) pulls torch+cu13 requiring Host CUDA >= 13.0 (NVIDIA Driver >= 580). Stop this instance now and rent a node with CUDA >= 13.0 / Driver >= 580!'"

"${PYTHON_BIN}" -c "import os, pathlib, shutil; free_gb = shutil.disk_usage('.').free / (1024 ** 3); gib = float(1024 ** 3); hf_dir = pathlib.Path(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))); data_dir = pathlib.Path('data'); cached_bytes = sum(f.stat().st_size for d in (hf_dir, data_dir) if d.is_dir() for f in d.rglob('*') if f.is_file() and not f.is_symlink()); cached_gb = min(70.0, cached_bytes / gib); e4b_dir = pathlib.Path('outputs/merged_gemma4_e4b_arabic'); moe_dir = pathlib.Path('outputs/merged_gemma4_26b_a4b_arabic'); e4b_ok = (e4b_dir / 'merge_manifest.json').is_file() and sum(f.stat().st_size for f in (e4b_dir / 'merged_16bit').glob('*.safetensors')) / gib >= 7.8 if (e4b_dir / 'merged_16bit').is_dir() else False; moe_ok = (moe_dir / 'merge_manifest.json').is_file() and sum(f.stat().st_size for f in (moe_dir / 'merged_16bit').glob('*.safetensors')) / gib >= 44.0 if (moe_dir / 'merged_16bit').is_dir() else False; req_gb = 20.0 if (e4b_ok and moe_ok) else (max(65.0, 90.0 - cached_gb * 0.35) if e4b_ok else max(110.0, 180.0 - cached_gb)); print(f'[Disk] Free workspace disk space: {free_gb:.1f} GB | Already cached HF+data: {cached_gb:.1f} GB (required for remaining stages: >= {req_gb:.1f} GB)'); assert free_gb >= req_gb, f'Insufficient free disk space ({free_gb:.1f} GB < {req_gb:.1f} GB required). Increase container disk to >= 250 GB.'"

if ! "${PYTHON_BIN}" -c "import vllm; from packaging.version import Version; assert Version(vllm.__version__) >= Version('0.30.0')" >/dev/null 2>&1; then
  echo "[Setup] Installing vLLM (>=0.30.0)..."
  "${PYTHON_BIN}" -m pip install -q "vllm>=0.30.0"
fi

if ! "${PYTHON_BIN}" -c "import lm_eval, evaluate, sqlitedict, pytablewriter, sacrebleu, rouge_score, matplotlib, hf_xet; from packaging.version import Version; assert Version(lm_eval.__version__) >= Version('0.4.13')" >/dev/null 2>&1; then
  echo "[Setup] Installing lm-eval>=0.4.13, matplotlib>=3.8.0, hf-xet>=1.1.8, and evaluation metric packages..."
  "${PYTHON_BIN}" -m pip install -q "lm-eval>=0.4.13" "evaluate>=0.4.3" "sqlitedict>=2.1.0" "pytablewriter>=1.2.0" "sacrebleu>=2.4.0" "rouge-score>=0.1.2" "matplotlib>=3.8.0" "hf-xet>=1.1.8" "hf-transfer>=0.1.8"
fi

if ! "${PYTHON_BIN}" -c "import lighteval, inspect_ai, inspect_evals, ray, more_itertools; from lighteval.tasks.tasks.lcb import main as _lcb; from packaging.version import Version; assert Version(inspect_ai.__version__) >= Version('0.3.290')" >/dev/null 2>&1; then
  echo "[Setup] Installing inspect-ai>=0.3.290, inspect-evals>=0.29.0, ray, and lighteval@main..."
  "${PYTHON_BIN}" -m pip install -q "inspect-ai>=0.3.290" "inspect-evals>=0.29.0" "datasets>=4.8.5,<5.0.0" "ray" "aenum==3.1.15" "termcolor>=2.3.0" "colorlog" "pycountry" "langcodes" "latex2sympy2_extended==1.0.6" "more_itertools" "nltk" "scikit-learn" "GitPython>=3.1.41" "typer>=0.12.0"
  "${PYTHON_BIN}" -m pip install -q --no-deps --force-reinstall "git+https://github.com/huggingface/lighteval.git@main"
fi

if ! "${PYTHON_BIN}" -c "import unsloth, unsloth_zoo; from packaging.version import Version; assert Version(unsloth.__version__) >= Version('2026.4.1')" >/dev/null 2>&1; then
  echo "[Setup] Installing/upgrading Unsloth (>=2026.4.1)..."
  "${PYTHON_BIN}" -m pip install -q --upgrade "unsloth>=2026.4.1" "unsloth_zoo>=2026.4.1"
fi

if ! "${PYTHON_BIN}" -c "import transformers, peft, trl, datasets, bitsandbytes; from packaging.version import Version; assert Version('5.5.0') <= Version(transformers.__version__) <= Version('5.17.0'); assert Version(peft.__version__) >= Version('0.21.3'); assert Version('1.15.0') <= Version(trl.__version__) <= Version('1.15.99'); assert Version('4.8.5') <= Version(datasets.__version__) < Version('5.0.0'); assert Version(bitsandbytes.__version__) >= Version('0.49.0')" >/dev/null 2>&1; then
  echo "[Setup] Aligning transformers==5.17.0, peft>=0.21.3, trl==1.15.0, datasets==4.8.5, and bitsandbytes>=0.49.0..."
  "${PYTHON_BIN}" -m pip install -q "transformers==5.17.0" "peft>=0.21.3" "trl==1.15.0" "datasets>=4.8.5,<5.0.0" "bitsandbytes>=0.49.0" "accelerate>=1.6.0"
fi

"${PYTHON_BIN}" -c "import bitsandbytes as bnb, hf_xet, inspect_ai, inspect_evals, lighteval, lm_eval, matplotlib, more_itertools, peft, ray, torch, transformers, trl, unsloth, vllm; from transformers import Gemma4ForConditionalGeneration; from packaging.version import Version; assert Version(vllm.__version__) >= Version('0.30.0'), f'vllm {vllm.__version__} < 0.30.0'; assert Version(unsloth.__version__) >= Version('2026.4.1'), f'unsloth {unsloth.__version__} < 2026.4.1'; assert Version(peft.__version__) >= Version('0.21.3'), f'peft {peft.__version__} < 0.21.3'; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Post-install CUDA check failed! Host driver is incompatible with installed PyTorch CUDA runtime.'; [((torch.ones((64, 64), device=f'cuda:{i}', dtype=torch.bfloat16) @ torch.ones((64, 64), device=f'cuda:{i}', dtype=torch.bfloat16)).sum().item()) for i in (0, 1)]; q_lin = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type='nf4').to('cuda:0'); _ = q_lin(torch.ones((2, 64), device='cuda:0', dtype=torch.bfloat16)).sum().item(); del q_lin; torch.cuda.empty_cache(); print(f'[Verified] GPUs=2xBF16+NF4 OK | torch={torch.__version__} | bnb={bnb.__version__} | unsloth={unsloth.__version__} | transformers={transformers.__version__} | peft={peft.__version__} | trl={trl.__version__} | vllm={vllm.__version__} | lm_eval={lm_eval.__version__}')"

if "${PYTHON_BIN}" -c "import pathlib, sys; gib = float(1024 ** 3); e4b = pathlib.Path('outputs/merged_gemma4_e4b_arabic'); moe = pathlib.Path('outputs/merged_gemma4_26b_a4b_arabic'); ok = all((d / 'merged_16bit' / 'config.json').is_file() and (d / 'merged_16bit' / 'chat_template.jinja').is_file() and (d / 'merge_manifest.json').is_file() for d in (e4b, moe)) and sum(f.stat().st_size for f in (e4b / 'merged_16bit').glob('*.safetensors')) / gib >= 7.8 and sum(f.stat().st_size for f in (moe / 'merged_16bit').glob('*.safetensors')) / gib >= 44.0; sys.exit(0 if ok else 1)" >/dev/null 2>&1; then
  echo "======================================================================"
  echo "[Skip] Both merged_16bit checkpoints & manifests already verified on disk:"
  echo "  - outputs/merged_gemma4_e4b_arabic/merged_16bit"
  echo "  - outputs/merged_gemma4_26b_a4b_arabic/merged_16bit"
  echo "Skipping Phase 1 (Dataset Pull) and Phase 2 (Training & Merging)."
  echo "======================================================================"
else
  echo "======================================================================"
  echo "[Phase 1/4] Preparing De-contaminated 30k SFT & 6k DPO V3 Datasets"
  echo "======================================================================"

  if "${PYTHON_BIN}" -c "import sys; from datasets import load_from_disk; ds = load_from_disk('data/sft_mixed'); sys.exit(0 if 28000 <= len(ds) <= 35000 else 1)" >/dev/null 2>&1; then
    echo "[Skip] Verified intact V3 data/sft_mixed on disk."
  else
    rm -rf data/sft_mixed
    echo "[Dataset] Pulling pre-built 30k SFT V3 dataset from Hugging Face Hub..."
    "${PYTHON_BIN}" scripts/build_sft_dataset.py \
      --config configs/sft_26b_a4b_a100.yaml \
      --output-dir data/sft_mixed \
      --pull-from-hub || \
    "${PYTHON_BIN}" scripts/build_sft_dataset.py \
      --config configs/sft_26b_a4b_a100.yaml \
      --output-dir data/sft_mixed \
      --max-samples 30000
  fi

  if "${PYTHON_BIN}" -c "import sys; from datasets import load_from_disk; ds = load_from_disk('data/dpo_mixed'); sys.exit(0 if 5500 <= len(ds) <= 7000 else 1)" >/dev/null 2>&1; then
    echo "[Skip] Verified intact V3 data/dpo_mixed on disk."
  else
    rm -rf data/dpo_mixed
    echo "[Dataset] Pulling pre-built 6k DPO V3 dataset from Hugging Face Hub..."
    "${PYTHON_BIN}" scripts/build_dpo_dataset.py \
      --config configs/dpo_26b_a4b_a100.yaml \
      --output-dir data/dpo_mixed \
      --pull-from-hub || \
    "${PYTHON_BIN}" scripts/build_dpo_dataset.py \
      --config configs/dpo_26b_a4b_a100.yaml \
      --output-dir data/dpo_mixed
  fi

  "${PYTHON_BIN}" -c "from datasets import load_from_disk; sft = load_from_disk('data/sft_mixed'); dpo = load_from_disk('data/dpo_mixed'); print(f'[Verified V3 Datasets] SFT rows={len(sft)} | DPO pairs={len(dpo)}'); assert 28000 <= len(sft) <= 35000, f'Unexpected V3 SFT count: {len(sft)}'; assert 5500 <= len(dpo) <= 7000, f'Unexpected V3 DPO count: {len(dpo)}'; assert isinstance(dpo[0].get('prompt_messages') or dpo[0].get('prompt'), list), 'DPO dataset must include conversational prompt_messages list'; assert '<start_of_turn>' not in sft[0]['text'], 'Legacy Gemma 2/3 tags detected in SFT'"

  echo "======================================================================"
  echo "[Phase 2A/4] Simultaneous Dual-GPU Training (SFT -> DPO, save_steps=100)"
  echo "  - GPU 0: Gemma-4-E4B-Arabic-It      (r=16, alpha=16)          -> logs/train_gpu0_e4b.log"
  echo "  - GPU 1: Gemma-4-26B-A4B-Arabic-It  (r=16, alpha=16, 128 exp) -> logs/train_gpu1_26b_a4b.log"
  echo "======================================================================"

  rm -rf unsloth_compiled_cache unsloth_compiled_cache_gpu0 unsloth_compiled_cache_gpu1

  (
    exec > >(tee -a logs/train_gpu0_e4b.log) 2>&1
    export CUDA_VISIBLE_DEVICES=0
    export UNSLOTH_COMPILE_LOCATION="unsloth_compiled_cache_gpu0"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_train_gpu0"
    export TRITON_CACHE_DIR="/tmp/triton_train_gpu0"

    if [[ -f "outputs/sft_e4b_a100/adapter_config.json" && -s "outputs/sft_e4b_a100/adapter_model.safetensors" && -f "outputs/sft_e4b_a100/training_metrics.json" ]]; then
      echo "[GPU 0] [Skip] Verified completed Stage 1 SFT adapter at outputs/sft_e4b_a100."
    else
      echo "[GPU 0] Starting/resuming Stage 1 QLoRA SFT for Gemma-4-E4B-Arabic-It (r=16, alpha=16)..."
      "${PYTHON_BIN}" src/train_sft.py \
        --config configs/sft_e4b_a100.yaml \
        --dataset-dir data/sft_mixed \
        --output-dir outputs/sft_e4b_a100
    fi

    if [[ -f "outputs/dpo_e4b_a100/adapter_config.json" && -s "outputs/dpo_e4b_a100/adapter_model.safetensors" && -f "outputs/dpo_e4b_a100/dpo_metrics.json" ]]; then
      echo "[GPU 0] [Skip] Verified completed Stage 2 DPO adapter at outputs/dpo_e4b_a100."
    else
      echo "[GPU 0] Starting/resuming Stage 2 DPO for Gemma-4-E4B-Arabic-It..."
      "${PYTHON_BIN}" src/train_dpo.py \
        --config configs/dpo_e4b_a100.yaml \
        --dataset-dir data/dpo_mixed \
        --sft-adapter-dir outputs/sft_e4b_a100 \
        --output-dir outputs/dpo_e4b_a100
    fi
    echo "[GPU 0] E4B Training (SFT + DPO) COMPLETED."
  ) &
  PID_TRAIN_E4B=$!

  (
    exec > >(tee -a logs/train_gpu1_26b_a4b.log) 2>&1
    export CUDA_VISIBLE_DEVICES=1
    export UNSLOTH_COMPILE_LOCATION="unsloth_compiled_cache_gpu1"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_train_gpu1"
    export TRITON_CACHE_DIR="/tmp/triton_train_gpu1"

    if [[ -f "outputs/sft_26b_a4b_a100/adapter_config.json" && -s "outputs/sft_26b_a4b_a100/adapter_model.safetensors" && -f "outputs/sft_26b_a4b_a100/training_metrics.json" ]]; then
      echo "[GPU 1] [Skip] Verified completed Stage 1 SFT adapter at outputs/sft_26b_a4b_a100."
    else
      echo "[GPU 1] Starting/resuming Stage 1 QLoRA SFT for Gemma-4-26B-A4B-Arabic-It (r=16, alpha=16, 128 3D Experts, Frozen Router)..."
      "${PYTHON_BIN}" src/train_sft.py \
        --config configs/sft_26b_a4b_a100.yaml \
        --dataset-dir data/sft_mixed \
        --output-dir outputs/sft_26b_a4b_a100
    fi

    if [[ -f "outputs/dpo_26b_a4b_a100/adapter_config.json" && -s "outputs/dpo_26b_a4b_a100/adapter_model.safetensors" && -f "outputs/dpo_26b_a4b_a100/dpo_metrics.json" ]]; then
      echo "[GPU 1] [Skip] Verified completed Stage 2 DPO adapter at outputs/dpo_26b_a4b_a100."
    else
      echo "[GPU 1] Starting/resuming Stage 2 DPO for Gemma-4-26B-A4B-Arabic-It..."
      rm -rf unsloth_compiled_cache_gpu1 /tmp/inductor_train_gpu1 /tmp/triton_train_gpu1
      "${PYTHON_BIN}" src/train_dpo.py \
        --config configs/dpo_26b_a4b_a100.yaml \
        --dataset-dir data/dpo_mixed \
        --sft-adapter-dir outputs/sft_26b_a4b_a100 \
        --output-dir outputs/dpo_26b_a4b_a100
    fi
    echo "[GPU 1] 26B-A4B Training (SFT + DPO) COMPLETED."
  ) &
  PID_TRAIN_26B=$!
  ACTIVE_PIDS=("${PID_TRAIN_E4B}" "${PID_TRAIN_26B}")

  echo "[Barrier] Waiting for BOTH GPU 0 (E4B PID=${PID_TRAIN_E4B}) and GPU 1 (26B-A4B PID=${PID_TRAIN_26B}) training to finish..."
  RC_TRAIN_E4B=0
  RC_TRAIN_26B=0
  wait "${PID_TRAIN_E4B}" || RC_TRAIN_E4B=$?
  wait "${PID_TRAIN_26B}" || RC_TRAIN_26B=$?
  ACTIVE_PIDS=()
  sleep 2

  if [[ "${RC_TRAIN_E4B}" -ne 0 || "${RC_TRAIN_26B}" -ne 0 ]]; then
    echo "[FATAL] Training barrier failed (RC_E4B=${RC_TRAIN_E4B}, RC_26B_A4B=${RC_TRAIN_26B}). Aborting before merge/eval."
    exit 1
  fi

  if [[ ! -f "outputs/dpo_e4b_a100/adapter_config.json" || ! -s "outputs/dpo_e4b_a100/adapter_model.safetensors" || ! -f "outputs/dpo_e4b_a100/dpo_metrics.json" || ! -f "outputs/dpo_26b_a4b_a100/adapter_config.json" || ! -s "outputs/dpo_26b_a4b_a100/adapter_model.safetensors" || ! -f "outputs/dpo_26b_a4b_a100/dpo_metrics.json" ]]; then
    echo "[FATAL] Incomplete Stage 2 DPO adapter artifacts detected after training barrier!"
    exit 1
  fi

  echo "======================================================================"
  echo "[Phase 2B/4] Lossless 16-Bit Adapter Merging (lambda=1.0 -> merged_16bit)"
  echo "======================================================================"

  HOST_RAM_GB="$("${PYTHON_BIN}" -c "import os, pathlib; names = getattr(os, 'sysconf_names', {}); pages = os.sysconf('SC_PHYS_PAGES') if 'SC_PHYS_PAGES' in names else 0; psize = os.sysconf('SC_PAGE_SIZE') if 'SC_PAGE_SIZE' in names else 4096; cands = [pages * psize] if pages > 0 else [];
for cg in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    p = pathlib.Path(cg)
    if p.is_file():
        raw = p.read_text(encoding='utf-8', errors='ignore').strip()
        if raw.isdigit() and 0 < int(raw) < (1 << 60):
            cands.append(int(raw))
print(int(min(cands) / (1024 ** 3)) if cands else 0)" 2>/dev/null || echo 0)"
  echo "[Container RAM] Effective available CPU RAM (min of host & Docker cgroup limit): ${HOST_RAM_GB} GB."

  run_merge_e4b() {
    export CUDA_VISIBLE_DEVICES=0
    export UNSLOTH_COMPILE_LOCATION="unsloth_compiled_cache_gpu0"
    echo "[GPU 0] Merging/verifying Gemma-4-E4B-Arabic-It adapter (lambda=1.0) -> outputs/merged_gemma4_e4b_arabic/merged_16bit..."
    "${PYTHON_BIN}" src/merge_and_export.py \
      --base-model unsloth/gemma-4-E4B-it \
      --adapter-dir outputs/dpo_e4b_a100 \
      --output-dir outputs/merged_gemma4_e4b_arabic \
      --lambda-scale 1.0 \
      --export-format merged_16bit
  }

  run_merge_26b() {
    export CUDA_VISIBLE_DEVICES=1
    export UNSLOTH_COMPILE_LOCATION="unsloth_compiled_cache_gpu1"
    echo "[GPU 1] Merging/verifying Gemma-4-26B-A4B-Arabic-It adapter (lambda=1.0, 128 3D experts) -> outputs/merged_gemma4_26b_a4b_arabic/merged_16bit..."
    "${PYTHON_BIN}" src/merge_and_export.py \
      --base-model unsloth/gemma-4-26B-A4B-it \
      --adapter-dir outputs/dpo_26b_a4b_a100 \
      --output-dir outputs/merged_gemma4_26b_a4b_arabic \
      --lambda-scale 1.0 \
      --export-format merged_16bit
  }

  if [[ "${HOST_RAM_GB}" -ge 140 ]]; then
    echo "[Merge] Effective Container RAM >= 140 GB: running E4B (GPU 0) and 26B-A4B (GPU 1) merges simultaneously..."
    ( exec > >(tee -a logs/merge_gpu0_e4b.log) 2>&1; run_merge_e4b ) &
    PID_MERGE_E4B=$!
    ( exec > >(tee -a logs/merge_gpu1_26b_a4b.log) 2>&1; run_merge_26b ) &
    PID_MERGE_26B=$!
    ACTIVE_PIDS=("${PID_MERGE_E4B}" "${PID_MERGE_26B}")
    RC_MERGE_E4B=0
    RC_MERGE_26B=0
    wait "${PID_MERGE_E4B}" || RC_MERGE_E4B=$?
    wait "${PID_MERGE_26B}" || RC_MERGE_26B=$?
    ACTIVE_PIDS=()
    sleep 2
    if [[ "${RC_MERGE_E4B}" -ne 0 || "${RC_MERGE_26B}" -ne 0 ]]; then
      echo "[FATAL] Parallel merge failed (RC_E4B=${RC_MERGE_E4B}, RC_26B_A4B=${RC_MERGE_26B})."
      exit 1
    fi
  else
    echo "[Merge] Effective Container RAM < 140 GB: running merges sequentially to prevent container cgroup OOM-kill..."
    ( exec > >(tee -a logs/merge_gpu0_e4b.log) 2>&1; run_merge_e4b )
    ( exec > >(tee -a logs/merge_gpu1_26b_a4b.log) 2>&1; run_merge_26b )
    sleep 2
  fi
fi

# Hard Merge Verification Barrier before ANY benchmark begins
for REQ_FILE in \
  "outputs/merged_gemma4_e4b_arabic/merged_16bit/config.json" \
  "outputs/merged_gemma4_e4b_arabic/merged_16bit/chat_template.jinja" \
  "outputs/merged_gemma4_e4b_arabic/merge_manifest.json" \
  "outputs/merged_gemma4_26b_a4b_arabic/merged_16bit/config.json" \
  "outputs/merged_gemma4_26b_a4b_arabic/merged_16bit/chat_template.jinja" \
  "outputs/merged_gemma4_26b_a4b_arabic/merge_manifest.json"; do
  if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[FATAL] Hard merge barrier failed: required artifact '${REQ_FILE}' is missing!"
    exit 1
  fi
done
echo "[Barrier Passed] Both E4B and 26B-A4B merged_16bit checkpoints are verified on disk."

"${PYTHON_BIN}" -c "import json, pathlib; rep = pathlib.Path('outputs/eval_reports'); [fp.unlink() for fp in rep.glob('*.json') if fp.is_file() and json.loads(fp.read_text(encoding='utf-8')).get('mode') in ('dry_run', 'reference')]" 2>/dev/null || true

echo "======================================================================"
echo "[Phase 3/4] Parallel Base-vs-Tuned Full-Dataset Benchmarking (10 Suites)"
echo "  Step 3.0: Prewarming all 10 benchmark datasets once on disk"
echo "======================================================================"

"${PYTHON_BIN}" scripts/run_benchmarks.py \
  --config configs/eval_config.yaml \
  --prewarm-datasets

echo "======================================================================"
echo "[Phase 2C/4] Mandatory 5-Minute Dual-GPU Pre-Flight Gate (Arabic Gain + Retention)"
echo "  - Verifies Tuned >= Base on ArabicMMLU + AlGhafa and near-zero regression on MMLU-Pro"
echo "  - Auto-falls back to --lambda-scale 0.70 --force if needed; aborts fast before 90m eval on failure"
echo "======================================================================"

"${PYTHON_BIN}" scripts/run_benchmarks.py \
  --config configs/eval_config.yaml \
  --model-key both \
  --preflight-gate \
  --use-vllm

echo "======================================================================"
echo "[Phase 3A/4] Benchmarking Gemma-4-E4B Family in Parallel across 2 GPUs"
echo "  - GPU 0: Base E4B   (unsloth/gemma-4-E4B-it)                        -> logs/eval_e4b_base_gpu0.log"
echo "  - GPU 1: Tuned E4B  (outputs/merged_gemma4_e4b_arabic/merged_16bit) -> logs/eval_e4b_tuned_gpu1.log"
echo "======================================================================"

(
  exec > >(tee -a logs/eval_e4b_base_gpu0.log) 2>&1
  export CUDA_VISIBLE_DEVICES=0
  export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_eval_e4b_gpu0"
  export TRITON_CACHE_DIR="/tmp/triton_eval_e4b_gpu0"
  export VLLM_CACHE_ROOT="/tmp/vllm_eval_e4b_gpu0"
  "${PYTHON_BIN}" scripts/run_benchmarks.py \
    --config configs/eval_config.yaml \
    --model-key e4b \
    --only-role base \
    --use-cached-eval \
    --use-vllm \
    --apply-chat-template \
    --no-reference-fallback
) &
PID_EVAL_E4B_BASE=$!

(
  exec > >(tee -a logs/eval_e4b_tuned_gpu1.log) 2>&1
  export CUDA_VISIBLE_DEVICES=1
  export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_eval_e4b_gpu1"
  export TRITON_CACHE_DIR="/tmp/triton_eval_e4b_gpu1"
  export VLLM_CACHE_ROOT="/tmp/vllm_eval_e4b_gpu1"
  "${PYTHON_BIN}" scripts/run_benchmarks.py \
    --config configs/eval_config.yaml \
    --model-key e4b \
    --only-role tuned \
    --model-path outputs/merged_gemma4_e4b_arabic/merged_16bit \
    --use-cached-eval \
    --use-vllm \
    --apply-chat-template \
    --no-reference-fallback
) &
PID_EVAL_E4B_TUNED=$!
ACTIVE_PIDS=("${PID_EVAL_E4B_BASE}" "${PID_EVAL_E4B_TUNED}")

echo "[Barrier] Waiting for E4B Base (GPU 0, PID=${PID_EVAL_E4B_BASE}) and E4B Tuned (GPU 1, PID=${PID_EVAL_E4B_TUNED})..."
RC_EVAL_E4B_BASE=0
RC_EVAL_E4B_TUNED=0
wait "${PID_EVAL_E4B_BASE}" || RC_EVAL_E4B_BASE=$?
wait "${PID_EVAL_E4B_TUNED}" || RC_EVAL_E4B_TUNED=$?
ACTIVE_PIDS=()
sleep 2

if [[ "${RC_EVAL_E4B_BASE}" -ne 0 || "${RC_EVAL_E4B_TUNED}" -ne 0 ]]; then
  echo "[FATAL] E4B parallel benchmark failed (RC_BASE_GPU0=${RC_EVAL_E4B_BASE}, RC_TUNED_GPU1=${RC_EVAL_E4B_TUNED})."
  exit 1
fi

"${PYTHON_BIN}" scripts/run_benchmarks.py \
  --config configs/eval_config.yaml \
  --model-key e4b \
  --combine-role-reports \
  --output-md outputs/eval_reports/benchmark_comparison_e4b.md \
  --output-json outputs/eval_reports/benchmark_results_e4b.json

echo "======================================================================"
echo "[Phase 3B/4] Benchmarking Gemma-4-26B-A4B Family in Parallel across 2 GPUs"
echo "  - GPU 0: Base 26B-A4B   (unsloth/gemma-4-26B-A4B-it)                        -> logs/eval_26b_base_gpu0.log"
echo "  - GPU 1: Tuned 26B-A4B  (outputs/merged_gemma4_26b_a4b_arabic/merged_16bit) -> logs/eval_26b_tuned_gpu1.log"
echo "======================================================================"

(
  exec > >(tee -a logs/eval_26b_base_gpu0.log) 2>&1
  export CUDA_VISIBLE_DEVICES=0
  export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_eval_26b_gpu0"
  export TRITON_CACHE_DIR="/tmp/triton_eval_26b_gpu0"
  export VLLM_CACHE_ROOT="/tmp/vllm_eval_26b_gpu0"
  "${PYTHON_BIN}" scripts/run_benchmarks.py \
    --config configs/eval_config.yaml \
    --model-key 26b_a4b \
    --only-role base \
    --use-cached-eval \
    --use-vllm \
    --apply-chat-template \
    --no-reference-fallback
) &
PID_EVAL_26B_BASE=$!

(
  exec > >(tee -a logs/eval_26b_tuned_gpu1.log) 2>&1
  export CUDA_VISIBLE_DEVICES=1
  export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_eval_26b_gpu1"
  export TRITON_CACHE_DIR="/tmp/triton_eval_26b_gpu1"
  export VLLM_CACHE_ROOT="/tmp/vllm_eval_26b_gpu1"
  "${PYTHON_BIN}" scripts/run_benchmarks.py \
    --config configs/eval_config.yaml \
    --model-key 26b_a4b \
    --only-role tuned \
    --model-path outputs/merged_gemma4_26b_a4b_arabic/merged_16bit \
    --use-cached-eval \
    --use-vllm \
    --apply-chat-template \
    --no-reference-fallback
) &
PID_EVAL_26B_TUNED=$!
ACTIVE_PIDS=("${PID_EVAL_26B_BASE}" "${PID_EVAL_26B_TUNED}")

echo "[Barrier] Waiting for 26B-A4B Base (GPU 0, PID=${PID_EVAL_26B_BASE}) and 26B-A4B Tuned (GPU 1, PID=${PID_EVAL_26B_TUNED})..."
RC_EVAL_26B_BASE=0
RC_EVAL_26B_TUNED=0
wait "${PID_EVAL_26B_BASE}" || RC_EVAL_26B_BASE=$?
wait "${PID_EVAL_26B_TUNED}" || RC_EVAL_26B_TUNED=$?
ACTIVE_PIDS=()
sleep 2

if [[ "${RC_EVAL_26B_BASE}" -ne 0 || "${RC_EVAL_26B_TUNED}" -ne 0 ]]; then
  echo "[FATAL] 26B-A4B parallel benchmark failed (RC_BASE_GPU0=${RC_EVAL_26B_BASE}, RC_TUNED_GPU1=${RC_EVAL_26B_TUNED})."
  exit 1
fi

"${PYTHON_BIN}" scripts/run_benchmarks.py \
  --config configs/eval_config.yaml \
  --model-key 26b_a4b \
  --combine-role-reports \
  --output-md outputs/eval_reports/benchmark_comparison_26b_a4b.md \
  --output-json outputs/eval_reports/benchmark_results_26b_a4b.json

"${PYTHON_BIN}" scripts/run_benchmarks.py \
  --config configs/eval_config.yaml \
  --model-key both \
  --combine-role-reports

echo "======================================================================"
echo "[Phase 4/4] Iso-Active Latency, Single-GPU vLLM (1-50 Users) & Sync Bundle"
echo "======================================================================"

"${PYTHON_BIN}" scripts/benchmark_latency_and_plots.py \
  --config configs/eval_config.yaml \
  --live-profile

"${PYTHON_BIN}" scripts/benchmark_vllm_concurrency.py \
  --config configs/eval_config.yaml \
  --live-vllm

if [[ -f "scripts/package_local_sync.py" ]]; then
  "${PYTHON_BIN}" scripts/package_local_sync.py
else
  echo "[Sync Bundle] Creating outputs/gemma4_arabic_local_sync_bundle.tar.gz..."
  "${PYTHON_BIN}" -c "import pathlib, tarfile; root = pathlib.Path('.'); out = root / 'outputs' / 'gemma4_arabic_local_sync_bundle.tar.gz'; out.parent.mkdir(parents=True, exist_ok=True); dirs = ('outputs/sft_e4b_a100', 'outputs/dpo_e4b_a100', 'outputs/sft_26b_a4b_a100', 'outputs/dpo_26b_a4b_a100', 'outputs/eval_reports', 'assets/plots', 'logs'); files = [fp for d in dirs if (root / d).exists() for fp in sorted((root / d).rglob('*')) if fp.is_file() and '.pretokenized_cache' not in fp.parts and not ('checkpoint-' in str(fp) and fp.name in ('optimizer.pt', 'optimizer.bin', 'scheduler.pt', 'rng_state.pth'))]; tf = tarfile.open(out, 'w:gz'); [tf.add(str(fp), arcname=fp.relative_to(root).as_posix()) for fp in files]; tf.close(); print(f'[Sync Bundle] Packaged {len(files)} files into {out}')"
fi

echo "======================================================================"
echo "[COMPLETE] All 4 phases finished successfully!"
echo "  - E4B Report:     outputs/eval_reports/benchmark_comparison_e4b.md"
echo "  - 26B-A4B Report: outputs/eval_reports/benchmark_comparison_26b_a4b.md"
echo "  - Family Report:  outputs/eval_reports/family_benchmark_comparison.md"
echo "  - Sync Bundle:    outputs/gemma4_arabic_local_sync_bundle.tar.gz"
echo "======================================================================"
