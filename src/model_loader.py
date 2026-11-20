"""Unsloth 4-bit NF4 model loader and QLoRA adapter injector for Gemma-4-Arabic (E4B & 26B-A4B MoE).

Supports both the dual-A100 80 GB production tier (zero CPU offload) and the 8.0 GB single-GPU tier:
1. Mapping `embed_tokens_per_layer`, `audio_tower`, and `vision_tower` to CPU RAM when `offload_ple_to_cpu=True`.
2. Registering the non-blocking CPU lookup forward hook on `embed_tokens_per_layer` (5.637 GB).
3. Offloading `audio_tower` (0.618 GB) and `vision_tower` (0.339 GB) to CPU RAM when configured.
4. Freezing token embeddings (`embed_tokens`, `lm_head`, `embed_tokens_per_layer`) and MoE `router` gates
   to preserve Gemma 4's 262,144 vocabulary (including 8,460 native Arabic tokens) and routing balance.
5. Attaching QLoRA (`use_rslora=False`, `r=32, lora_alpha=32` on Dense E4B; `r=16, lora_alpha=16` across
   attention, shared MLP, and all 128 3D expert projections via `target_modules=None` on Sparse 26B-A4B MoE)
   with `use_gradient_checkpointing="unsloth"`.
"""

import builtins
import gc
import inspect
import json
import logging
import math
import os
import pathlib
import re
import shutil
import sys
import types
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _ensure_xformers_for_unsloth(force: bool = False) -> None:
    """Ensure `xformers.ops.fmha` is importable if Unsloth requires it on CUDA 13.x containers.

    Modern Unsloth (`>=2026.x`) uses PyTorch 2's built-in SDPA natively when `xformers` is
    absent. If an older Unsloth build or broken `xformers` binary wheel fails during import,
    this registers a spec-complete, PyTorch SDPA-backed module shim in `sys.modules`.
    """
    if not force:
        try:
            import xformers  # type: ignore

            if not getattr(xformers, "_has_cpp_library", True):
                force = True
            else:
                import xformers.ops.fmha  # type: ignore # noqa: F401
                from xformers.ops.fmha.attn_bias import (  # type: ignore # noqa: F401
                    BlockDiagonalCausalMask,
                    LowerTriangularFromBottomRightMask,
                    LowerTriangularMaskWithTensorBias,
                )

                return
        except Exception:
            force = True

    import importlib.machinery

    def _make_module(name: str, is_pkg: bool = False) -> types.ModuleType:
        mod = types.ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
        if is_pkg:
            spec.submodule_search_locations = []
            mod.__path__ = []  # type: ignore[attr-defined]
        mod.__spec__ = spec
        return mod

    class _AttentionBias:
        """Fallback attention bias base marker for Unsloth's xformers symbol imports."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def from_seqlens(
            cls, q_seqlen: Sequence[int], kv_seqlen: Optional[Sequence[int]] = None
        ) -> "_AttentionBias":
            return cls(q_seqlen=q_seqlen, kv_seqlen=kv_seqlen)

    class _LowerTriangularMask(_AttentionBias):
        pass

    class _BlockDiagonalCausalMask(_AttentionBias):
        pass

    class _LowerTriangularMaskWithTensorBias(_AttentionBias):
        pass

    class _LowerTriangularFromBottomRightMask(_AttentionBias):
        pass

    def _sdpa_memory_efficient_attention(
        query: Any,
        key: Any,
        value: Any,
        attn_bias: Any = None,
        p: float = 0.0,
        scale: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        import torch.nn.functional as F

        # xformers uses (B, M, H, K); torch SDPA uses (B, H, M, K)
        q_t = query.transpose(1, 2)
        k_t = key.transpose(1, 2)
        v_t = value.transpose(1, 2)

        is_causal = attn_bias is not None and (
            isinstance(
                attn_bias,
                (
                    _AttentionBias,
                    type,
                ),
            )
            or "Causal" in type(attn_bias).__name__
            or "LowerTriangular" in type(attn_bias).__name__
        )
        attn_mask = attn_bias if hasattr(attn_bias, "shape") else None
        if attn_mask is not None:
            is_causal = False

        sdpa_kwargs: Dict[str, Any] = {
            "attn_mask": attn_mask,
            "dropout_p": float(p) if p else 0.0,
            "is_causal": is_causal,
        }
        if scale is not None:
            sdpa_kwargs["scale"] = scale
        if q_t.shape[1] != k_t.shape[1]:
            sdpa_kwargs["enable_gqa"] = True

        try:
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, **sdpa_kwargs)
        except TypeError:
            sdpa_kwargs.pop("enable_gqa", None)
            if q_t.shape[1] != k_t.shape[1] and k_t.shape[1] > 0:
                repeat_factor = q_t.shape[1] // k_t.shape[1]
                k_t = k_t.repeat_interleave(repeat_factor, dim=1)
                v_t = v_t.repeat_interleave(repeat_factor, dim=1)
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, **sdpa_kwargs)
        return out.transpose(1, 2).contiguous()

    xf_mod = _make_module("xformers", is_pkg=True)
    xf_mod.__version__ = "0.0.26.post1"  # type: ignore[attr-defined]
    setattr(xf_mod, "_has_cpp_library", True)

    cpp_lib_mod = _make_module("xformers._cpp_lib", is_pkg=False)
    setattr(cpp_lib_mod, "_register_extensions", lambda: None)
    setattr(xf_mod, "_cpp_lib", cpp_lib_mod)

    ops_mod = _make_module("xformers.ops", is_pkg=True)
    fmha_mod = _make_module("xformers.ops.fmha", is_pkg=True)
    attn_bias_mod = _make_module("xformers.ops.fmha.attn_bias", is_pkg=False)

    setattr(attn_bias_mod, "AttentionBias", _AttentionBias)
    setattr(attn_bias_mod, "LowerTriangularMask", _LowerTriangularMask)
    setattr(attn_bias_mod, "BlockDiagonalCausalMask", _BlockDiagonalCausalMask)
    setattr(attn_bias_mod, "LowerTriangularMaskWithTensorBias", _LowerTriangularMaskWithTensorBias)
    setattr(attn_bias_mod, "LowerTriangularFromBottomRightMask", _LowerTriangularFromBottomRightMask)

    for _sub_name in ("cutlass", "flash", "triton_splitk", "small_k", "common"):
        _sub_mod = _make_module(f"xformers.ops.fmha.{_sub_name}", is_pkg=False)
        setattr(_sub_mod, "FwOp", object)
        setattr(_sub_mod, "BwOp", object)
        setattr(fmha_mod, _sub_name, _sub_mod)
        sys.modules[f"xformers.ops.fmha.{_sub_name}"] = _sub_mod

    setattr(fmha_mod, "attn_bias", attn_bias_mod)
    setattr(fmha_mod, "memory_efficient_attention", _sdpa_memory_efficient_attention)
    setattr(fmha_mod, "AttentionBias", _AttentionBias)
    setattr(fmha_mod, "LowerTriangularMask", _LowerTriangularMask)

    setattr(ops_mod, "fmha", fmha_mod)
    setattr(ops_mod, "memory_efficient_attention", _sdpa_memory_efficient_attention)
    setattr(ops_mod, "AttentionBias", _AttentionBias)
    setattr(ops_mod, "LowerTriangularMask", _LowerTriangularMask)
    setattr(ops_mod, "BlockDiagonalCausalMask", _BlockDiagonalCausalMask)
    setattr(ops_mod, "LowerTriangularMaskWithTensorBias", _LowerTriangularMaskWithTensorBias)
    setattr(ops_mod, "LowerTriangularFromBottomRightMask", _LowerTriangularFromBottomRightMask)

    setattr(xf_mod, "ops", ops_mod)

    sys.modules["xformers"] = xf_mod
    sys.modules["xformers._cpp_lib"] = cpp_lib_mod
    sys.modules["xformers.ops"] = ops_mod
    sys.modules["xformers.ops.fmha"] = fmha_mod
    sys.modules["xformers.ops.fmha.attn_bias"] = attn_bias_mod


def configure_clean_logging() -> None:
    """Silence verbose third-party config dumps (Gemma4Config JSON, httpx, SyntaxWarnings) while keeping pipeline INFO logs visible."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("DATASETS_VERBOSITY", "warning")
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
    warnings.filterwarnings("ignore", message=".*Reading global config value for per-layer attribute.*")
    warnings.filterwarnings("ignore", message=".*rope_scaling.*")

    for noisy_logger in (
        "transformers",
        "transformers.configuration_utils",
        "transformers.modeling_utils",
        "transformers.tokenization_utils_base",
        "transformers.integrations.heterogeneity.configuration_utils",
        "httpx",
        "httpx2",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "datasets",
        "accelerate",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR if "heterogeneity" in noisy_logger else logging.WARNING)

    try:
        import transformers.utils.logging as hf_logging

        hf_logging.set_verbosity_warning()
    except Exception:
        pass

    try:
        import datasets.utils.logging as ds_logging  # type: ignore

        ds_logging.set_verbosity_warning()
    except Exception:
        pass


def ensure_unsloth_builtins() -> None:
    """Import `unsloth` first (before `transformers`), then populate `builtins` with Transformers symbols."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("DATASETS_VERBOSITY", "warning")
    # Disable whole-decoder-layer block compilation (`_block_compile.py`) on Gemma 4's
    # heterogeneous sliding/global/KV-shared layers so non-reentrant activation
    # checkpointing never mixes 2D block-compiled and 4D eager Q/K/V norm activations,
    # while keeping all individual Unsloth Triton/Inductor kernels enabled.
    os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")
    # Force symmetric separated MoE grouped-GEMM arithmetic across eval (`compute_ref_log_probs`)
    # and train (`training_step` + GC) on Linux Triton so 128-expert routing never drifts by 1 ULP.
    os.environ.setdefault("UNSLOTH_MOE_LORA_FOLD", "0")
    os.environ.setdefault("UNSLOTH_MOE_GC_KEEP_BASE", "0")
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
    warnings.filterwarnings("ignore", message=".*Reading global config value for per-layer attribute.*")
    warnings.filterwarnings("ignore", message=".*rope_scaling.*")

    if "unsloth" not in sys.modules:
        try:
            import unsloth  # type: ignore # noqa: F401
        except Exception:
            _ensure_xformers_for_unsloth(force=True)
            try:
                import unsloth  # type: ignore # noqa: F401
            except Exception:
                pass

    for _builtin_dec in ("auto_docstring", "strict"):
        if not hasattr(builtins, _builtin_dec):
            try:
                import transformers.utils as _hf_utils  # type: ignore

                setattr(builtins, _builtin_dec, getattr(_hf_utils, _builtin_dec))
            except Exception:
                def _fallback_decorator(*args: Any, **kwargs: Any) -> Any:
                    if args and callable(args[0]) and len(args) == 1 and not kwargs:
                        return args[0]
                    return lambda fn: fn

                setattr(builtins, _builtin_dec, _fallback_decorator)

    import importlib

    for _mod_name in (
        "transformers",
        "transformers.utils",
        "transformers.configuration_utils",
        "transformers.modeling_utils",
        "transformers.cache_utils",
        "transformers.modeling_outputs",
        "transformers.generation",
        "transformers.processing_utils",
        "transformers.models.gemma4.configuration_gemma4",
        "transformers.models.gemma4.modeling_gemma4",
    ):
        try:
            _mod = importlib.import_module(_mod_name)
            for _attr_name in dir(_mod):
                if not _attr_name.startswith("_") and not hasattr(builtins, _attr_name):
                    try:
                        setattr(builtins, _attr_name, getattr(_mod, _attr_name))
                    except Exception:
                        pass
        except Exception:
            pass
    configure_clean_logging()


ensure_unsloth_builtins()

import torch
import torch.nn as nn


def _configure_dynamo_for_heterogeneous_checkpointing() -> None:
    """Disable Dynamo LRU cache reordering (pytorch#166926) and raise recompile limits for Gemma 4."""
    try:
        if hasattr(torch._C, "_dynamo") and hasattr(torch._C._dynamo, "eval_frame"):
            torch._C._dynamo.eval_frame._set_lru_cache(False)
    except Exception:
        pass
    try:
        import torch._dynamo.config as _dynamo_cfg

        for _attr, _min_val in (
            ("cache_size_limit", 256),
            ("recompile_limit", 256),
            ("accumulated_cache_size_limit", 1024),
            ("accumulated_recompile_limit", 1024),
        ):
            if hasattr(_dynamo_cfg, _attr):
                setattr(_dynamo_cfg, _attr, max(int(getattr(_dynamo_cfg, _attr)), _min_val))
    except Exception:
        pass


_configure_dynamo_for_heterogeneous_checkpointing()

from src.constants import (
    BASE_MODEL_ID,
    DEFAULT_MAX_SEQ_LENGTH,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_EXCLUDE_MODULES,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MOE_ROUTER_KEYWORDS,
    MULTIMODAL_TOWER_PATTERNS,
    PLE_MODULE_PATTERN,
    USE_RSLORA,
    load_dotenv_if_present,
)
from src.vram_hooks import (
    OffloadState,
    apply_8gb_vram_optimizations,
    log_vram_telemetry,
    offload_multimodal_towers_to_cpu,
    register_ple_cpu_offload_hook,
)

load_dotenv_if_present()
logger = logging.getLogger(__name__)


def _get_unsloth_loader_cls() -> Any:
    """Prefer Unsloth `FastModel` (native Gemma 4 VLM + 3D MoE support), falling back to `FastLanguageModel`."""
    try:
        from unsloth import FastModel  # type: ignore

        return FastModel
    except Exception:
        from unsloth import FastLanguageModel  # type: ignore

        return FastLanguageModel


def resolve_torch_dtype(dtype_spec: Union[str, torch.dtype]) -> torch.dtype:
    """Resolve a string or torch.dtype specification into a torch.dtype."""
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    normalized = str(dtype_spec).strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype specification: {dtype_spec!r}")
    return mapping[normalized]


def build_8gb_device_map(cuda_device_index: int = 0) -> Dict[str, Union[str, int]]:
    """Construct the device_map that routes PLE and multimodal towers directly to CPU RAM."""
    return build_device_map(
        cuda_device_index=cuda_device_index,
        offload_ple_to_cpu=True,
        offload_multimodal_towers_to_cpu=True,
    )


def build_device_map(
    cuda_device_index: int = 0,
    offload_ple_to_cpu: bool = True,
    offload_multimodal_towers_to_cpu: bool = True,
) -> Dict[str, Union[str, int]]:
    """Build hardware-aware device_map for either 8 GB Edge (CPU offload) or A100 80 GB (pure HBM2e).

    When CPU offload is active, maps all 12 `Gemma4ForConditionalGeneration` submodules explicitly
    without a root `""` entry so `accelerate` does not attach a root `AlignDevicesHook` that drags
    the 5.25 GiB `embed_tokens_per_layer` table onto GPU VRAM.
    """
    if not torch.cuda.is_available():
        return {"": "cpu"}
    if not offload_ple_to_cpu and not offload_multimodal_towers_to_cpu:
        return {"": cuda_device_index}
    ple_target: Union[str, int] = "cpu" if offload_ple_to_cpu else cuda_device_index
    mm_target: Union[str, int] = "cpu" if offload_multimodal_towers_to_cpu else cuda_device_index
    return {
        "model.language_model.embed_tokens": cuda_device_index,
        "model.language_model.layers": cuda_device_index,
        "model.language_model.norm": cuda_device_index,
        "model.language_model.rotary_emb": cuda_device_index,
        "model.language_model.per_layer_model_projection": cuda_device_index,
        "model.language_model.per_layer_projection_norm": cuda_device_index,
        "lm_head": cuda_device_index,
        "model.language_model.embed_tokens_per_layer": ple_target,
        "model.audio_tower": mm_target,
        "model.vision_tower": mm_target,
        "model.embed_audio": mm_target,
        "model.embed_vision": mm_target,
    }


def build_bnb_4bit_config(
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    exclude_modules: Sequence[str] = LORA_EXCLUDE_MODULES,
    enable_fp32_cpu_offload: bool = False,
) -> Any:
    """Build a BitsAndBytesConfig for 4-bit NF4 quantization."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
        llm_int8_skip_modules=list(exclude_modules),
        llm_int8_enable_fp32_cpu_offload=bool(enable_fp32_cpu_offload),
    )


def freeze_embeddings_and_towers(
    model: nn.Module,
    freeze_moe_router: bool = True,
) -> List[str]:
    """Freeze input/output token embeddings, PLE tables, multimodal towers, and MoE routers.

    Never freezes language/expert LoRA adapter weights (`lora_A`, `lora_B`, `lora_embedding`),
    while guaranteeing `router`, `vision_tower`, `audio_tower`, and embeddings remain frozen.
    """
    frozen_names: List[str] = []
    freeze_tokens: List[str] = [
        "embed_tokens",
        "lm_head",
        PLE_MODULE_PATTERN,
        *MULTIMODAL_TOWER_PATTERNS,
        "embed_audio",
        "embed_vision",
        "multi_modal_projector",
    ]
    if freeze_moe_router:
        freeze_tokens.extend(MOE_ROUTER_KEYWORDS)
    for name, param in model.named_parameters():
        is_lora_param = "lora_" in name
        matches_freeze = any(tok in name for tok in freeze_tokens) or (
            ".ref." in name or name.endswith(".ref")
        )
        if matches_freeze:
            param.requires_grad_(False)
            frozen_names.append(name)
        elif is_lora_param and not param.requires_grad:
            param.requires_grad_(True)
    return frozen_names


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: Optional[int] = None,
    num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    chunk_tokens: int = 512,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute exact FP32 cross-entropy in 512-token chunks to prevent 80+ GB FP32 logit OOMs.

    Used automatically on the Transformers/PEFT fallback path so Gemma 4's 262,144-token
    vocabulary never materializes a full `[B, S, 262144]` FP32 tensor in VRAM at once.
    Mathematically identical to `ForCausalLMLoss` (exact same loss and gradients).
    """
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    if shift_labels is None:
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
    else:
        shift_logits = logits

    v_size = int(vocab_size) if vocab_size is not None else int(shift_logits.shape[-1])
    flat_logits = shift_logits.reshape(-1, v_size)
    flat_labels = shift_labels.reshape(-1).to(flat_logits.device)

    total_tokens = flat_logits.shape[0]
    if total_tokens == 0:
        return flat_logits.sum() * 0.0

    def _ce_chunk_step(logit_slice: torch.Tensor, label_slice: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logit_slice.float(),
            label_slice,
            ignore_index=ignore_index,
            reduction="sum",
        )

    use_ckpt = bool(torch.is_grad_enabled() and flat_logits.requires_grad)
    losses: List[torch.Tensor] = []
    for start_idx in range(0, total_tokens, chunk_tokens):
        end_idx = min(start_idx + chunk_tokens, total_tokens)
        lbl_chunk = flat_labels[start_idx:end_idx]
        if not torch.any(lbl_chunk != ignore_index):
            continue
        logit_chunk = flat_logits[start_idx:end_idx]
        if use_ckpt:
            losses.append(
                checkpoint(_ce_chunk_step, logit_chunk, lbl_chunk, use_reentrant=False)
            )
        else:
            losses.append(_ce_chunk_step(logit_chunk, lbl_chunk))

    if not losses:
        return flat_logits.sum() * 0.0

    total_loss = torch.stack(losses).sum()
    if num_items_in_batch is not None:
        denom = (
            num_items_in_batch.to(total_loss.device, dtype=total_loss.dtype)
            if torch.is_tensor(num_items_in_batch)
            else float(num_items_in_batch)
        )
        return total_loss / denom
    valid_count = (flat_labels != ignore_index).sum().clamp_min(1).to(total_loss.dtype)
    return total_loss / valid_count


def install_chunked_causal_lm_loss(model: nn.Module) -> None:
    """Install `chunked_causal_lm_loss` on a Transformers/PEFT fallback model hierarchy."""
    for obj in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None),
    ):
        if obj is not None and isinstance(obj, nn.Module):
            try:
                setattr(obj, "loss_function", chunked_causal_lm_loss)
            except Exception:
                pass


def ensure_tied_lm_head(model: nn.Module) -> bool:
    """Ensure `model.config.tie_word_embeddings = True` and tie `lm_head.weight` to `embed_tokens.weight`."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for obj in (model, base):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            try:
                cfg.tie_word_embeddings = True
            except Exception:
                pass
            if getattr(cfg, "text_config", None) is not None:
                try:
                    cfg.text_config.tie_word_embeddings = True
                except Exception:
                    pass

    in_emb: Any = None
    if hasattr(base, "get_input_embeddings"):
        try:
            in_emb = base.get_input_embeddings()
        except Exception:
            in_emb = None
    if in_emb is None:
        lm_backbone = getattr(getattr(base, "model", None), "language_model", None)
        in_emb = getattr(lm_backbone, "embed_tokens", None) or getattr(
            getattr(base, "model", base), "embed_tokens", None
        )

    out_emb: Any = getattr(base, "lm_head", None)
    if out_emb is None and hasattr(base, "get_output_embeddings"):
        try:
            out_emb = base.get_output_embeddings()
        except Exception:
            out_emb = None
    if out_emb is None and hasattr(base, "model"):
        out_emb = getattr(base.model, "lm_head", None)

    if (
        in_emb is not None
        and out_emb is not None
        and hasattr(in_emb, "weight")
        and hasattr(out_emb, "weight")
        and in_emb.weight is not None
        and out_emb.weight is not None
    ):
        if in_emb.weight.data_ptr() != out_emb.weight.data_ptr():
            out_emb.weight = in_emb.weight
            logger.info("Tied lm_head.weight to embed_tokens.weight (ptr=%d).", in_emb.weight.data_ptr())
        return out_emb.weight.data_ptr() == in_emb.weight.data_ptr()
    return False


def _normalize_offloaded_hf_device_map_for_training(
    model: nn.Module,
    device_idx: int = 0,
) -> None:
    """Normalize `hf_device_map` to `{"": device_idx}` after frozen PLE/tower CPU offload hooks are installed.

    `accelerate >= 1.6`'s `Accelerator.prepare_model` rejects 4-bit/8-bit training whenever `"cpu"` or
    `"disk"` appears in `model.hf_device_map`, even when 100% of quantized decoder layers and trainable
    LoRA adapters reside on `cuda:{device_idx}` and only frozen non-quantized embeddings/towers are
    offloaded. Setting `hf_device_map = {"": device_idx}` satisfies `Accelerator.prepare_model` while
    preserving `Trainer._move_model_to_device`'s early return so offloaded frozen weights stay on CPU.
    """
    target_map: Dict[str, Any] = {"": int(device_idx)}
    for obj in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None),
    ):
        if obj is None:
            continue
        cur_map = getattr(obj, "hf_device_map", None)
        if isinstance(cur_map, dict):
            vals = set(cur_map.values())
            if "cpu" in vals or "disk" in vals:
                setattr(obj, "hf_device_map", dict(target_map))
        elif obj is model and cur_map is None:
            base_map = getattr(getattr(getattr(model, "base_model", None), "model", None), "hf_device_map", None)
            if isinstance(base_map, dict):
                setattr(obj, "hf_device_map", dict(target_map))


def _configure_trainable_model_flags(model: nn.Module, is_trainable: bool = True) -> None:
    """Ensure `use_cache=False` across top-level and nested `text_config` objects during training."""
    if not is_trainable:
        return
    seen_configs = set()
    for obj in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None),
    ):
        if obj is None:
            continue
        cfg = getattr(obj, "config", None)
        if cfg is not None and id(cfg) not in seen_configs:
            seen_configs.add(id(cfg))
            try:
                cfg.use_cache = False
            except Exception:
                pass
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                try:
                    text_cfg.use_cache = False
                except Exception:
                    pass


def inject_rslora_adapters(
    model: nn.Module,
    r: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    bias: str = LORA_BIAS,
    use_rslora: bool = USE_RSLORA,
    target_modules: Optional[Sequence[str]] = None,
    exclude_modules: Sequence[str] = LORA_EXCLUDE_MODULES,
    use_gradient_checkpointing: Union[str, bool] = "unsloth",
    random_state: int = 3407,
    is_moe: bool = False,
) -> nn.Module:
    """Attach LoRA/rsLoRA adapters across language attention, shared MLP, and (on 26B MoE) all 128 3D experts.

    When `target_modules is None` or `is_moe=True`, delegates target discovery to Unsloth's
    `FastModel.get_peft_model` (`finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True`) so Gemma-4-26B-A4B trains
    both 2D attention/shared-MLP projections AND 3D `ParamWrapper` routed experts
    (`experts.gate_up_proj`, `experts.down_proj`, ~498.5M params at `r=16`) while keeping
    `router`, `vision_tower`, `audio_tower`, and embeddings frozen.
    """
    # Sanitize any legacy "mlp.gate_proj" prefixes into bare leaf names if target_modules was supplied
    sanitized_targets: Optional[List[str]] = None
    if target_modules is not None and not is_moe:
        sanitized_targets = [
            m.split(".")[-1] if m.startswith("mlp.") else m for m in target_modules
        ]
    fallback_target_list = sanitized_targets if sanitized_targets else list(LORA_TARGET_MODULES)
    exclude_list = list(dict.fromkeys([*LORA_EXCLUDE_MODULES, *exclude_modules]))
    exclude_regex = (
        r".*(" + "|".join(exclude_list) + r").*"
        if exclude_list
        else None
    )

    try:
        loader_cls = _get_unsloth_loader_cls()

        logger.info(
            "Attaching LoRA via Unsloth %s.get_peft_model "
            "(r=%d, alpha=%d, use_rslora=%s, is_moe=%s, targets=%s).",
            loader_cls.__name__,
            r,
            lora_alpha,
            use_rslora,
            is_moe,
            "auto_language_and_experts" if sanitized_targets is None else sanitized_targets,
        )
        unsloth_peft_kwargs: Dict[str, Any] = {
            "r": r,
            "lora_alpha": lora_alpha,
            "use_rslora": use_rslora,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
        }
        sig = inspect.signature(loader_cls.get_peft_model)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        supports_auto_flags = "finetune_mlp_modules" in sig.parameters or has_var_kw
        if "finetune_vision_layers" in sig.parameters or has_var_kw:
            unsloth_peft_kwargs["finetune_vision_layers"] = False
        if "finetune_language_layers" in sig.parameters or has_var_kw:
            unsloth_peft_kwargs["finetune_language_layers"] = True
        if "finetune_attention_modules" in sig.parameters or has_var_kw:
            unsloth_peft_kwargs["finetune_attention_modules"] = True
        if "finetune_mlp_modules" in sig.parameters or has_var_kw:
            unsloth_peft_kwargs["finetune_mlp_modules"] = True

        if sanitized_targets is not None and not is_moe:
            unsloth_peft_kwargs["target_modules"] = sanitized_targets
        elif not supports_auto_flags:
            unsloth_peft_kwargs["target_modules"] = fallback_target_list

        try:
            peft_model = loader_cls.get_peft_model(
                model,
                exclude_modules=exclude_list,
                **unsloth_peft_kwargs,
            )
        except TypeError:
            peft_model = loader_cls.get_peft_model(
                model,
                **unsloth_peft_kwargs,
            )
        if not use_gradient_checkpointing:
            if hasattr(peft_model, "gradient_checkpointing_disable"):
                try:
                    peft_model.gradient_checkpointing_disable()
                except Exception:
                    pass
            for mod in peft_model.modules():
                if hasattr(mod, "gradient_checkpointing"):
                    try:
                        mod.gradient_checkpointing = False
                    except Exception:
                        pass
        setattr(peft_model, "_is_unsloth_model", True)
        return peft_model
    except Exception as unsloth_exc:
        logger.warning(
            "Unsloth get_peft_model unavailable (%s); falling back to PEFT LoraConfig.",
            unsloth_exc,
        )
        from peft import LoraConfig, get_peft_model

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()

        lora_kwargs: Dict[str, Any] = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "use_rslora": use_rslora,
            "target_modules": fallback_target_list,
            "exclude_modules": exclude_regex,
            "task_type": "CAUSAL_LM",
        }
        if is_moe and "target_parameters" in inspect.signature(LoraConfig.__init__).parameters:
            has_3d_experts = any(
                "experts.gate_up_proj" in n for n, _ in model.named_parameters()
            )
            if has_3d_experts:
                lora_kwargs["target_parameters"] = [
                    "experts.gate_up_proj",
                    "experts.down_proj",
                ]
        lora_cfg = LoraConfig(**lora_kwargs)
        peft_model = get_peft_model(model, lora_cfg)
        if not getattr(model, "_is_unsloth_model", False):
            install_chunked_causal_lm_loss(peft_model)
        return peft_model


def load_gemma4_8gb_model_and_tokenizer(
    model_name: str = BASE_MODEL_ID,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    dtype: Union[str, torch.dtype] = torch.bfloat16,
    load_in_4bit: bool = True,
    attach_lora: bool = True,
    lora_r: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    use_rslora: bool = USE_RSLORA,
    use_gradient_checkpointing: Union[str, bool] = "unsloth",
    random_state: int = 3407,
    adapter_checkpoint: Optional[str] = None,
    is_trainable: bool = True,
    offload_ple_to_cpu: bool = True,
    offload_multimodal_towers_to_cpu: bool = True,
    freeze_moe_router: bool = True,
    target_modules: Optional[Sequence[str]] = None,
    exclude_modules: Optional[Sequence[str]] = None,
    cuda_device_index: Optional[int] = None,
) -> Tuple[nn.Module, Any, OffloadState]:
    """Load Gemma-4-E4B or Gemma-4-26B-A4B (MoE) with conditional VRAM hooks and rsLoRA."""
    configure_clean_logging()
    resolved_dtype = resolve_torch_dtype(dtype)
    device_idx = (
        int(os.environ.get("LOCAL_RANK", "0"))
        if cuda_device_index is None
        else int(cuda_device_index)
    )
    enable_cpu_offload = bool(offload_ple_to_cpu or offload_multimodal_towers_to_cpu)
    device_map = build_device_map(
        cuda_device_index=device_idx,
        offload_ple_to_cpu=offload_ple_to_cpu,
        offload_multimodal_towers_to_cpu=offload_multimodal_towers_to_cpu,
    )
    is_moe_arch = "26b" in model_name.lower() or "a4b" in model_name.lower()
    resolved_targets: Optional[List[str]] = (
        None
        if (target_modules is None or is_moe_arch)
        else list(target_modules)
    )
    resolved_excludes = list(exclude_modules) if exclude_modules is not None else list(LORA_EXCLUDE_MODULES)
    load_path = adapter_checkpoint if adapter_checkpoint is not None else model_name

    model: nn.Module
    tokenizer: Any

    hf_config: Any = None
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name)
        text_subcfg = getattr(hf_config, "text_config", hf_config)
        if getattr(text_subcfg, "num_local_experts", 0) or getattr(text_subcfg, "enable_moe_block", False):
            is_moe_arch = True
            resolved_targets = None
        try:
            hf_config.tie_word_embeddings = True
            if getattr(hf_config, "text_config", None) is not None:
                hf_config.text_config.tie_word_embeddings = True
        except Exception:
            pass
        if is_trainable:
            try:
                hf_config.use_cache = False
            except Exception:
                pass
            if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
                try:
                    hf_config.text_config.use_cache = False
                except Exception:
                    pass
        if enable_cpu_offload:
            quant_cfg = getattr(hf_config, "quantization_config", None)
            if isinstance(quant_cfg, dict):
                quant_cfg["llm_int8_enable_fp32_cpu_offload"] = True
            elif quant_cfg is not None and hasattr(quant_cfg, "llm_int8_enable_fp32_cpu_offload"):
                quant_cfg.llm_int8_enable_fp32_cpu_offload = True
    except Exception as cfg_exc:
        logger.debug("AutoConfig preload skipped for '%s': %s", model_name, cfg_exc)

    used_unsloth_loader = False
    try:
        loader_cls = _get_unsloth_loader_cls()

        logger.info(
            "Loading '%s' via Unsloth %s (max_seq_length=%d, load_in_4bit=%s, is_moe=%s, cpu_offload=%s).",
            load_path,
            loader_cls.__name__,
            max_seq_length,
            load_in_4bit,
            is_moe_arch,
            enable_cpu_offload,
        )
        unsloth_kwargs: Dict[str, Any] = {
            "model_name": load_path,
            "max_seq_length": max_seq_length,
            "dtype": resolved_dtype,
            "load_in_4bit": load_in_4bit,
            "device_map": device_map,
        }
        sig_fp = inspect.signature(loader_cls.from_pretrained)
        fp_has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_fp.parameters.values()
        )
        if not load_in_4bit and ("load_in_16bit" in sig_fp.parameters or fp_has_var_kw):
            unsloth_kwargs["load_in_16bit"] = True

        orig_bnb_validate: Any = None
        bnb_quantizer_cls: Any = None
        orig_alloc_warmup: Any = None
        modeling_utils_mod: Any = None
        if enable_cpu_offload:
            try:
                import transformers.modeling_utils as _hf_mu  # type: ignore

                modeling_utils_mod = _hf_mu
                orig_alloc_warmup = getattr(_hf_mu, "caching_allocator_warmup", None)
                if orig_alloc_warmup is not None:
                    _hf_mu.caching_allocator_warmup = lambda *a, **kw: None  # type: ignore[assignment]
            except Exception:
                modeling_utils_mod = None
        if enable_cpu_offload and load_in_4bit:
            try:
                from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer  # type: ignore

                bnb_quantizer_cls = Bnb4BitHfQuantizer
                orig_bnb_validate = Bnb4BitHfQuantizer.validate_environment

                def _cpu_offload_aware_validate(self_q: Any, *v_args: Any, **v_kwargs: Any) -> Any:
                    dmap = v_kwargs.get("device_map")
                    if isinstance(dmap, dict) and "cpu" in set(dmap.values()):
                        self_q.quantization_config.llm_int8_enable_fp32_cpu_offload = True
                    return orig_bnb_validate(self_q, *v_args, **v_kwargs)

                Bnb4BitHfQuantizer.validate_environment = _cpu_offload_aware_validate  # type: ignore[assignment]
            except Exception:
                bnb_quantizer_cls = None

        try:
            try:
                model, raw_proc_or_tok = loader_cls.from_pretrained(**unsloth_kwargs)
            except TypeError:
                unsloth_kwargs.pop("load_in_16bit", None)
                model, raw_proc_or_tok = loader_cls.from_pretrained(**unsloth_kwargs)
        finally:
            if bnb_quantizer_cls is not None and orig_bnb_validate is not None:
                bnb_quantizer_cls.validate_environment = orig_bnb_validate
            if modeling_utils_mod is not None and orig_alloc_warmup is not None:
                modeling_utils_mod.caching_allocator_warmup = orig_alloc_warmup

        if hasattr(raw_proc_or_tok, "tokenizer") and not hasattr(raw_proc_or_tok, "encode"):
            inner_tok = raw_proc_or_tok.tokenizer
            if (
                getattr(inner_tok, "chat_template", None) is None
                and getattr(raw_proc_or_tok, "chat_template", None) is not None
            ):
                inner_tok.chat_template = raw_proc_or_tok.chat_template
            setattr(inner_tok, "_gemma4_processor", raw_proc_or_tok)
            tokenizer = inner_tok
        else:
            tokenizer = raw_proc_or_tok

        if adapter_checkpoint is not None and is_trainable:
            if hasattr(loader_cls, "for_training"):
                try:
                    loader_cls.for_training(
                        model,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                    )
                except TypeError:
                    loader_cls.for_training(model)
            if not use_gradient_checkpointing:
                if hasattr(model, "gradient_checkpointing_disable"):
                    try:
                        model.gradient_checkpointing_disable()
                    except Exception:
                        pass
                for mod in model.modules():
                    if hasattr(mod, "gradient_checkpointing"):
                        try:
                            mod.gradient_checkpointing = False
                        except Exception:
                            pass
                    if hasattr(mod, "_offloaded_gradient_checkpointing"):
                        try:
                            mod._offloaded_gradient_checkpointing = False
                        except Exception:
                            pass
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)

        used_unsloth_loader = True
        setattr(model, "_is_unsloth_model", True)
    except Exception as unsloth_exc:
        logger.warning(
            "Unsloth FastLanguageModel.from_pretrained unavailable (%s); using Transformers loader.",
            unsloth_exc,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hf_config is None:
            hf_config = AutoConfig.from_pretrained(model_name)
        try:
            hf_config.tie_word_embeddings = True
            if getattr(hf_config, "text_config", None) is not None:
                hf_config.text_config.tie_word_embeddings = True
        except Exception:
            pass
        if is_trainable:
            try:
                hf_config.use_cache = False
            except Exception:
                pass
            if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
                try:
                    hf_config.text_config.use_cache = False
                except Exception:
                    pass

        quant_cfg = getattr(hf_config, "quantization_config", None)
        has_prequant = quant_cfg is not None
        if enable_cpu_offload:
            if isinstance(quant_cfg, dict):
                quant_cfg["llm_int8_enable_fp32_cpu_offload"] = True
            elif quant_cfg is not None and hasattr(quant_cfg, "llm_int8_enable_fp32_cpu_offload"):
                quant_cfg.llm_int8_enable_fp32_cpu_offload = True

        load_kwargs: Dict[str, Any] = {
            "config": hf_config,
            "dtype": resolved_dtype,
            "device_map": device_map,
        }
        if torch.cuda.is_available() and getattr(hf_config, "audio_config", None) is None:
            load_kwargs["attn_implementation"] = "sdpa"
        if not has_prequant and load_in_4bit and torch.cuda.is_available():
            load_kwargs["quantization_config"] = build_bnb_4bit_config(
                compute_dtype=resolved_dtype,
                exclude_modules=resolved_excludes,
                enable_fp32_cpu_offload=enable_cpu_offload,
            )

        arch_list = getattr(hf_config, "architectures", None) or []
        is_multimodal_arch = (
            getattr(hf_config, "model_type", "") == "gemma4"
            or any("ConditionalGeneration" in a for a in arch_list)
        )
        loader_cls: Any = AutoModelForCausalLM
        if is_multimodal_arch:
            from transformers import AutoModelForImageTextToText

            loader_cls = AutoModelForImageTextToText

        try:
            model = loader_cls.from_pretrained(model_name, **load_kwargs)
        except (TypeError, ValueError):
            load_kwargs.pop("attn_implementation", None)
            model = loader_cls.from_pretrained(model_name, **load_kwargs)

        if is_trainable:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                try:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    model.gradient_checkpointing_enable()

        install_chunked_causal_lm_loss(model)

        if adapter_checkpoint is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                adapter_checkpoint,
                is_trainable=is_trainable,
            )
            install_chunked_causal_lm_loss(model)

    ensure_tied_lm_head(model)
    _configure_trainable_model_flags(model, is_trainable=is_trainable)

    # Normalize Gemma 4 instruct EOS token (`<turn|>` = 106) and `[1, 106, 50]` stop IDs
    # (`unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` ships with `eos_token="<eos>"` / `eos_token_id=1`).
    try:
        if hasattr(tokenizer, "convert_tokens_to_ids") and tokenizer.convert_tokens_to_ids("<turn|>") == 106:
            if getattr(tokenizer, "eos_token", None) != "<turn|>":
                tokenizer.eos_token = "<turn|>"
            proc_attached = getattr(tokenizer, "_gemma4_processor", None)
            if proc_attached is not None and hasattr(proc_attached, "tokenizer"):
                if getattr(proc_attached.tokenizer, "eos_token", None) != "<turn|>":
                    proc_attached.tokenizer.eos_token = "<turn|>"
            for cfg_holder in (
                model,
                getattr(model, "base_model", None),
                getattr(getattr(model, "base_model", None), "model", None),
            ):
                if cfg_holder is None:
                    continue
                m_cfg = getattr(cfg_holder, "config", None)
                if m_cfg is not None and getattr(m_cfg, "model_type", "") == "gemma4":
                    m_cfg.eos_token_id = [1, 106, 50]
                    if getattr(m_cfg, "text_config", None) is not None:
                        m_cfg.text_config.eos_token_id = [1, 106, 50]
                gen_cfg = getattr(cfg_holder, "generation_config", None)
                if gen_cfg is not None:
                    gen_cfg.eos_token_id = [1, 106, 50]
    except Exception as eos_exc:
        logger.debug("Gemma 4 EOS normalization skipped: %s", eos_exc)

    # Freeze embeddings, multimodal towers, and MoE router gates
    freeze_embeddings_and_towers(model, freeze_moe_router=freeze_moe_router)

    target_dev = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    if offload_ple_to_cpu and offload_multimodal_towers_to_cpu:
        offload_state = apply_8gb_vram_optimizations(
            model=model,
            target_device=target_dev,
            target_dtype=resolved_dtype,
            non_blocking=True,
        )
    else:
        offload_state = OffloadState()
        if offload_ple_to_cpu:
            register_ple_cpu_offload_hook(
                model=model,
                target_device=target_dev,
                target_dtype=resolved_dtype,
                non_blocking=True,
                state=offload_state,
            )
        if offload_multimodal_towers_to_cpu:
            offload_multimodal_towers_to_cpu(
                model=model,
                patterns=MULTIMODAL_TOWER_PATTERNS,
                state=offload_state,
            )
        setattr(model, "_vram_offload_state", offload_state)

    if attach_lora and adapter_checkpoint is None:
        model = inject_rslora_adapters(
            model=model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=LORA_BIAS,
            use_rslora=use_rslora,
            target_modules=resolved_targets,
            exclude_modules=resolved_excludes,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            is_moe=is_moe_arch,
        )
        # Re-verify PLE, multimodal towers, and MoE routers remain frozen after PEFT wrapping
        freeze_embeddings_and_towers(model, freeze_moe_router=freeze_moe_router)
        ensure_tied_lm_head(model)
        setattr(model, "_vram_offload_state", offload_state)

    if not used_unsloth_loader and not getattr(model, "_is_unsloth_model", False):
        install_chunked_causal_lm_loss(model)
    _configure_trainable_model_flags(model, is_trainable=is_trainable)
    if enable_cpu_offload and is_trainable and torch.cuda.is_available():
        _normalize_offloaded_hf_device_map_for_training(model, device_idx=device_idx)

    log_vram_telemetry(model, stage="post_model_load")
    return model, tokenizer, offload_state


load_gemma4_model_and_tokenizer = load_gemma4_8gb_model_and_tokenizer


def atomic_write_json(path: Union[str, pathlib.Path], payload: Dict[str, Any]) -> None:
    """Atomically write `payload` to `path` via temp file + `fsync` + `os.replace`."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(serialized)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp_path), str(target))


def is_adapter_stage_complete(
    output_dir: Union[str, pathlib.Path],
    metrics_filename: str,
    cleanup_partial_root: bool = False,
) -> bool:
    """Return `True` iff a PEFT training stage wrote valid adapter weights AND its terminal metrics JSON.

    If `cleanup_partial_root=True` and a crash happened mid-final-save (leaving root `adapter_config.json`
    without `metrics_filename`), removes the partial root adapter artifacts while keeping `checkpoint-*`
    intact so `find_latest_valid_checkpoint` resumes cleanly.
    """
    out = pathlib.Path(output_dir)
    if not out.is_dir():
        return False

    cfg_file = out / "adapter_config.json"
    metrics_file = out / metrics_filename
    safetensors_file = out / "adapter_model.safetensors"
    bin_file = out / "adapter_model.bin"

    has_weights = (
        (safetensors_file.is_file() and safetensors_file.stat().st_size > 256)
        or (bin_file.is_file() and bin_file.stat().st_size > 256)
    )
    if cfg_file.is_file() and has_weights and metrics_file.is_file():
        try:
            json.loads(cfg_file.read_text(encoding="utf-8"))
            json.loads(metrics_file.read_text(encoding="utf-8"))
            return True
        except Exception:
            pass

    if cleanup_partial_root and (cfg_file.exists() or safetensors_file.exists() or bin_file.exists()):
        logger.warning(
            "Detected incomplete final adapter save in %s (missing/corrupt %s); cleaning partial root files to allow checkpoint resume.",
            out,
            metrics_filename,
        )
        for partial in (cfg_file, safetensors_file, bin_file, metrics_file):
            try:
                if partial.is_file():
                    partial.unlink()
            except OSError:
                pass
    return False


def find_latest_valid_checkpoint(
    output_dir: Union[str, pathlib.Path],
    require_optimizer_state: bool = True,
    cleanup_corrupt: bool = True,
) -> Optional[str]:
    """Locate the highest-step complete `checkpoint-<N>` directory, skipping/cleaning half-written saves."""
    out = pathlib.Path(output_dir)
    if not out.is_dir():
        return None

    candidates: List[Tuple[int, pathlib.Path]] = []
    for child in out.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^checkpoint-(\d+)$", child.name)
        if m:
            candidates.append((int(m.group(1)), child))

    candidates.sort(key=lambda item: item[0], reverse=True)

    for step_num, ckpt_dir in candidates:
        reason: Optional[str] = None
        state_file = ckpt_dir / "trainer_state.json"
        cfg_file = ckpt_dir / "adapter_config.json"
        st_file = ckpt_dir / "adapter_model.safetensors"
        bin_file = ckpt_dir / "adapter_model.bin"
        opt_pt = ckpt_dir / "optimizer.pt"
        opt_bin = ckpt_dir / "optimizer.bin"

        if not state_file.is_file():
            reason = "missing trainer_state.json"
        else:
            try:
                st_data = json.loads(state_file.read_text(encoding="utf-8"))
                if int(st_data.get("global_step", 0)) <= 0:
                    reason = "trainer_state.json global_step <= 0"
            except Exception as exc:
                reason = f"corrupt trainer_state.json ({exc})"

        if reason is None:
            if not cfg_file.is_file():
                reason = "missing adapter_config.json"
            else:
                try:
                    json.loads(cfg_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    reason = f"corrupt adapter_config.json ({exc})"

        if reason is None:
            has_adapter_weights = (
                (st_file.is_file() and st_file.stat().st_size > 256)
                or (bin_file.is_file() and bin_file.stat().st_size > 256)
            )
            if not has_adapter_weights:
                reason = "missing or empty adapter_model.safetensors/.bin"

        if reason is None and require_optimizer_state:
            has_opt = (
                (opt_pt.is_file() and opt_pt.stat().st_size > 64)
                or (opt_bin.is_file() and opt_bin.stat().st_size > 64)
            )
            if not has_opt:
                reason = "missing or empty optimizer.pt/.bin"

        ref_subdir = ckpt_dir / "ref"
        if reason is None and ref_subdir.is_dir():
            ref_cfg = ref_subdir / "adapter_config.json"
            ref_st = ref_subdir / "adapter_model.safetensors"
            ref_bin = ref_subdir / "adapter_model.bin"
            if not ref_cfg.is_file() or not (
                (ref_st.is_file() and ref_st.stat().st_size > 256)
                or (ref_bin.is_file() and ref_bin.stat().st_size > 256)
            ):
                reason = "incomplete DPO ref/ adapter subdirectory"

        if reason is None:
            return str(ckpt_dir)

        logger.warning(
            "Skipping incomplete checkpoint-%d at %s (%s)%s.",
            step_num,
            ckpt_dir,
            reason,
            "; removing broken directory" if cleanup_corrupt else "",
        )
        if cleanup_corrupt:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    return None


def verify_moe_peft_version() -> None:
    """Verify `peft >= 0.21.3` on live CUDA runs for Gemma-4-26B-A4B 3D MoE `ParamWrapper` support."""
    try:
        import peft
        from packaging.version import Version

        cur = Version(str(peft.__version__).split("+")[0])
        req = Version("0.21.3")
        if cur < req:
            raise RuntimeError(
                f"CRITICAL Dependency Error: Gemma-4-26B-A4B 3D MoE LoRA requires `peft>=0.21.3`, "
                f"found `peft=={peft.__version__}`. Upgrade via `pip install -U 'peft>=0.21.3'`."
            )
    except ImportError:
        pass


def verify_moe_lora_integrity(
    model: nn.Module,
    is_moe: bool = False,
    smoke_test: bool = False,
    min_moe_trainable_params: int = 250_000_000,
    expect_trained_adapter: bool = False,
    adapter_dir: Optional[Union[str, pathlib.Path]] = None,
    require_frozen_router: bool = True,
) -> Dict[str, Any]:
    """Verify 3D MoE `ParamWrapper` LoRA attachment, `rank_major` layout, and SFT weight loading."""
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if not is_moe:
        return {
            "is_moe": False,
            "trainable_params": trainable_params,
            "num_expert_wrappers": 0,
            "lora_b_layout": "n/a",
            "expert_lora_b_l2_norm": 0.0,
        }

    if not smoke_test and trainable_params < int(min_moe_trainable_params):
        raise RuntimeError(
            f"CRITICAL MoE LoRA Error: Gemma-4-26B-A4B has only {trainable_params:,} trainable "
            f"parameters (< {min_moe_trainable_params:,})! All 128 3D routed experts "
            "(`experts.gate_up_proj`, `experts.down_proj`) MUST be attached (~498.5M params at r=16)."
        )

    if require_frozen_router:
        trainable_routers = [
            n for n, p in model.named_parameters() if p.requires_grad and "router" in n.lower()
        ]
        if trainable_routers:
            raise RuntimeError(
                f"CRITICAL MoE Router Error: MoE router parameters must remain frozen, "
                f"found trainable router params: {trainable_routers[:5]}"
            )

    expert_wrappers: List[Tuple[str, nn.Module]] = []
    lora_b_sq_sum = 0.0
    lora_a_sq_sum = 0.0
    repacked_tensors: List[str] = []

    for mod_name, mod in model.named_modules():
        param_name = getattr(mod, "parameter_name", None)
        if param_name not in ("gate_up_proj", "down_proj"):
            continue
        lora_a_dict = getattr(mod, "lora_A", None)
        lora_b_dict = getattr(mod, "lora_B", None)
        if lora_a_dict is None or lora_b_dict is None or "default" not in lora_b_dict:
            continue
        expert_wrappers.append((mod_name, mod))
        w_a = lora_a_dict["default"].weight
        w_b = lora_b_dict["default"].weight
        if not w_a.requires_grad or not w_b.requires_grad:
            raise RuntimeError(
                f"CRITICAL MoE LoRA Error: Expert ParamWrapper '{mod_name}' ({param_name}) "
                "has frozen `default` LoRA weights (`requires_grad=False`)!"
            )
        w_a_f = w_a.detach().float()
        w_b_f = w_b.detach().float()
        if not bool(torch.isfinite(w_a_f).all().item()) or not bool(torch.isfinite(w_b_f).all().item()):
            raise RuntimeError(
                f"CRITICAL MoE LoRA Error: NaN/Inf detected in expert LoRA weights at '{mod_name}'!"
            )
        lora_a_sq_sum += float(w_a_f.pow(2).sum().item())
        lora_b_sq_sum += float(w_b_f.pow(2).sum().item())
        if getattr(w_b, "_unsloth_lora_b_repacked", None) is not None:
            repacked_tensors.append(mod_name)

    if not expert_wrappers:
        raise RuntimeError(
            "CRITICAL MoE LoRA Error: Zero 3D `ParamWrapper` expert modules (`gate_up_proj`, `down_proj`) "
            "found on MoE model! Routed experts are NOT receiving LoRA adapters."
        )

    lora_b_norm = math.sqrt(lora_b_sq_sum)
    lora_a_norm = math.sqrt(lora_a_sq_sum)
    if repacked_tensors:
        raise RuntimeError(
            f"CRITICAL MoE LoRA Layout Error: Unsloth legacy-repacked {len(repacked_tensors)} `lora_B` "
            f"tensors ({repacked_tensors[:3]}), indicating missing `lora_B_layout='rank_major'` metadata!"
        )
    if expect_trained_adapter and lora_b_norm <= 1e-6:
        raise RuntimeError(
            f"CRITICAL MoE Adapter Load Error: Loaded Stage 1 SFT MoE adapter has zero expert `lora_B` "
            f"L2 norm ({lora_b_norm:.3e})! Stage 1 3D expert weights failed to load into Stage 2 DPO."
        )

    resolved_layout = "rank_major"
    try:
        from unsloth_zoo.temporary_patches.moe_utils import moe_lora_b_layout

        resolved_layout = str(moe_lora_b_layout())
        if resolved_layout != "rank_major":
            raise RuntimeError(
                f"CRITICAL MoE LoRA Layout Error: Active `moe_lora_b_layout()` is '{resolved_layout}', "
                "expected 'rank_major'!"
            )
    except ImportError:
        pass

    if adapter_dir is not None:
        cfg_file = pathlib.Path(adapter_dir) / "adapter_config.json"
        if cfg_file.is_file():
            cfg_json = json.loads(cfg_file.read_text(encoding="utf-8"))
            target_params = cfg_json.get("target_parameters") or []
            if not target_params:
                raise RuntimeError(
                    f"CRITICAL MoE Adapter Config Error: `{cfg_file}` is missing `target_parameters`!"
                )
            declared_layout = cfg_json.get("lora_B_layout")
            auto_map = cfg_json.get("auto_mapping") or {}
            detail = auto_map.get("unsloth_fused_expert_lora")
            if declared_layout is not None and declared_layout != "rank_major":
                raise RuntimeError(
                    f"CRITICAL MoE Adapter Config Error: `{cfg_file}` declares "
                    f"`lora_B_layout={declared_layout!r}` (expected 'rank_major')!"
                )
            if isinstance(detail, dict) and isinstance(detail.get("parameters"), dict):
                bad_params = {
                    k: v.get("lora_B_layout")
                    for k, v in detail["parameters"].items()
                    if isinstance(v, dict) and v.get("lora_B_layout") != "rank_major"
                }
                if bad_params:
                    raise RuntimeError(
                        f"CRITICAL MoE Adapter Config Error: `{cfg_file}` has non-rank_major "
                        f"expert parameters: {bad_params}"
                    )

    summary = {
        "is_moe": True,
        "trainable_params": trainable_params,
        "num_expert_wrappers": len(expert_wrappers),
        "lora_b_layout": resolved_layout,
        "expert_lora_a_l2_norm": round(lora_a_norm, 6),
        "expert_lora_b_l2_norm": round(lora_b_norm, 6),
    }
    logger.info(
        "[MoE LoRA Integrity Verified] trainable_params=%s | expert_wrappers=%d | layout=%s | lora_A_norm=%.4f | lora_B_norm=%.4f",
        f"{trainable_params:,}",
        len(expert_wrappers),
        resolved_layout,
        lora_a_norm,
        lora_b_norm,
    )
    return summary


def verify_saved_moe_adapter_config(
    output_dir: Union[str, pathlib.Path],
    is_moe: bool = False,
) -> None:
    """Ensure saved MoE `adapter_config.json` explicitly records `lora_B_layout='rank_major'` and `target_parameters`."""
    if not is_moe:
        return
    cfg_path = pathlib.Path(output_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Saved MoE adapter_config.json missing at {cfg_path}")
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    target_params = cfg_data.get("target_parameters") or []
    if not target_params:
        raise RuntimeError(
            f"Saved MoE adapter_config.json at {cfg_path} is missing 3D `target_parameters`!"
        )
    existing_layout = cfg_data.get("lora_B_layout")
    if existing_layout is not None and existing_layout != "rank_major":
        raise RuntimeError(
            f"Saved MoE adapter_config.json at {cfg_path} wrote unexpected lora_B_layout={existing_layout!r}!"
        )
    if existing_layout is None:
        cfg_data["lora_B_layout"] = "rank_major"
        atomic_write_json(cfg_path, cfg_data)
