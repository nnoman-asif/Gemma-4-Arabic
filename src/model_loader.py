"""Unsloth 4-bit NF4 model loader and rsLoRA adapter injector for Gemma-4-Arabic.

Integrates `src/vram_hooks.py` to enforce the 8.0 GB GPU VRAM budget by:
1. Mapping `embed_tokens_per_layer`, `audio_tower`, and `vision_tower` to CPU RAM during load.
2. Registering the non-blocking CPU lookup forward hook on `embed_tokens_per_layer` (5.637 GB).
3. Offloading `audio_tower` (0.618 GB) and `vision_tower` (0.339 GB) to CPU RAM.
4. Freezing token embeddings (`embed_tokens`, `lm_head`, `embed_tokens_per_layer`) to preserve
   Gemma 4's 262,144 vocabulary (including 8,460 native Arabic tokens).
5. Attaching rsLoRA (`use_rslora=True`, `r=32`, `lora_alpha=64`, `lora_dropout=0.0`, `bias="none"`)
   across all 7 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
   `up_proj`, `down_proj`) with `use_gradient_checkpointing="unsloth"`.
"""

import builtins
import gc
import inspect
import logging
import os
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
)
from src.vram_hooks import (
    OffloadState,
    apply_8gb_vram_optimizations,
    log_vram_telemetry,
    offload_multimodal_towers_to_cpu,
    register_ple_cpu_offload_hook,
)

logger = logging.getLogger(__name__)


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
    if not torch.cuda.is_available():
        return {"": "cpu"}
    return {
        "model.language_model.embed_tokens_per_layer": "cpu",
        "model.audio_tower": "cpu",
        "model.vision_tower": "cpu",
        "model.embed_audio": "cpu",
        "model.embed_vision": "cpu",
        "": cuda_device_index,
    }


def build_device_map(
    cuda_device_index: int = 0,
    offload_ple_to_cpu: bool = True,
    offload_multimodal_towers_to_cpu: bool = True,
) -> Dict[str, Union[str, int]]:
    """Build hardware-aware device_map for either 8 GB Edge (CPU offload) or A100 80 GB (pure HBM2e)."""
    if not torch.cuda.is_available():
        return {"": "cpu"}
    if not offload_ple_to_cpu and not offload_multimodal_towers_to_cpu:
        return {"": cuda_device_index}
    dmap: Dict[str, Union[str, int]] = {"": cuda_device_index}
    if offload_ple_to_cpu:
        dmap["model.language_model.embed_tokens_per_layer"] = "cpu"
    if offload_multimodal_towers_to_cpu:
        dmap["model.audio_tower"] = "cpu"
        dmap["model.vision_tower"] = "cpu"
        dmap["model.embed_audio"] = "cpu"
        dmap["model.embed_vision"] = "cpu"
    return dmap


def build_bnb_4bit_config(
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    exclude_modules: Sequence[str] = LORA_EXCLUDE_MODULES,
) -> Any:
    """Build a BitsAndBytesConfig for 4-bit NF4 quantization."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
        llm_int8_skip_modules=list(exclude_modules),
    )


def freeze_embeddings_and_towers(
    model: nn.Module,
    freeze_moe_router: bool = True,
) -> List[str]:
    """Freeze input/output token embeddings, PLE tables, multimodal towers, and MoE routers."""
    frozen_names: List[str] = []
    freeze_tokens: List[str] = [
        "embed_tokens",
        "lm_head",
        PLE_MODULE_PATTERN,
        *MULTIMODAL_TOWER_PATTERNS,
        "multi_modal_projector",
    ]
    if freeze_moe_router:
        freeze_tokens.extend(MOE_ROUTER_KEYWORDS)
    for name, param in model.named_parameters():
        if any(tok in name for tok in freeze_tokens):
            param.requires_grad_(False)
            frozen_names.append(name)
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
    target_modules: Sequence[str] = LORA_TARGET_MODULES,
    exclude_modules: Sequence[str] = LORA_EXCLUDE_MODULES,
    use_gradient_checkpointing: Union[str, bool] = "unsloth",
    random_state: int = 3407,
) -> nn.Module:
    """Attach rsLoRA adapters across the 7 language linear projections.

    Uses Unsloth's `FastLanguageModel.get_peft_model` when available, with a
    clean fallback to Hugging Face PEFT `get_peft_model` + `LoraConfig`.
    """
    target_list = list(target_modules)
    exclude_list = list(dict.fromkeys([*LORA_EXCLUDE_MODULES, *exclude_modules]))
    exclude_regex = (
        r".*(" + "|".join(exclude_list) + r").*"
        if exclude_list
        else None
    )

    try:
        from unsloth import FastLanguageModel  # type: ignore

        logger.info(
            "Attaching rsLoRA via Unsloth FastLanguageModel.get_peft_model "
            "(r=%d, alpha=%d, use_rslora=%s, targets=%s).",
            r,
            lora_alpha,
            use_rslora,
            target_list,
        )
        unsloth_peft_kwargs: Dict[str, Any] = {
            "r": r,
            "lora_alpha": lora_alpha,
            "use_rslora": use_rslora,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "target_modules": target_list,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
        }
        try:
            sig = inspect.signature(FastLanguageModel.get_peft_model)
            if "finetune_vision_layers" in sig.parameters:
                unsloth_peft_kwargs["finetune_vision_layers"] = False
            if "finetune_language_layers" in sig.parameters:
                unsloth_peft_kwargs["finetune_language_layers"] = True
        except Exception:
            pass

        try:
            peft_model = FastLanguageModel.get_peft_model(
                model,
                exclude_modules=exclude_list,
                **unsloth_peft_kwargs,
            )
        except TypeError:
            peft_model = FastLanguageModel.get_peft_model(
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
        logger.info(
            "Unsloth get_peft_model unavailable or bypassed (%s); falling back to PEFT LoraConfig.",
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

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_rslora=use_rslora,
            target_modules=target_list,
            exclude_modules=exclude_regex,
            task_type="CAUSAL_LM",
        )
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
    resolved_targets = list(target_modules) if target_modules is not None else list(LORA_TARGET_MODULES)
    resolved_excludes = list(exclude_modules) if exclude_modules is not None else list(LORA_EXCLUDE_MODULES)
    load_path = adapter_checkpoint if adapter_checkpoint is not None else model_name

    model: nn.Module
    tokenizer: Any

    hf_config: Any = None
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name)
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
        from unsloth import FastLanguageModel  # type: ignore

        logger.info(
            "Loading '%s' via Unsloth FastLanguageModel (max_seq_length=%d, load_in_4bit=%s, cpu_offload=%s).",
            load_path,
            max_seq_length,
            load_in_4bit,
            enable_cpu_offload,
        )
        unsloth_kwargs: Dict[str, Any] = {
            "model_name": load_path,
            "max_seq_length": max_seq_length,
            "dtype": resolved_dtype,
            "load_in_4bit": load_in_4bit,
            "device_map": device_map,
        }
        if hf_config is not None and adapter_checkpoint is None:
            unsloth_kwargs["config"] = hf_config
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(**unsloth_kwargs)
        except TypeError:
            unsloth_kwargs.pop("config", None)
            model, tokenizer = FastLanguageModel.from_pretrained(**unsloth_kwargs)

        if hasattr(tokenizer, "tokenizer") and not hasattr(tokenizer, "encode"):
            inner_tok = tokenizer.tokenizer
            if (
                getattr(inner_tok, "chat_template", None) is None
                and getattr(tokenizer, "chat_template", None) is not None
            ):
                inner_tok.chat_template = tokenizer.chat_template
            tokenizer = inner_tok

        if adapter_checkpoint is not None and is_trainable:
            if hasattr(FastLanguageModel, "for_training"):
                try:
                    FastLanguageModel.for_training(
                        model,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                    )
                except TypeError:
                    FastLanguageModel.for_training(model)
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

    _configure_trainable_model_flags(model, is_trainable=is_trainable)

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
        )
        # Re-verify PLE, multimodal towers, and MoE routers remain frozen after PEFT wrapping
        freeze_embeddings_and_towers(model, freeze_moe_router=freeze_moe_router)
        setattr(model, "_vram_offload_state", offload_state)

    if not used_unsloth_loader and not getattr(model, "_is_unsloth_model", False):
        install_chunked_causal_lm_loss(model)
    _configure_trainable_model_flags(model, is_trainable=is_trainable)

    log_vram_telemetry(model, stage="post_model_load")
    return model, tokenizer, offload_state


load_gemma4_model_and_tokenizer = load_gemma4_8gb_model_and_tokenizer
