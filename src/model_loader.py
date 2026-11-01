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

import gc
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import unsloth  # type: ignore # noqa: F401
except ImportError:
    pass

import torch
import torch.nn as nn

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
    exclude_list = list(exclude_modules)

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
        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=lora_alpha,
            use_rslora=use_rslora,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_list,
            exclude_modules=exclude_list,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
        )
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

        exclude_regex = (
            r".*(" + "|".join(exclude_list) + r").*"
            if exclude_list
            else None
        )
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
        return get_peft_model(model, lora_cfg)


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
        if enable_cpu_offload:
            quant_cfg = getattr(hf_config, "quantization_config", None)
            if isinstance(quant_cfg, dict):
                quant_cfg["llm_int8_enable_fp32_cpu_offload"] = True
            elif quant_cfg is not None and hasattr(quant_cfg, "llm_int8_enable_fp32_cpu_offload"):
                quant_cfg.llm_int8_enable_fp32_cpu_offload = True
    except Exception as cfg_exc:
        logger.debug("AutoConfig preload skipped for '%s': %s", model_name, cfg_exc)

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
        if hf_config is not None:
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
        if is_multimodal_arch:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **load_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
            )
        if adapter_checkpoint is not None:
            from peft import PeftModel

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
            model = PeftModel.from_pretrained(
                model,
                adapter_checkpoint,
                is_trainable=is_trainable,
            )

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

    log_vram_telemetry(model, stage="post_model_load")
    return model, tokenizer, offload_state


load_gemma4_model_and_tokenizer = load_gemma4_8gb_model_and_tokenizer
