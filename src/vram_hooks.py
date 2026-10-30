"""8.0 GB VRAM optimization hooks for Gemma-4-Arabic.

Implements dynamic named_modules() discovery and CPU RAM forward hooks for:
1. `embed_tokens_per_layer` (5.637 GB frozen BF16 lookup table [262144, 10752]):
   - Moves embedding table to CPU RAM (`cpu`).
   - Performs token ID indexing on CPU via a pre-forward/forward hook pair.
   - Transfers only the looked-up per-layer embedding tensor ([B, L, 10752] or [B, L, 256])
     to the target CUDA device/dtype with `non_blocking=True`.
2. `audio_tower` (0.618 GB) and `vision_tower` (0.339 GB):
   - Dynamically locates multimodal towers via `model.named_modules()`.
   - Offloads them to CPU RAM during text SFT/DPO training.
   - Restores them cleanly before final checkpoint merge and export.
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from src.constants import (
    MULTIMODAL_TOWER_PATTERNS,
    PLE_MODULE_PATTERN,
    PLE_PER_LAYER_DIM,
)

logger = logging.getLogger(__name__)


@dataclass
class OffloadState:
    """Tracks hook handles and original placement of offloaded submodules."""

    ple_module_names: List[str] = field(default_factory=list)
    multimodal_module_names: List[str] = field(default_factory=list)
    original_devices: Dict[str, torch.device] = field(default_factory=dict)
    original_dtypes: Dict[str, torch.dtype] = field(default_factory=dict)
    hook_handles: List[Any] = field(default_factory=list)
    ple_bytes_offloaded: int = 0
    multimodal_bytes_offloaded: int = 0

    def remove_hooks(self) -> None:
        """Remove all registered forward and pre-forward hooks."""
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles.clear()


def _module_param_bytes(module: nn.Module) -> int:
    """Calculate total parameter and buffer memory footprint in bytes for a module."""
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True))
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers(recurse=True))
    return param_bytes + buffer_bytes


def _first_param_device_and_dtype(
    module: nn.Module,
) -> Tuple[torch.device, torch.dtype]:
    """Return the device and dtype of the first parameter or buffer in a module."""
    for p in module.parameters(recurse=True):
        return p.device, p.dtype
    for b in module.buffers(recurse=True):
        return b.device, b.dtype
    return torch.device("cpu"), torch.bfloat16


def find_modules_by_pattern(
    model: nn.Module,
    patterns: Sequence[str],
) -> Dict[str, nn.Module]:
    """Dynamically locate top-level matching submodules via `model.named_modules()`.

    When a module path matches a pattern (for example `model.audio_tower`), its child
    submodules (`model.audio_tower.layers.0`) are skipped so each tower or embedding
    table is handled as a single unit.
    """
    matched: Dict[str, nn.Module] = {}
    matched_prefixes: List[str] = []

    for name, module in model.named_modules():
        if not name:
            continue
        if any(name.startswith(prefix + ".") for prefix in matched_prefixes):
            continue
        parts = name.split(".")
        leaf = parts[-1]
        if any(pat == leaf or pat in parts for pat in patterns):
            matched[name] = module
            matched_prefixes.append(name)

    return matched


def register_ple_cpu_offload_hook(
    model: nn.Module,
    target_device: Optional[Union[str, torch.device]] = None,
    target_dtype: torch.dtype = torch.bfloat16,
    non_blocking: bool = True,
    pattern: str = PLE_MODULE_PATTERN,
    state: Optional[OffloadState] = None,
) -> OffloadState:
    """Move `embed_tokens_per_layer` to CPU RAM and register a CPU-lookup forward hook.

    During forward execution:
    1. Pre-hook captures the caller's `input_ids.device` (or explicit `target_device`)
       and moves `input_ids` to CPU RAM (`cpu`).
    2. `embed_tokens_per_layer` performs the frozen table lookup on CPU RAM.
    3. Forward hook transfers only the looked-up `[batch, seq_len, ...]` tensor
       (which slices into `[batch, seq_len, 256]` per layer across the 42 layers)
       to the target device and dtype with `non_blocking=True`.
    """
    if state is None:
        state = OffloadState()

    ple_modules = find_modules_by_pattern(model, [pattern])
    if not ple_modules:
        logger.warning("No module matching '%s' found in model.named_modules().", pattern)
        return state

    resolved_target_device = (
        torch.device(target_device) if target_device is not None else None
    )

    for name, ple_mod in ple_modules.items():
        orig_dev, orig_dtype = _first_param_device_and_dtype(ple_mod)
        state.original_devices[name] = orig_dev
        state.original_dtypes[name] = orig_dtype
        state.ple_module_names.append(name)
        state.ple_bytes_offloaded += _module_param_bytes(ple_mod)

        ple_mod.to("cpu")
        for param in ple_mod.parameters(recurse=True):
            param.requires_grad_(False)

        # Store runtime state on the module for input device tracking across pre-hook and post-hook
        ple_mod._ple_last_input_device = resolved_target_device or orig_dev
        ple_mod._ple_per_layer_dim = PLE_PER_LAYER_DIM

        def _pre_hook(
            mod: nn.Module,
            args: Tuple[Any, ...],
        ) -> Tuple[Any, ...]:
            if not args:
                return args
            first = args[0]
            if isinstance(first, torch.Tensor):
                if resolved_target_device is not None:
                    mod._ple_last_input_device = resolved_target_device
                else:
                    mod._ple_last_input_device = first.device
                if first.device.type != "cpu":
                    return (first.to(device="cpu", non_blocking=False),) + args[1:]
            return args

        def _post_hook(
            mod: nn.Module,
            args: Tuple[Any, ...],
            output: Any,
        ) -> Any:
            dest_device = getattr(mod, "_ple_last_input_device", resolved_target_device)
            if dest_device is None:
                dest_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            if isinstance(output, torch.Tensor):
                return output.to(
                    device=dest_device,
                    dtype=target_dtype,
                    non_blocking=non_blocking,
                )
            if isinstance(output, (tuple, list)):
                transferred = [
                    t.to(device=dest_device, dtype=target_dtype, non_blocking=non_blocking)
                    if isinstance(t, torch.Tensor)
                    else t
                    for t in output
                ]
                return type(output)(transferred)
            return output

        pre_handle = ple_mod.register_forward_pre_hook(_pre_hook)
        post_handle = ple_mod.register_forward_hook(_post_hook)
        state.hook_handles.extend([pre_handle, post_handle])

        logger.info(
            "Offloaded PLE module '%s' (%.3f GB) to CPU with non_blocking=%s forward hook.",
            name,
            _module_param_bytes(ple_mod) / (1024 ** 3),
            non_blocking,
        )

    return state


def offload_multimodal_towers_to_cpu(
    model: nn.Module,
    patterns: Sequence[str] = MULTIMODAL_TOWER_PATTERNS,
    state: Optional[OffloadState] = None,
) -> OffloadState:
    """Dynamically locate `audio_tower` and `vision_tower` and offload them to CPU RAM."""
    if state is None:
        state = OffloadState()

    towers = find_modules_by_pattern(model, patterns)
    for name, tower_mod in towers.items():
        orig_dev, orig_dtype = _first_param_device_and_dtype(tower_mod)
        state.original_devices[name] = orig_dev
        state.original_dtypes[name] = orig_dtype
        state.multimodal_module_names.append(name)
        mod_bytes = _module_param_bytes(tower_mod)
        state.multimodal_bytes_offloaded += mod_bytes

        tower_mod.to("cpu")
        for param in tower_mod.parameters(recurse=True):
            param.requires_grad_(False)

        logger.info(
            "Offloaded multimodal tower '%s' (%.3f GB) to CPU RAM.",
            name,
            mod_bytes / (1024 ** 3),
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return state


def restore_multimodal_towers(
    model: nn.Module,
    state: Optional[OffloadState] = None,
    target_device: Optional[Union[str, torch.device]] = None,
    target_dtype: Optional[torch.dtype] = None,
    restore_ple: bool = False,
) -> List[str]:
    """Restore offloaded `audio_tower` and `vision_tower` (and optionally PLE) before export."""
    restored: List[str] = []
    named_dict = dict(model.named_modules())

    if state is not None:
        candidates = list(state.multimodal_module_names)
        if restore_ple:
            state.remove_hooks()
            candidates.extend(state.ple_module_names)
    else:
        search_patterns = list(MULTIMODAL_TOWER_PATTERNS)
        if restore_ple:
            search_patterns.append(PLE_MODULE_PATTERN)
        candidates = list(find_modules_by_pattern(model, search_patterns).keys())

    for name in candidates:
        mod = named_dict.get(name)
        if mod is None:
            continue
        dest_dev = (
            torch.device(target_device)
            if target_device is not None
            else (state.original_devices.get(name, torch.device("cpu")) if state else torch.device("cpu"))
        )
        dest_dtype = (
            target_dtype
            if target_dtype is not None
            else (state.original_dtypes.get(name) if state else None)
        )
        if dest_dtype is not None:
            mod.to(device=dest_dev, dtype=dest_dtype)
        else:
            mod.to(device=dest_dev)
        restored.append(name)
        logger.info("Restored module '%s' to device=%s, dtype=%s.", name, dest_dev, dest_dtype)

    if state is None and hasattr(model, "_vram_offload_state"):
        attached_state: OffloadState = getattr(model, "_vram_offload_state")
        if restore_ple:
            attached_state.remove_hooks()

    return restored


def apply_8gb_vram_optimizations(
    model: nn.Module,
    target_device: Optional[Union[str, torch.device]] = None,
    target_dtype: torch.dtype = torch.bfloat16,
    non_blocking: bool = True,
) -> OffloadState:
    """Apply both PLE CPU forward hook and multimodal tower CPU offload in one call."""
    state = OffloadState()
    register_ple_cpu_offload_hook(
        model=model,
        target_device=target_device,
        target_dtype=target_dtype,
        non_blocking=non_blocking,
        state=state,
    )
    offload_multimodal_towers_to_cpu(
        model=model,
        patterns=MULTIMODAL_TOWER_PATTERNS,
        state=state,
    )
    setattr(model, "_vram_offload_state", state)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def get_vram_telemetry(model: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Collect GPU VRAM and CPU/GPU parameter memory telemetry."""
    cuda_available = torch.cuda.is_available()
    allocated_gb = (
        torch.cuda.memory_allocated() / (1024 ** 3) if cuda_available else 0.0
    )
    reserved_gb = (
        torch.cuda.memory_reserved() / (1024 ** 3) if cuda_available else 0.0
    )
    max_allocated_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if cuda_available else 0.0
    )

    cpu_param_bytes = 0
    gpu_param_bytes = 0
    trainable_params = 0
    total_params = 0

    if model is not None:
        seen_data_ptrs = set()
        for param in model.parameters():
            ptr = param.data_ptr()
            if ptr in seen_data_ptrs:
                continue
            seen_data_ptrs.add(ptr)
            n_params = param.numel()
            p_bytes = n_params * param.element_size()
            total_params += n_params
            if param.requires_grad:
                trainable_params += n_params
            if param.device.type == "cpu":
                cpu_param_bytes += p_bytes
            else:
                gpu_param_bytes += p_bytes

    return {
        "cuda_available": cuda_available,
        "gpu_allocated_gb": round(allocated_gb, 4),
        "gpu_reserved_gb": round(reserved_gb, 4),
        "gpu_max_allocated_gb": round(max_allocated_gb, 4),
        "cpu_param_gb": round(cpu_param_bytes / (1024 ** 3), 4),
        "gpu_param_gb": round(gpu_param_bytes / (1024 ** 3), 4),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio_pct": round(
            (100.0 * trainable_params / total_params) if total_params > 0 else 0.0, 4
        ),
    }


def log_vram_telemetry(
    model: Optional[nn.Module] = None,
    stage: str = "telemetry",
) -> Dict[str, Any]:
    """Compute and log VRAM and parameter telemetry at a pipeline checkpoint."""
    stats = get_vram_telemetry(model)
    logger.info(
        "[VRAM %s] GPU Alloc: %.3f GB | GPU Reserved: %.3f GB | "
        "GPU Params: %.3f GB | CPU Offloaded Params: %.3f GB | "
        "Trainable Params: %d / %d (%.4f%%)",
        stage,
        stats["gpu_allocated_gb"],
        stats["gpu_reserved_gb"],
        stats["gpu_param_gb"],
        stats["cpu_param_gb"],
        stats["trainable_params"],
        stats["total_params"],
        stats["trainable_ratio_pct"],
    )
    return stats
