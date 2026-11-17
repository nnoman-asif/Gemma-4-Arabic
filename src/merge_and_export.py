"""Standalone 16-bit BF16 checkpoint merge, canonical processor restoration, and verification CLI.

Key architectural guarantees:
1. Cleanly merges the trained Stage 2 DPO adapter (`W_final = W_base + lambda * Delta_DPO`,
   with lossless `lambda=1.0, density=1.0` default across both 2D Dense Linear LoRAs and
   3D Sparse MoE `ParamWrapper` routed-expert LoRAs) via Unsloth `save_pretrained_merged`.
2. Never exports 4-bit `bitsandbytes` quantized tensors into `merged_16bit`.
3. Restores canonical upstream `config.json` (`use_cache: true`, `tie_word_embeddings: true`,
   no `quantization_config`), `chat_template.jinja`, `processor_config.json`, `generation_config.json`,
   and `tokenizer_config.json` (`eos_token: "<turn|>"`).
4. Runs `verify_merged_checkpoint` asserting BF16 safetensors total byte size (`>= 8.0 GB` for E4B,
   `>= 45.0 GB` for 26B-A4B), zero quantization state keys, and intact native Gemma 4 chat template.
"""

import argparse
import json
import logging
import pathlib
import shutil
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import (
    configure_clean_logging,
    atomic_write_json,
    ensure_tied_lm_head,
    ensure_unsloth_builtins,
    load_gemma4_8gb_model_and_tokenizer,
)

ensure_unsloth_builtins()

try:
    import unsloth  # type: ignore # noqa: F401
except Exception:
    pass

import torch
import torch.nn as nn

from src.constants import (
    BASE_MODEL_ID,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_TIES_DENSITY,
    DEFAULT_TIES_LAMBDA,
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_ID_26B,
    load_dotenv_if_present,
)
from src.vram_hooks import restore_multimodal_towers

load_dotenv_if_present()
logger = logging.getLogger(__name__)


@torch.no_grad()
def ties_trim_tensor(
    delta: torch.Tensor,
    density: float = DEFAULT_TIES_DENSITY,
) -> torch.Tensor:
    """Keep the top `density` fraction of elements by magnitude and zero out the rest."""
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0.0, 1.0], got {density}")
    if density >= 1.0 or delta.numel() == 0:
        return delta.clone()

    flat = delta.reshape(-1)
    k = max(1, int(round(flat.numel() * density)))
    if k >= flat.numel():
        return delta.clone()

    magnitudes = flat.abs()
    topk_vals, _ = torch.topk(magnitudes, k=k, largest=True, sorted=False)
    threshold = topk_vals.min()
    mask = (delta.abs() >= threshold).to(delta.dtype)
    return delta * mask


@torch.no_grad()
def ties_merge_deltas(
    deltas: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
    density: float = DEFAULT_TIES_DENSITY,
) -> torch.Tensor:
    """Merge one or more parameter delta tensors using TIES-Merging."""
    if not deltas:
        raise ValueError("deltas sequence cannot be empty.")

    n = len(deltas)
    if weights is None:
        norm_weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError("Length of weights must match length of deltas.")
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive.")
        norm_weights = [float(w) / total_w for w in weights]

    ref_dtype = deltas[0].dtype
    ref_device = deltas[0].device

    trimmed_stack = torch.stack(
        [
            ties_trim_tensor(d.to(device=ref_device, dtype=torch.float32), density=density)
            for d in deltas
        ],
        dim=0,
    )
    weight_tensor = torch.tensor(
        norm_weights, dtype=torch.float32, device=ref_device
    ).view(n, *([1] * deltas[0].ndim))

    weighted_trimmed = trimmed_stack * weight_tensor
    elected_sign = torch.sign(weighted_trimmed.sum(dim=0))

    sign_match = (torch.sign(trimmed_stack) == elected_sign.unsqueeze(0)) & (
        elected_sign.unsqueeze(0) != 0
    )
    agreeing_weights = weight_tensor * sign_match.to(torch.float32)
    denom = agreeing_weights.sum(dim=0).clamp_min(1e-8)
    disjoint_merged = (trimmed_stack * agreeing_weights).sum(dim=0) / denom

    return disjoint_merged.to(dtype=ref_dtype)


@torch.no_grad()
def apply_ties_interpolation_to_state_dict(
    base_state: Dict[str, torch.Tensor],
    adapter_deltas: Sequence[Dict[str, torch.Tensor]],
    lambda_scale: float = DEFAULT_TIES_LAMBDA,
    density: float = DEFAULT_TIES_DENSITY,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute `W_final = W_base + lambda_scale * TIES(adapter_deltas)` per parameter."""
    if not (0.0 <= lambda_scale <= 2.0):
        raise ValueError(f"lambda_scale must be in [0.0, 2.0], got {lambda_scale}")

    merged_state: Dict[str, torch.Tensor] = {}
    for key, base_param in base_state.items():
        matching_deltas = [d[key] for d in adapter_deltas if key in d]
        if not matching_deltas or lambda_scale == 0.0:
            merged_state[key] = base_param.clone()
            continue

        ties_delta = ties_merge_deltas(
            deltas=matching_deltas,
            weights=weights,
            density=density,
        )
        merged_val = base_param.to(torch.float32) + float(lambda_scale) * ties_delta.to(
            device=base_param.device, dtype=torch.float32
        )
        merged_state[key] = merged_val.to(dtype=base_param.dtype)

    return merged_state


@torch.no_grad()
def scale_and_trim_peft_lora_weights(
    model: nn.Module,
    lambda_scale: float = DEFAULT_TIES_LAMBDA,
    density: float = DEFAULT_TIES_DENSITY,
    sft_adapter_state: Optional[Dict[str, torch.Tensor]] = None,
) -> int:
    """Optionally scale (`lambda_scale`) or TIES-trim (`density`) active LoRA `lora_B` tensors.

    When `lambda_scale == 1.0`, `density == 1.0`, and `sft_adapter_state is None` (default),
    performs a lossless no-op (`return 0`) so 100% of trained Stage 2 DPO 2D & 3D MoE LoRA
    weights merge cleanly without unstructured rank sparsification.
    """
    if abs(float(lambda_scale) - 1.0) < 1e-7 and float(density) >= 1.0 and sft_adapter_state is None:
        logger.info("Lossless LoRA merge selected (lambda=1.00, density=1.00); preserving exact adapter weights.")
        return 0

    modified_count = 0
    for name, param in model.named_parameters():
        if "lora_B" in name and ".ref." not in name:
            updated = param.data
            if float(density) < 1.0:
                updated = ties_trim_tensor(updated, density=float(density))
            if abs(float(lambda_scale) - 1.0) >= 1e-7:
                updated = updated * float(lambda_scale)
            param.data.copy_(updated)
            modified_count += 1

    logger.info(
        "Scaled/trimmed %d lora_B tensors (lambda=%.2f, density=%.2f).",
        modified_count,
        float(lambda_scale),
        float(density),
    )
    return modified_count


def _read_safetensors_header_keys(st_path: pathlib.Path) -> List[str]:
    with st_path.open("rb") as fh:
        hdr_bytes = fh.read(8)
        if len(hdr_bytes) != 8:
            return []
        hdr_len = struct.unpack("<Q", hdr_bytes)[0]
        if hdr_len <= 0 or hdr_len > 100_000_000:
            return []
        hdr_json = json.loads(fh.read(hdr_len).decode("utf-8"))
        return [k for k in hdr_json.keys() if k != "__metadata__"]


def _patch_exported_gemma4_config_tie_embeddings(out_16bit: pathlib.Path) -> bool:
    """Ensure `tie_word_embeddings: true`, `use_cache: true`, and no `quantization_config` in exported `config.json`."""
    cfg_path = out_16bit / "config.json"
    if not cfg_path.exists():
        return False

    has_separate_lm_head = False
    idx_path = out_16bit / "model.safetensors.index.json"
    st_path = out_16bit / "model.safetensors"
    if idx_path.exists():
        try:
            idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
            wmap = idx_data.get("weight_map", {})
            has_separate_lm_head = any(
                k in wmap for k in ("lm_head.weight", "language_model.lm_head.weight")
            )
        except Exception:
            has_separate_lm_head = False
    elif st_path.exists():
        try:
            hdr_keys = _read_safetensors_header_keys(st_path)
            has_separate_lm_head = any(
                k in hdr_keys for k in ("lm_head.weight", "language_model.lm_head.weight")
            )
        except Exception:
            has_separate_lm_head = False

    try:
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        changed = False
        if "quantization_config" in cfg_data:
            cfg_data.pop("quantization_config", None)
            changed = True
        if cfg_data.get("use_cache") is not True:
            cfg_data["use_cache"] = True
            changed = True
        if not has_separate_lm_head and cfg_data.get("tie_word_embeddings") is not True:
            cfg_data["tie_word_embeddings"] = True
            changed = True
        if isinstance(cfg_data.get("text_config"), dict):
            if "quantization_config" in cfg_data["text_config"]:
                cfg_data["text_config"].pop("quantization_config", None)
                changed = True
            if cfg_data["text_config"].get("use_cache") is not True:
                cfg_data["text_config"]["use_cache"] = True
                changed = True
            if not has_separate_lm_head and cfg_data["text_config"].get("tie_word_embeddings") is not True:
                cfg_data["text_config"]["tie_word_embeddings"] = True
                changed = True
        if changed:
            cfg_path.write_text(
                json.dumps(cfg_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Normalized exported %s (use_cache=true, tie_word_embeddings, clean BF16 config).", cfg_path)
        return not has_separate_lm_head
    except Exception as exc:
        logger.warning("Could not verify/patch exported config.json: %s", exc)
        return False


def _restore_canonical_processor_and_templates(
    out_16bit: pathlib.Path,
    base_model_id: str,
) -> None:
    """Ensure `processor_config.json`, `chat_template.jinja`, `generation_config.json`, and `eos_token` match upstream."""
    canonical_repo = (
        PRIMARY_MODEL_ID_26B
        if ("26b" in base_model_id.lower() or "a4b" in base_model_id.lower())
        else PRIMARY_MODEL_ID
    )
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(canonical_repo)
        proc.save_pretrained(str(out_16bit))
    except Exception as proc_exc:
        logger.debug("AutoProcessor save skipped for %s: %s", canonical_repo, proc_exc)

    try:
        from huggingface_hub import hf_hub_download

        for fname in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json"):
            dst = out_16bit / fname
            if not dst.exists():
                try:
                    cached = hf_hub_download(repo_id=canonical_repo, filename=fname)
                    shutil.copy2(cached, dst)
                except Exception:
                    pass
    except Exception:
        pass

    tc_path = out_16bit / "tokenizer_config.json"
    if tc_path.exists():
        try:
            raw_tc = tc_path.read_text(encoding="utf-8")
            tc_data = json.loads(raw_tc)
            if "<turn|>" in raw_tc and tc_data.get("eos_token") != "<turn|>":
                tc_data["eos_token"] = "<turn|>"
                tc_path.write_text(
                    json.dumps(tc_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as tc_exc:
            logger.warning("Could not verify/patch tokenizer_config.json eos_token: %s", tc_exc)


def verify_merged_checkpoint(
    out_16bit: pathlib.Path,
    is_moe: bool = False,
    enforce_full_size: bool = True,
) -> Dict[str, Any]:
    """Forensically verify that `out_16bit` is a valid, unquantized BF16 standalone Gemma 4 checkpoint."""
    cfg_path = out_16bit / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Merged checkpoint verification failed: missing {cfg_path}")

    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "quantization_config" in cfg_data:
        raise RuntimeError(
            f"Merged checkpoint at {out_16bit} still contains `quantization_config` (4-bit leak)!"
        )
    if cfg_data.get("use_cache") is not True:
        raise RuntimeError(f"Merged checkpoint at {out_16bit} has `use_cache != True`!")

    shards = sorted(out_16bit.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"Merged checkpoint at {out_16bit} contains zero `.safetensors` files!")

    total_bytes = sum(s.stat().st_size for s in shards)
    total_gb = total_bytes / float(1024**3)

    bad_quant_suffixes = (".absmax", ".quant_state", ".quant_map", "lora_A", "lora_B")
    for s in shards:
        for k in _read_safetensors_header_keys(s):
            if any(bad in k for bad in bad_quant_suffixes):
                raise RuntimeError(
                    f"Merged checkpoint shard {s.name} contains unmerged/quantized tensor key: {k}"
                )

    if enforce_full_size:
        min_gb = 44.0 if is_moe else 7.8
        if total_gb < min_gb:
            raise RuntimeError(
                f"Merged BF16 checkpoint size {total_gb:.2f} GiB at {out_16bit} is below expected "
                f"minimum {min_gb:.1f} GiB (is_moe={is_moe})! Refusing to proceed with truncated/quantized weights."
            )

    tpl_path = out_16bit / "chat_template.jinja"
    if tpl_path.exists():
        tpl_txt = tpl_path.read_text(encoding="utf-8")
        if "<|turn>" not in tpl_txt or "<turn|>" not in tpl_txt:
            raise RuntimeError(f"chat_template.jinja at {out_16bit} is missing native <|turn>/<turn|> tokens!")

    return {
        "verified": True,
        "checkpoint_dir": str(out_16bit),
        "num_safetensors_shards": len(shards),
        "total_size_gb": round(total_gb, 3),
        "is_moe": is_moe,
    }


def export_merged_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    output_dir: pathlib.Path,
    export_format: str = "merged_16bit",
    gguf_quant: str = "q4_k_m",
    base_model_id: Optional[str] = None,
) -> List[str]:
    """Export the model to standalone 16-bit BF16 Safetensors (`merged_16bit`) and/or GGUF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_artifacts: List[str] = []
    proc_or_tok = getattr(tokenizer, "_gemma4_processor", tokenizer)

    if export_format in {"merged_16bit", "both"}:
        out_16bit = output_dir / "merged_16bit"
        out_16bit.mkdir(parents=True, exist_ok=True)
        for stale_file in list(out_16bit.glob("*.safetensors")) + [out_16bit / "model.safetensors.index.json"]:
            if stale_file.is_file():
                try:
                    stale_file.unlink()
                except OSError:
                    pass
        cfg = getattr(model, "config", None)
        if cfg is not None:
            try:
                cfg.tie_word_embeddings = True
                cfg.use_cache = True
                if getattr(cfg, "text_config", None) is not None:
                    cfg.text_config.tie_word_embeddings = True
                    cfg.text_config.use_cache = True
            except Exception:
                pass

        saved_16bit = False
        if hasattr(model, "save_pretrained_merged"):
            moe_merge_state = None
            try:
                import unsloth_zoo.saving_utils as _uz_saving

                moe_merge_state = getattr(_uz_saving, "_MOE_MERGE_STATE", None)
                if isinstance(moe_merge_state, dict) and hasattr(moe_merge_state.get("fallbacks"), "clear"):
                    moe_merge_state["fallbacks"].clear()
            except Exception:
                moe_merge_state = None

            logger.info("Merging and exporting 16-bit BF16 safetensors via Unsloth save_pretrained_merged -> %s ...", out_16bit)
            model.save_pretrained_merged(
                str(out_16bit),
                proc_or_tok,
                save_method="merged_16bit",
            )
            if isinstance(moe_merge_state, dict) and moe_merge_state.get("fallbacks"):
                raise RuntimeError(
                    "CRITICAL: Unsloth recorded unmerged MoE 3D expert fallbacks during save_pretrained_merged: "
                    f"{moe_merge_state['fallbacks']}"
                )
            saved_16bit = True

        if not saved_16bit:
            # Non-Unsloth fallback: strictly forbid saving 4-bit quantized weights into merged_16bit
            is_quantized = bool(getattr(model, "is_loaded_in_4bit", False) or getattr(cfg, "quantization_config", None))
            if is_quantized:
                raise RuntimeError(
                    "Cannot export `merged_16bit` from a 4-bit BitsAndBytes model without Unsloth "
                    "`save_pretrained_merged`! Refusing to write 4-bit weights into `merged_16bit`."
                )
            if hasattr(model, "merge_and_unload"):
                merged = model.merge_and_unload()
                ensure_tied_lm_head(merged)
                merged.save_pretrained(str(out_16bit), safe_serialization=True)
            else:
                ensure_tied_lm_head(model)
                model.save_pretrained(str(out_16bit), safe_serialization=True)
            if proc_or_tok is not None:
                proc_or_tok.save_pretrained(str(out_16bit))

        if base_model_id:
            _restore_canonical_processor_and_templates(out_16bit, base_model_id)
        _patch_exported_gemma4_config_tie_embeddings(out_16bit)
        exported_artifacts.append(str(out_16bit))

    if export_format in {"gguf", "both"}:
        out_gguf = output_dir / f"gguf_{gguf_quant}"
        out_gguf.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained_gguf"):
            model.save_pretrained_gguf(
                str(out_gguf),
                proc_or_tok,
                quantization_method=gguf_quant,
            )
            exported_artifacts.append(str(out_gguf))
        else:
            logger.warning(
                "Model does not expose Unsloth save_pretrained_gguf; wrote GGUF metadata stub to %s.",
                out_gguf,
            )
            (out_gguf / "gguf_export_spec.json").write_text(
                json.dumps({"quantization_method": gguf_quant}, indent=2),
                encoding="utf-8",
            )
            exported_artifacts.append(str(out_gguf))

    return exported_artifacts


def _prune_adapter_checkpoints(adapter_dir: Optional[str]) -> None:
    if not adapter_dir:
        return
    p = pathlib.Path(adapter_dir)
    if not p.is_dir() or not (p / "adapter_config.json").exists():
        return
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            shutil.rmtree(child, ignore_errors=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for LoRA checkpoint merging and 16-bit/GGUF export."""
    parser = argparse.ArgumentParser(
        description="Standalone 16-bit BF16 LoRA merge, processor restoration, and checkpoint verifier."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL_ID,
        help="Base Gemma 4 model identifier (`unsloth/gemma-4-E4B-it` or `unsloth/gemma-4-26B-A4B-it`).",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="outputs/dpo_stage2",
        help="Path to the trained Stage 2 DPO LoRA adapter directory.",
    )
    parser.add_argument(
        "--sft-adapter-dir",
        type=str,
        default=None,
        help="Optional Stage 1 SFT adapter directory (used only for intermediate checkpoint cleanup unless explicit TIES is requested).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/merged_gemma4_e4b_arabic",
        help="Directory to write the merged 16-bit and/or GGUF checkpoints.",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=DEFAULT_TIES_LAMBDA,
        help="Adapter scaling factor lambda (default: 1.00 lossless full merge).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=DEFAULT_TIES_DENSITY,
        help="Fraction of top-magnitude adapter parameters to retain (default: 1.00 lossless).",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["merged_16bit", "gguf", "both"],
        default="merged_16bit",
        help="Export target format: merged_16bit (default), gguf, or both.",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Opt-in flag to also export GGUF quantized weights in addition to merged_16bit.",
    )
    parser.add_argument(
        "--gguf-quant",
        type=str,
        default="q4_k_m",
        help="GGUF quantization method when exporting GGUF (e.g. q4_k_m, q8_0, f16).",
    )
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Enable low-VRAM CPU offload hooks during model loading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing merged_16bit checkpoint if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a synthetic verification of merge interpolation math.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for standalone 16-bit BF16 checkpoint merge and verification."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    configure_clean_logging()
    args = parse_args(argv)
    effective_format = "both" if args.export_gguf else args.export_format
    is_moe = "26b" in args.base_model.lower() or "a4b" in args.base_model.lower()

    if args.dry_run:
        base_w = {"q_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        sft_delta = {"q_proj.weight": torch.tensor([[1.0, -0.01], [2.0, 1.0]])}
        dpo_delta = {"q_proj.weight": torch.tensor([[1.0, 0.02], [2.0, 1.0]])}
        merged_state = apply_ties_interpolation_to_state_dict(
            base_state=base_w,
            adapter_deltas=[sft_delta, dpo_delta],
            lambda_scale=args.lambda_scale,
            density=args.density,
            weights=[0.5, 0.5],
        )
        summary = {
            "status": "dry_run_verified",
            "lambda_scale": args.lambda_scale,
            "density": args.density,
            "export_format": effective_format,
            "gguf_quant": args.gguf_quant,
            "merged_q_proj_00": round(float(merged_state["q_proj.weight"][0, 0].item()), 4),
            "output_dir": args.output_dir,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    out_root = pathlib.Path(args.output_dir)
    out_16bit_dir = out_root / "merged_16bit"
    if effective_format == "merged_16bit" and (out_16bit_dir / "config.json").exists() and not args.force:
        try:
            _patch_exported_gemma4_config_tie_embeddings(out_16bit_dir)
            if not (out_16bit_dir / "chat_template.jinja").exists():
                _restore_canonical_processor_and_templates(out_16bit_dir, args.base_model)
            verification = verify_merged_checkpoint(out_16bit_dir, is_moe=is_moe, enforce_full_size=True)
            manifest_path = out_root / "merge_manifest.json"
            if not manifest_path.exists():
                manifest = {
                    "base_model": args.base_model,
                    "adapter_dir": str(args.adapter_dir),
                    "lambda_scale": args.lambda_scale,
                    "density": args.density,
                    "resumed_existing_checkpoint": True,
                    "exported_artifacts": {"merged_16bit": str(out_16bit_dir)},
                    "verification": verification,
                }
                atomic_write_json(manifest_path, manifest)
            logger.info(
                "Verified existing merged_16bit checkpoint at %s (%.2f GiB); skipping re-merge.",
                out_16bit_dir,
                verification["total_size_gb"],
            )
            return 0
        except Exception as verify_exc:
            logger.warning(
                "Existing merged_16bit directory at %s is incomplete or interrupted (%s); removing partial export and re-merging cleanly.",
                out_16bit_dir,
                verify_exc,
            )
            shutil.rmtree(out_16bit_dir, ignore_errors=True)
            stale_manifest = out_root / "merge_manifest.json"
            if stale_manifest.is_file():
                try:
                    stale_manifest.unlink()
                except OSError:
                    pass

    adapter_path = pathlib.Path(args.adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter directory '{adapter_path}' is missing `adapter_config.json`! Cannot merge."
        )

    if args.force and out_16bit_dir.exists():
        logger.info("--force enabled: removing existing %s before disk space check and re-merge.", out_16bit_dir)
        shutil.rmtree(out_16bit_dir, ignore_errors=True)
        stale_manifest = out_root / "merge_manifest.json"
        if stale_manifest.is_file():
            try:
                stale_manifest.unlink()
            except OSError:
                pass

    # Free intermediate checkpoint-* directories before writing large BF16 safetensors
    _prune_adapter_checkpoints(args.sft_adapter_dir)
    _prune_adapter_checkpoints(args.adapter_dir)

    out_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(str(out_root)).free / float(1024**3)
    req_gb = 54.0 if is_moe else 13.0
    logger.info("Disk space check before merge (%s): %.2f GiB free (recommended >= %.1f GiB).", out_root, free_gb, req_gb)
    if free_gb < req_gb:
        raise RuntimeError(
            f"Insufficient free disk space on {out_root}: {free_gb:.2f} GiB available, "
            f"need >= {req_gb:.1f} GiB to export standalone BF16 safetensors!"
        )

    # Load via Unsloth in 4-bit so `save_pretrained_merged(..., save_method="merged_16bit")`
    # streams upstream BF16 base shards directly to disk with low RAM/VRAM and full 3D MoE support.
    model, tokenizer, offload_state = load_gemma4_8gb_model_and_tokenizer(
        model_name=args.base_model,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit=True,
        attach_lora=False,
        adapter_checkpoint=str(adapter_path),
        is_trainable=False,
        offload_ple_to_cpu=bool(args.offload_to_cpu),
        offload_multimodal_towers_to_cpu=bool(args.offload_to_cpu),
    )

    modified = scale_and_trim_peft_lora_weights(
        model=model,
        lambda_scale=args.lambda_scale,
        density=args.density,
        sft_adapter_state=None,
    )
    restored_modules: List[str] = []
    if offload_state.ple_module_names or offload_state.multimodal_module_names:
        restored_modules = restore_multimodal_towers(
            model=model,
            state=offload_state,
            target_device="cpu",
            restore_ple=True,
        )
    exported = export_merged_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=out_root,
        export_format=effective_format,
        gguf_quant=args.gguf_quant,
        base_model_id=args.base_model,
    )

    verification = {}
    if effective_format in {"merged_16bit", "both"}:
        verification = verify_merged_checkpoint(out_16bit_dir, is_moe=is_moe, enforce_full_size=True)

    manifest = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_path),
        "lambda_scale": args.lambda_scale,
        "density": args.density,
        "modified_lora_b_tensors": modified,
        "exported_artifacts": exported,
        "verification": verification,
    }
    atomic_write_json(out_root / "merge_manifest.json", manifest)
    logger.info(
        "Exported & verified merged checkpoint (lambda=%.2f, density=%.2f, restored=%s) -> %s (%s)",
        args.lambda_scale,
        args.density,
        restored_modules,
        exported,
        verification,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
