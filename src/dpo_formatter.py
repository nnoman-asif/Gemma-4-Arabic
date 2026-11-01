"""Stage 2 DPO preference pair normalizers for Gemma-4-Arabic.

Implements schema rules and preprocessing validation for both Stage 2 DPO datasets:
1. `HeshamHaroon/arabic-dialect-dpo`:
   - MUST be loaded with `name="saudi"` (no default configuration exists).
   - Normalizes `prompt`, `chosen`, and `rejected` strings with Gemma 4 chat formatting.
2. `arcee-globe/arabic-orpo-dpo-mix-40k-filtered`:
   - Extracts the canonical user prompt from `chosen[0]["content"]` and overwrites
     `rejected[0]["content"]` so both preference branches share the exact same prompt string.
"""

import copy
import logging
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.chat_template import ensure_assistant_thinking_header, format_gemma4_chat
from src.constants import (
    ARABIC_DIALECT_DPO_REPO,
    DEFAULT_MSA_SYSTEM_PROMPT,
    DPO_DATASETS,
    END_OF_TURN_TOKEN,
    SAUDI_DIALECT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def _format_dpo_triplet(
    prompt_text: str,
    chosen_response: str,
    rejected_response: str,
    system_prompt: str,
    tokenizer: Optional[Any] = None,
    enable_thinking: bool = False,
) -> Dict[str, str]:
    """Format a (prompt, chosen, rejected) triplet with Gemma 4 chat template rules."""
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]
    formatted_prompt = format_gemma4_chat(
        messages=prompt_messages,
        tokenizer=tokenizer,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        enforce_thinking_verification=False,
    )
    formatted_chosen = (
        ensure_assistant_thinking_header(chosen_response, enable_thinking=enable_thinking)
        + f"{END_OF_TURN_TOKEN}\n"
    )
    formatted_rejected = (
        ensure_assistant_thinking_header(rejected_response, enable_thinking=enable_thinking)
        + f"{END_OF_TURN_TOKEN}\n"
    )
    return {
        "prompt": formatted_prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected,
    }


def normalize_arabic_dialect_dpo_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    enable_thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """Normalize a preference pair from `HeshamHaroon/arabic-dialect-dpo` (`name="saudi"`)."""
    prompt_text = str(row.get("prompt") or row.get("question") or "").strip()
    chosen_text = str(row.get("chosen") or "").strip()
    rejected_text = str(row.get("rejected") or "").strip()

    if not prompt_text or not chosen_text or not rejected_text:
        return None
    if chosen_text == rejected_text:
        return None

    triplet = _format_dpo_triplet(
        prompt_text=prompt_text,
        chosen_response=chosen_text,
        rejected_response=rejected_text,
        system_prompt=SAUDI_DIALECT_SYSTEM_PROMPT,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )

    chosen_messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": chosen_text},
    ]
    rejected_messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": rejected_text},
    ]

    return {
        "source": "arabic_dialect_dpo_saudi",
        "config_name": "saudi",
        "prompt": triplet["prompt"],
        "chosen": triplet["chosen"],
        "rejected": triplet["rejected"],
        "raw_prompt": prompt_text,
        "chosen_messages": chosen_messages,
        "rejected_messages": rejected_messages,
        "enable_thinking": enable_thinking,
    }


def normalize_arcee_arabic_dpo_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    enable_thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """Normalize a preference pair from `arcee-globe/arabic-orpo-dpo-mix-40k-filtered`.

    Extracts the user prompt from `chosen[0]["content"]` and overwrites
    `rejected[0]["content"]` so both branches share the exact same prompt string.
    """
    raw_chosen = row.get("chosen")
    raw_rejected = row.get("rejected")

    if not isinstance(raw_chosen, list) or len(raw_chosen) < 2:
        return None
    if not isinstance(raw_rejected, list) or len(raw_rejected) < 2:
        return None

    chosen_list = copy.deepcopy(raw_chosen)
    rejected_list = copy.deepcopy(raw_rejected)

    if not isinstance(chosen_list[0], dict) or not isinstance(rejected_list[0], dict):
        return None

    unified_prompt = str(chosen_list[0].get("content") or "").strip()
    if not unified_prompt:
        return None

    # Critical schema rule: overwrite rejected[0]["content"] with chosen[0]["content"]
    rejected_list[0]["content"] = unified_prompt
    chosen_list[0]["content"] = unified_prompt

    chosen_response = str(chosen_list[-1].get("content") or "").strip()
    rejected_response = str(rejected_list[-1].get("content") or "").strip()

    if not chosen_response or not rejected_response:
        return None
    if chosen_response == rejected_response:
        return None

    system_prompt = str(row.get("system") or "").strip() or DEFAULT_MSA_SYSTEM_PROMPT

    triplet = _format_dpo_triplet(
        prompt_text=unified_prompt,
        chosen_response=chosen_response,
        rejected_response=rejected_response,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )

    return {
        "source": "arabic_orpo_dpo_mix_40k",
        "config_name": "default",
        "prompt": triplet["prompt"],
        "chosen": triplet["chosen"],
        "rejected": triplet["rejected"],
        "raw_prompt": unified_prompt,
        "chosen_messages": chosen_list,
        "rejected_messages": rejected_list,
        "enable_thinking": enable_thinking,
    }


def normalize_dpo_iterable(
    source_key: str,
    rows: Iterable[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Normalize an iterable of raw preference rows from either Stage 2 DPO dataset."""
    dispatch: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
        "arabic_dialect_dpo_saudi": lambda r: normalize_arabic_dialect_dpo_row(
            r, tokenizer=tokenizer
        ),
        "arabic_orpo_dpo_mix_40k": lambda r: normalize_arcee_arabic_dpo_row(
            r, tokenizer=tokenizer
        ),
    }
    if source_key not in dispatch:
        raise KeyError(
            f"Unknown DPO source_key: {source_key!r}. Valid keys: {sorted(dispatch.keys())}"
        )

    normalizer = dispatch[source_key]
    results: List[Dict[str, Any]] = []
    try:
        for row in rows:
            try:
                item = normalizer(row)
            except Exception:
                continue
            if item is not None:
                results.append(item)
                if max_samples is not None and len(results) >= max_samples:
                    break
    except Exception:
        pass
    return results


def validate_dpo_load_config(repo_id: str, config_name: Optional[str]) -> str:
    """Enforce that `HeshamHaroon/arabic-dialect-dpo` is loaded with `name="saudi"`."""
    if repo_id == ARABIC_DIALECT_DPO_REPO:
        if config_name != "saudi":
            raise ValueError(
                f"Dataset '{ARABIC_DIALECT_DPO_REPO}' has no default config and MUST "
                f"be loaded with name='saudi' (got {config_name!r})."
            )
        return "saudi"
    return config_name or "default"


def load_and_normalize_dpo_source(
    source_key: str,
    max_samples: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    streaming: bool = True,
) -> List[Dict[str, Any]]:
    """Load and normalize a Stage 2 DPO dataset from Hugging Face."""
    from datasets import load_dataset

    if source_key not in DPO_DATASETS:
        raise KeyError(f"Unknown DPO source key: {source_key!r}")

    spec = DPO_DATASETS[source_key]
    raw_cfg = spec.config_names[0] if spec.config_names else None
    cfg_name = validate_dpo_load_config(spec.repo_id, raw_cfg)

    load_kwargs: Dict[str, Any] = {
        "path": spec.repo_id,
        "split": spec.split,
        "streaming": streaming,
    }
    if spec.repo_id == ARABIC_DIALECT_DPO_REPO or (cfg_name and cfg_name != "default"):
        load_kwargs["name"] = cfg_name

    ds = load_dataset(**load_kwargs)
    return normalize_dpo_iterable(
        source_key=source_key,
        rows=ds,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )


def mix_dpo_preference_datasets(
    saudi_dpo_samples: List[Dict[str, Any]],
    arcee_dpo_samples: List[Dict[str, Any]],
    saudi_weight: float = 0.40,
    arcee_weight: float = 0.60,
    seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Combine and shuffle Stage 2 DPO preference pairs according to target weights."""
    if not saudi_dpo_samples and not arcee_dpo_samples:
        raise ValueError("At least one DPO preference sample list must be non-empty.")

    rng = random.Random(seed)
    combined = list(saudi_dpo_samples) + list(arcee_dpo_samples)
    rng.shuffle(combined)

    saudi_count = sum(
        1 for x in combined if x.get("source") == "arabic_dialect_dpo_saudi"
    )
    arcee_count = len(combined) - saudi_count
    stats = {
        "total_pairs": len(combined),
        "saudi_dialect_pairs": saudi_count,
        "arcee_arabic_pairs": arcee_count,
        "target_saudi_weight": saudi_weight,
        "target_arcee_weight": arcee_weight,
    }
    return combined, stats
