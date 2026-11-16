"""Stage 2 DPO preference dataset normalizers, hygiene filters, and conversational mixer.

Covers all 3 curated Stage 2 DPO sources (20,000 preference pairs total):
1. `HeshamHaroon/arabic-dialect-dpo` (`config="saudi"` — strictly filtered against mid-sentence
   truncation and English prompt leaks; target 4,000 pairs / 20%).
2. `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` (unified user prompt between `chosen` and
   `rejected`, filtered against degenerate length ratios; target 13,000 pairs / 65%).
3. `argilla/ultrafeedback-binarized-preferences-cleaned` (`split="train"`, `chosen-rating >= 4.5`
   and margin `>= 1.0` — English/Math/Code retention anchor; target 3,000 pairs / 15%).
"""

import hashlib
import logging
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.chat_template import (
    format_gemma4_chat,
    make_uniform_message,
    parse_reasoning_and_content,
)
from src.constants import (
    ARABIC_DIALECT_DPO_REPO,
    ARABIC_DIALECT_REQUIRED_CONFIG,
    ARABIC_SYSTEM_PROMPT_POOL,
    DPO_DATASETS,
    DPO_MAX_PROMPT_TOKEN_FILTER,
    DPO_MAX_TOKEN_LENGTH_FILTER,
    ENGLISH_SYSTEM_PROMPT_POOL,
    SYSTEM_PROMPT_INJECTION_PROB,
    load_dotenv_if_present,
)
from src.data_normalizer import (
    BenchmarkDecontaminator,
    arabic_char_ratio,
    choose_optional_system_prompt,
    estimate_gemma4_token_count,
    has_english_meta_leak,
)

logger = logging.getLogger(__name__)

_SAUDI_LEAK_RE = re.compile(
    r"(?:in\s+saudi\s+dialect|write\s+a\s+response|translate\s+the\s+following|here\s+is\s+the|sure,\s+here)",
    flags=re.IGNORECASE,
)

_VALID_ENDING_CHARS: Tuple[str, ...] = (
    ".",
    "!",
    "?",
    "؟",
    "»",
    '"',
    "'",
    ")",
    "]",
    "}",
    "`",
    "ه",
    "ة",
    "ي",
    "ى",
    "ا",
    "م",
    "ن",
    "ر",
    "ل",
    "ك",
    "د",
    "ع",
    "ب",
    "ت",
    "س",
    "ف",
    "ق",
    "و",
)


def validate_dpo_load_config(repo_id: str, config_name: Optional[str]) -> str:
    """Enforce `name='saudi'` when loading `HeshamHaroon/arabic-dialect-dpo`."""
    if repo_id == ARABIC_DIALECT_DPO_REPO:
        if config_name != ARABIC_DIALECT_REQUIRED_CONFIG:
            raise ValueError(
                f"Dataset '{ARABIC_DIALECT_DPO_REPO}' MUST be loaded with "
                f"config name='{ARABIC_DIALECT_REQUIRED_CONFIG}' (received: {config_name!r})."
            )
        return config_name
    return config_name or "default"


def _extract_prompt_and_response(
    branch: Any,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract `(system_prompt, user_prompt, assistant_response)` from a string or message list."""
    if isinstance(branch, str):
        return None, None, branch.strip()

    if isinstance(branch, list):
        sys_txt: Optional[str] = None
        user_txt: Optional[str] = None
        asst_txt: Optional[str] = None
        for msg in branch:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "").strip().lower()
            content = str(
                msg.get("content") if msg.get("content") is not None else (msg.get("value") or "")
            ).strip()
            if role == "system" and sys_txt is None:
                sys_txt = content
            elif role in ("user", "human") and user_txt is None:
                user_txt = content
            elif role in ("assistant", "gpt", "model"):
                asst_txt = content
        return sys_txt, user_txt, asst_txt

    return None, None, None


def _build_dpo_record(
    source: str,
    config_name: str,
    raw_prompt: str,
    chosen_resp: str,
    rejected_resp: str,
    system_prompt: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Construct a validated, Arrow-uniform conversational & rendered DPO preference record."""
    clean_prompt = (raw_prompt or "").strip()
    clean_chosen = (chosen_resp or "").strip()
    clean_rejected = (rejected_resp or "").strip()

    if not clean_prompt or not clean_chosen or not clean_rejected:
        return None
    if clean_chosen == clean_rejected:
        return None

    c_reason, c_ans = parse_reasoning_and_content(clean_chosen)
    r_reason, r_ans = parse_reasoning_and_content(clean_rejected)
    enable_thinking = bool(c_reason and r_reason)
    if not enable_thinking:
        c_ans = c_ans if c_reason else clean_chosen
        r_ans = r_ans if r_reason else clean_rejected
        c_reason = ""
        r_reason = ""

    if not c_ans or not r_ans or c_ans == r_ans:
        return None

    prompt_messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        prompt_messages.append(make_uniform_message("system", system_prompt.strip()))
    prompt_messages.append(make_uniform_message("user", clean_prompt))

    if decontaminator is not None and decontaminator.is_messages_contaminated(prompt_messages):
        return None

    chosen_only = [make_uniform_message("assistant", c_ans, reasoning_content=c_reason)]
    rejected_only = [make_uniform_message("assistant", r_ans, reasoning_content=r_reason)]

    # Full unified conversations (chosen_messages[0] == rejected_messages[0] invariant!)
    chosen_full = [
        make_uniform_message("user", clean_prompt),
        chosen_only[0],
    ]
    rejected_full = [
        make_uniform_message("user", clean_prompt),
        rejected_only[0],
    ]

    prompt_rendered = format_gemma4_chat(
        prompt_messages,
        enable_thinking=enable_thinking,
        add_generation_prompt=True,
        add_bos=True,
    )
    chosen_rendered = format_gemma4_chat(
        prompt_messages + chosen_only,
        enable_thinking=enable_thinking,
        add_generation_prompt=False,
        add_bos=True,
    )
    rejected_rendered = format_gemma4_chat(
        prompt_messages + rejected_only,
        enable_thinking=enable_thinking,
        add_generation_prompt=False,
        add_bos=True,
    )

    # Enforce DPO sequence length ceilings so zero completions truncate during Stage 2 DPO
    if max(len(chosen_rendered), len(rejected_rendered)) > DPO_MAX_TOKEN_LENGTH_FILTER * 6:
        return None
    prompt_toks = estimate_gemma4_token_count(prompt_rendered, tokenizer=tokenizer)
    chosen_toks = estimate_gemma4_token_count(chosen_rendered, tokenizer=tokenizer)
    rejected_toks = estimate_gemma4_token_count(rejected_rendered, tokenizer=tokenizer)

    if prompt_toks > DPO_MAX_PROMPT_TOKEN_FILTER:
        return None
    if max(chosen_toks, rejected_toks) > DPO_MAX_TOKEN_LENGTH_FILTER:
        return None

    return {
        "source": source,
        "config_name": config_name,
        "prompt": prompt_rendered,
        "chosen": chosen_rendered,
        "rejected": rejected_rendered,
        "raw_prompt": clean_prompt,
        "prompt_messages": prompt_messages,
        "chosen_only_messages": chosen_only,
        "rejected_only_messages": rejected_only,
        "chosen_messages": chosen_full,
        "rejected_messages": rejected_full,
        "enable_thinking": enable_thinking,
    }


# ---------------------------------------------------------------------------
# DPO 1: HeshamHaroon/arabic-dialect-dpo (`name="saudi"`, strictly filtered)
# ---------------------------------------------------------------------------
def normalize_arabic_dialect_dpo_row(
    row: Dict[str, Any],
    system_prompt: Optional[str] = None,
    strict_hygiene: bool = False,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `HeshamHaroon/arabic-dialect-dpo` (`saudi` config)."""
    _, p_from_c, chosen_txt = _extract_prompt_and_response(row.get("chosen"))
    _, _, rejected_txt = _extract_prompt_and_response(row.get("rejected"))
    prompt_txt = str(
        row.get("prompt") or row.get("question") or row.get("instruction") or p_from_c or ""
    ).strip()

    if not prompt_txt or not chosen_txt or not rejected_txt:
        return None

    if has_english_meta_leak(prompt_txt) or has_english_meta_leak(chosen_txt) or has_english_meta_leak(rejected_txt):
        return None
    if _SAUDI_LEAK_RE.search(prompt_txt) or _SAUDI_LEAK_RE.search(chosen_txt) or _SAUDI_LEAK_RE.search(rejected_txt):
        return None

    if strict_hygiene:
        if len(prompt_txt) < 10 or len(chosen_txt) < 30 or len(rejected_txt) < 10:
            return None
        if arabic_char_ratio(prompt_txt) < 0.70 or arabic_char_ratio(chosen_txt) < 0.75:
            return None
        # Reject mid-word / mid-comma cutoffs common in synthetic dialect generations
        stripped_end = chosen_txt.rstrip()
        if stripped_end.endswith((",", "،", ":", "؛", "-", "...")):
            return None
        if not any(stripped_end.endswith(ch) for ch in _VALID_ENDING_CHARS):
            return None

    chosen_sys = system_prompt if system_prompt is not None else choose_optional_system_prompt(
        f"saudi_dpo:{prompt_txt[:120]}",
        pool=ARABIC_SYSTEM_PROMPT_POOL,
        prob=SYSTEM_PROMPT_INJECTION_PROB,
    )
    return _build_dpo_record(
        source="arabic_dialect_dpo_saudi",
        config_name=ARABIC_DIALECT_REQUIRED_CONFIG,
        raw_prompt=prompt_txt,
        chosen_resp=chosen_txt,
        rejected_resp=rejected_txt,
        system_prompt=chosen_sys,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# DPO 2: arcee-globe/arabic-orpo-dpo-mix-40k-filtered
# ---------------------------------------------------------------------------
def normalize_arcee_arabic_dpo_row(
    row: Dict[str, Any],
    system_prompt: Optional[str] = None,
    strict_hygiene: bool = False,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` with unified user prompt."""
    c_sys, chosen_user_prompt, chosen_resp = _extract_prompt_and_response(row.get("chosen"))
    _, rejected_user_prompt, rejected_resp = _extract_prompt_and_response(row.get("rejected"))

    unified_prompt = str(
        chosen_user_prompt or row.get("prompt") or rejected_user_prompt or ""
    ).strip()
    if not unified_prompt or not chosen_resp or not rejected_resp:
        return None

    if has_english_meta_leak(unified_prompt) or has_english_meta_leak(chosen_resp):
        return None

    if strict_hygiene:
        if len(unified_prompt) < 12 or len(chosen_resp) < 30 or len(rejected_resp) < 15:
            return None
        if arabic_char_ratio(unified_prompt + " " + chosen_resp) < 0.60:
            return None
        # Guard against extreme length-bias hacking where chosen is 5x longer than rejected
        len_ratio = len(chosen_resp) / float(max(1, len(rejected_resp)))
        if len_ratio > 4.0 or len_ratio < 0.25:
            return None

    chosen_sys = system_prompt or c_sys or choose_optional_system_prompt(
        f"arcee_dpo:{unified_prompt[:120]}",
        pool=ARABIC_SYSTEM_PROMPT_POOL,
        prob=SYSTEM_PROMPT_INJECTION_PROB,
    )
    return _build_dpo_record(
        source="arabic_orpo_dpo_mix_40k",
        config_name="default",
        raw_prompt=unified_prompt,
        chosen_resp=chosen_resp,
        rejected_resp=rejected_resp,
        system_prompt=chosen_sys,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# DPO 3: argilla/ultrafeedback-binarized-preferences-cleaned (`split="train"`)
# ---------------------------------------------------------------------------
def normalize_ultrafeedback_dpo_row(
    row: Dict[str, Any],
    min_chosen_rating: float = 4.5,
    min_rating_margin: float = 1.0,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize high-margin English retention pairs from `argilla/ultrafeedback-binarized-preferences-cleaned`."""
    c_rating = row.get("chosen-rating")
    r_rating = row.get("rejected-rating")
    if isinstance(c_rating, (int, float)) and float(c_rating) < min_chosen_rating:
        return None
    if isinstance(c_rating, (int, float)) and isinstance(r_rating, (int, float)):
        if (float(c_rating) - float(r_rating)) < min_rating_margin:
            return None

    c_sys, c_prompt, chosen_resp = _extract_prompt_and_response(row.get("chosen"))
    _, _, rejected_resp = _extract_prompt_and_response(row.get("rejected"))
    prompt_txt = str(row.get("prompt") or c_prompt or "").strip()

    if len(prompt_txt) < 15 or not chosen_resp or not rejected_resp:
        return None
    if len(chosen_resp) < 30 or len(rejected_resp) < 15:
        return None
    len_ratio = len(chosen_resp) / float(max(1, len(rejected_resp)))
    if len_ratio > 3.5 or len_ratio < 0.30:
        return None

    chosen_sys = c_sys or choose_optional_system_prompt(
        f"uf_dpo:{prompt_txt[:120]}",
        pool=ENGLISH_SYSTEM_PROMPT_POOL,
        prob=SYSTEM_PROMPT_INJECTION_PROB,
    )
    return _build_dpo_record(
        source="ultrafeedback_english_dpo",
        config_name="default",
        raw_prompt=prompt_txt,
        chosen_resp=chosen_resp,
        rejected_resp=rejected_resp,
        system_prompt=chosen_sys,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Streaming Loader & Weighted DPO Mixer
# ---------------------------------------------------------------------------
def load_and_normalize_dpo_source(
    source_key: str,
    max_samples: Optional[int] = None,
    streaming: bool = True,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
    seen_prompts: Optional[Set[bytes]] = None,
) -> List[Dict[str, Any]]:
    """Load and normalize preference pairs for any configured Stage 2 DPO source."""
    load_dotenv_if_present()
    from datasets import load_dataset

    if source_key not in DPO_DATASETS:
        raise KeyError(f"Unknown DPO dataset key: {source_key}")

    spec = DPO_DATASETS[source_key]
    limit = max_samples if max_samples is not None else spec.target_samples
    raw_cfg = spec.configs[0]

    if spec.repo_id == ARABIC_DIALECT_DPO_REPO:
        validated_cfg = validate_dpo_load_config(spec.repo_id, raw_cfg)
        ds = load_dataset(
            spec.repo_id,
            validated_cfg,
            split=spec.default_split,
            streaming=streaming,
        )
    elif raw_cfg:
        ds = load_dataset(spec.repo_id, raw_cfg, split=spec.default_split, streaming=streaming)
    else:
        ds = load_dataset(spec.repo_id, split=spec.default_split, streaming=streaming)

    dedup_set = seen_prompts if seen_prompts is not None else set()
    collected: List[Dict[str, Any]] = []

    for row in ds:
        if source_key == "arabic_dialect_dpo_saudi":
            item = normalize_arabic_dialect_dpo_row(
                row,
                strict_hygiene=True,
                tokenizer=tokenizer,
                decontaminator=decontaminator,
            )
        elif source_key == "arabic_orpo_dpo_mix_40k":
            item = normalize_arcee_arabic_dpo_row(
                row,
                strict_hygiene=True,
                tokenizer=tokenizer,
                decontaminator=decontaminator,
            )
        elif source_key == "ultrafeedback_english_dpo":
            item = normalize_ultrafeedback_dpo_row(
                row,
                tokenizer=tokenizer,
                decontaminator=decontaminator,
            )
        else:
            item = None

        if item is None:
            continue

        sig = hashlib.blake2b(
            item["raw_prompt"].strip().lower().encode("utf-8", errors="ignore"),
            digest_size=12,
        ).digest()
        if sig in dedup_set:
            continue
        dedup_set.add(sig)

        collected.append(item)
        if len(collected) >= limit:
            break

    logger.info(
        "Collected %d / %d normalized DPO preference pairs from '%s'.",
        len(collected),
        limit,
        source_key,
    )
    return collected


def mix_dpo_preference_datasets(
    saudi_dpo_samples: Sequence[Dict[str, Any]],
    arcee_dpo_samples: Sequence[Dict[str, Any]],
    english_dpo_samples: Optional[Sequence[Dict[str, Any]]] = None,
    saudi_weight: float = 0.20,
    arcee_weight: float = 0.65,
    english_weight: float = 0.15,
    seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Shuffle and combine Saudi, MSA, and English retention DPO preference pairs."""
    eng_list = list(english_dpo_samples or [])
    combined = list(saudi_dpo_samples) + list(arcee_dpo_samples) + eng_list
    if not combined:
        raise ValueError("Combined DPO preference pool cannot be empty.")

    rng = random.Random(seed)
    rng.shuffle(combined)

    total = len(combined)
    stats = {
        "saudi_dpo_count": len(saudi_dpo_samples),
        "arcee_dpo_count": len(arcee_dpo_samples),
        "english_dpo_count": len(eng_list),
        "total_count": total,
        "configured_saudi_weight": saudi_weight,
        "configured_arcee_weight": arcee_weight,
        "configured_english_weight": english_weight,
        "actual_saudi_fraction": round(len(saudi_dpo_samples) / total, 4),
        "actual_arcee_fraction": round(len(arcee_dpo_samples) / total, 4),
        "actual_english_fraction": round(len(eng_list) / total, 4),
    }
    return combined, stats
