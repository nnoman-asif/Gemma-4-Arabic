"""Stage 2 DPO preference dataset normalizers, hygiene filters, and conversational mixer.

Covers the V3 High-Purity Stage 2 DPO curriculum (6,000 length-balanced preference pairs):
1. `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` (`arabic_orpo_dpo_mix_40k` — unified user prompt,
   strict terminal punctuation, balanced length ratio `0.55..1.75`, non-thinking; 2,600 pairs).
2. `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` (`arabic_reasoning_thinking_dpo` — multi-step
   Arabic reasoning split into `<|channel>thought` + final answer, `enable_thinking=True`; 1,600 pairs).
3. `argilla/ultrafeedback-binarized-preferences-cleaned` (`ultrafeedback_english_dpo` — high-margin
   English non-thinking retention anchor; 800 pairs).
4. `argilla/distilabel-math-preference-dpo` (`distilabel_math_thinking_dpo` — step-by-step math
   preference pairs with `<|channel>thought` + final answer, `enable_thinking=True`; 1,000 pairs).
5. `HeshamHaroon/arabic-dialect-dpo` (`name="saudi"` — kept with strict terminal-punctuation
   normalizer for schema compatibility; `target_samples=0` in V3 due to upstream truncation noise).
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
    "$",
)

DPO_MIN_LEN_RATIO: float = 0.55
DPO_MAX_LEN_RATIO: float = 1.75


def _has_complete_terminal_ending(text: str) -> bool:
    """Return True iff `text` ends cleanly with sentence/code/math terminal punctuation."""
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if stripped.endswith((",", "،", ":", "؛", "-", "...", "…", "(", "[", "{", "/")):
        return False
    return any(stripped.endswith(ch) for ch in _VALID_ENDING_CHARS)


def _split_into_thinking_and_answer(
    text: str,
    min_reasoning_chars: int = 80,
    min_answer_chars: int = 20,
) -> Optional[Tuple[str, str]]:
    """Split a multi-step response into `(reasoning_trace, final_answer)` for thinking-mode DPO."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    explicit_r, explicit_a = parse_reasoning_and_content(cleaned)
    if explicit_r and explicit_a:
        if (
            len(explicit_r) >= min_reasoning_chars
            and len(explicit_a) >= min_answer_chars
            and _has_complete_terminal_ending(explicit_a)
        ):
            return explicit_r, explicit_a
        return None

    paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if len(paras) < 2:
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if len(lines) >= 3:
            paras = lines
        else:
            return None

    if len(paras[-1]) < min_answer_chars and len(paras) >= 3:
        reasoning = "\n\n".join(paras[:-2]).strip()
        answer = "\n\n".join(paras[-2:]).strip()
    else:
        reasoning = "\n\n".join(paras[:-1]).strip()
        answer = paras[-1].strip()

    if len(reasoning) < min_reasoning_chars or len(answer) < min_answer_chars:
        return None
    if not _has_complete_terminal_ending(answer):
        return None
    return reasoning, answer


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
        if len(prompt_txt) < 10 or len(chosen_txt) < 30 or len(rejected_txt) < 20:
            return None
        if arabic_char_ratio(prompt_txt) < 0.70 or arabic_char_ratio(chosen_txt) < 0.75:
            return None
        if not _has_complete_terminal_ending(chosen_txt) or not _has_complete_terminal_ending(rejected_txt):
            return None
        len_ratio = len(chosen_txt) / float(max(1, len(rejected_txt)))
        if len_ratio < DPO_MIN_LEN_RATIO or len_ratio > DPO_MAX_LEN_RATIO:
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
# DPO 2: arcee-globe/arabic-orpo-dpo-mix-40k-filtered (Non-Thinking MSA)
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
        if len(unified_prompt) < 12 or len(chosen_resp) < 40 or len(rejected_resp) < 25:
            return None
        if arabic_char_ratio(unified_prompt + " " + chosen_resp) < 0.60:
            return None
        if not _has_complete_terminal_ending(chosen_resp) or not _has_complete_terminal_ending(rejected_resp):
            return None
        # Strict length balance prevents sigmoid DPO verbosity drift / <turn|> EOS suppression
        len_ratio = len(chosen_resp) / float(max(1, len(rejected_resp)))
        if len_ratio < DPO_MIN_LEN_RATIO or len_ratio > DPO_MAX_LEN_RATIO:
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
# DPO 3: Arabic Reasoning Thinking DPO (`enable_thinking=True` CoT Alignment)
# ---------------------------------------------------------------------------
def normalize_arabic_reasoning_thinking_dpo_row(
    row: Dict[str, Any],
    system_prompt: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize multi-step Arabic reasoning pairs into `<|channel>thought` thinking-mode DPO pairs."""
    c_sys, chosen_user_prompt, chosen_resp = _extract_prompt_and_response(row.get("chosen"))
    _, rejected_user_prompt, rejected_resp = _extract_prompt_and_response(row.get("rejected"))

    unified_prompt = str(
        chosen_user_prompt or row.get("prompt") or rejected_user_prompt or ""
    ).strip()
    if len(unified_prompt) < 15 or not chosen_resp or not rejected_resp:
        return None
    if has_english_meta_leak(unified_prompt) or has_english_meta_leak(chosen_resp):
        return None
    if arabic_char_ratio(unified_prompt + " " + chosen_resp) < 0.60:
        return None
    if not _has_complete_terminal_ending(chosen_resp) or not _has_complete_terminal_ending(rejected_resp):
        return None

    len_ratio = len(chosen_resp) / float(max(1, len(rejected_resp)))
    if len_ratio < DPO_MIN_LEN_RATIO or len_ratio > DPO_MAX_LEN_RATIO:
        return None

    c_split = _split_into_thinking_and_answer(chosen_resp, min_reasoning_chars=80, min_answer_chars=25)
    r_split = _split_into_thinking_and_answer(rejected_resp, min_reasoning_chars=60, min_answer_chars=20)
    if c_split is None or r_split is None:
        return None

    c_reason, c_ans = c_split
    r_reason, r_ans = r_split
    if c_ans == r_ans:
        return None

    chosen_sys = system_prompt or c_sys or choose_optional_system_prompt(
        f"ar_think_dpo:{unified_prompt[:120]}",
        pool=ARABIC_SYSTEM_PROMPT_POOL,
        prob=SYSTEM_PROMPT_INJECTION_PROB,
    )
    rec = _build_dpo_record(
        source="arabic_reasoning_thinking_dpo",
        config_name="thinking_cot",
        raw_prompt=unified_prompt,
        chosen_resp=f"<think>\n{c_reason}\n</think>\n{c_ans}",
        rejected_resp=f"<think>\n{r_reason}\n</think>\n{r_ans}",
        system_prompt=chosen_sys,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )
    if rec is None or not rec.get("enable_thinking"):
        return None
    return rec


# ---------------------------------------------------------------------------
# DPO 4: argilla/ultrafeedback-binarized-preferences-cleaned (`split="train"`)
# ---------------------------------------------------------------------------
def normalize_ultrafeedback_dpo_row(
    row: Dict[str, Any],
    min_chosen_rating: float = 4.5,
    min_rating_margin: float = 1.0,
    strict_hygiene: bool = False,
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
    if strict_hygiene:
        if not _has_complete_terminal_ending(chosen_resp) or not _has_complete_terminal_ending(rejected_resp):
            return None
        if len_ratio < DPO_MIN_LEN_RATIO or len_ratio > DPO_MAX_LEN_RATIO:
            return None
    elif len_ratio > 3.5 or len_ratio < 0.30:
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
# DPO 5: argilla/distilabel-math-preference-dpo (`enable_thinking=True` STEM CoT)
# ---------------------------------------------------------------------------
def normalize_distilabel_math_thinking_dpo_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize step-by-step math preference pairs into `<|channel>thought` thinking-mode DPO pairs."""
    c_rating = row.get("chosen_rating") if row.get("chosen_rating") is not None else row.get("chosen-rating")
    r_rating = row.get("rejected_rating") if row.get("rejected_rating") is not None else row.get("rejected-rating")
    if isinstance(c_rating, (int, float)) and isinstance(r_rating, (int, float)):
        c_val = float(c_rating)
        r_val = float(r_rating)
        # Support both 1-10 rating scales (distilabel-math) and 1-5 rating scales (ultrafeedback fallback)
        min_c = 7.0 if max(c_val, r_val) > 5.0 else 4.0
        if c_val < min_c or (c_val - r_val) < 1.0:
            return None

    prompt_txt = str(
        row.get("instruction") or row.get("prompt") or row.get("question") or ""
    ).strip()
    raw_chosen = row.get("chosen_response") if row.get("chosen_response") is not None else row.get("chosen")
    raw_rejected = row.get("rejected_response") if row.get("rejected_response") is not None else row.get("rejected")
    c_sys, c_prompt, chosen_resp = _extract_prompt_and_response(raw_chosen)
    _, _, rejected_resp = _extract_prompt_and_response(raw_rejected)
    if not prompt_txt and c_prompt:
        prompt_txt = c_prompt.strip()

    if len(prompt_txt) < 15 or not chosen_resp or not rejected_resp:
        return None
    if not _has_complete_terminal_ending(chosen_resp) or not _has_complete_terminal_ending(rejected_resp):
        return None

    len_ratio = len(chosen_resp) / float(max(1, len(rejected_resp)))
    if len_ratio < DPO_MIN_LEN_RATIO or len_ratio > DPO_MAX_LEN_RATIO:
        return None

    c_split = _split_into_thinking_and_answer(chosen_resp, min_reasoning_chars=80, min_answer_chars=20)
    r_split = _split_into_thinking_and_answer(rejected_resp, min_reasoning_chars=60, min_answer_chars=15)
    if c_split is None or r_split is None:
        return None

    c_reason, c_ans = c_split
    r_reason, r_ans = r_split
    if c_ans == r_ans:
        return None

    chosen_sys = c_sys or choose_optional_system_prompt(
        f"math_think_dpo:{prompt_txt[:120]}",
        pool=ENGLISH_SYSTEM_PROMPT_POOL,
        prob=SYSTEM_PROMPT_INJECTION_PROB,
    )
    rec = _build_dpo_record(
        source="distilabel_math_thinking_dpo",
        config_name="thinking_cot",
        raw_prompt=prompt_txt,
        chosen_resp=f"<think>\n{c_reason}\n</think>\n{c_ans}",
        rejected_resp=f"<think>\n{r_reason}\n</think>\n{r_ans}",
        system_prompt=chosen_sys,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )
    if rec is None or not rec.get("enable_thinking"):
        return None
    return rec


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
    limit = int(max_samples if max_samples is not None else spec.target_samples)
    if limit <= 0:
        return []
    raw_cfg = spec.configs[0]

    candidate_repos = [spec.repo_id]
    if spec.fallback_repo_id and spec.fallback_repo_id != spec.repo_id:
        candidate_repos.append(spec.fallback_repo_id)

    dedup_set = seen_prompts if seen_prompts is not None else set()
    collected: List[Dict[str, Any]] = []

    for active_repo in candidate_repos:
        if len(collected) >= limit:
            break
        try:
            if active_repo == ARABIC_DIALECT_DPO_REPO:
                validated_cfg = validate_dpo_load_config(active_repo, raw_cfg)
                ds = load_dataset(
                    active_repo,
                    validated_cfg,
                    split=spec.default_split,
                    streaming=streaming,
                )
            elif raw_cfg and active_repo == spec.repo_id:
                ds = load_dataset(active_repo, raw_cfg, split=spec.default_split, streaming=streaming)
            else:
                ds = load_dataset(active_repo, split=spec.default_split, streaming=streaming)
        except Exception as exc:
            logger.warning("Failed opening DPO source '%s' (%s): %s", source_key, active_repo, exc)
            continue

        max_raw_scan = max(limit * 45, 2500)
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > max_raw_scan:
                break
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
            elif source_key == "arabic_reasoning_thinking_dpo":
                item = normalize_arabic_reasoning_thinking_dpo_row(
                    row,
                    tokenizer=tokenizer,
                    decontaminator=decontaminator,
                )
            elif source_key == "ultrafeedback_english_dpo":
                item = normalize_ultrafeedback_dpo_row(
                    row,
                    min_chosen_rating=4.5,
                    min_rating_margin=1.5,
                    strict_hygiene=True,
                    tokenizer=tokenizer,
                    decontaminator=decontaminator,
                )
            elif source_key == "distilabel_math_thinking_dpo":
                item = normalize_distilabel_math_thinking_dpo_row(
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
    arabic_thinking_samples: Optional[Sequence[Dict[str, Any]]] = None,
    english_thinking_samples: Optional[Sequence[Dict[str, Any]]] = None,
    saudi_weight: float = 0.0,
    arcee_weight: float = 0.4333,
    english_weight: float = 0.1333,
    arabic_thinking_weight: float = 0.2667,
    english_thinking_weight: float = 0.1667,
    seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Shuffle and combine non-thinking + `<|channel>thought` thinking DPO preference pairs."""
    saudi_list = list(saudi_dpo_samples)
    arcee_list = list(arcee_dpo_samples)
    eng_list = list(english_dpo_samples or [])
    ar_think_list = list(arabic_thinking_samples or [])
    en_think_list = list(english_thinking_samples or [])

    combined = saudi_list + arcee_list + ar_think_list + eng_list + en_think_list
    if not combined:
        raise ValueError("Combined DPO preference pool cannot be empty.")

    rng = random.Random(seed)
    rng.shuffle(combined)

    total = len(combined)
    arabic_total = len(saudi_list) + len(arcee_list) + len(ar_think_list)
    english_total = len(eng_list) + len(en_think_list)
    thinking_total = sum(1 for item in combined if item.get("enable_thinking"))
    stats = {
        "saudi_dpo_count": len(saudi_list),
        "arcee_dpo_count": len(arcee_list),
        "arabic_thinking_dpo_count": len(ar_think_list),
        "english_dpo_count": len(eng_list),
        "english_thinking_dpo_count": len(en_think_list),
        "arabic_total_count": arabic_total,
        "english_total_count": english_total,
        "thinking_pair_count": thinking_total,
        "total_count": total,
        "configured_saudi_weight": saudi_weight,
        "configured_arcee_weight": arcee_weight,
        "configured_arabic_thinking_weight": arabic_thinking_weight,
        "configured_english_weight": english_weight,
        "configured_english_thinking_weight": english_thinking_weight,
        "actual_saudi_fraction": round(len(saudi_list) / total, 4),
        "actual_arcee_fraction": round(len(arcee_list) / total, 4),
        "actual_arabic_thinking_fraction": round(len(ar_think_list) / total, 4),
        "actual_english_fraction": round(len(eng_list) / total, 4),
        "actual_english_thinking_fraction": round(len(en_think_list) / total, 4),
        "actual_arabic_total_fraction": round(arabic_total / total, 4),
        "actual_thinking_fraction": round(thinking_total / total, 4),
    }
    return combined, stats
