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

_CJK_LEAK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_PERSIAN_ONLY_CHARS_RE = re.compile(r"[گچپژک]")
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
_WORD_TOKEN_RE = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]{3,}")

_SIGNOFF_FILLER_RE = re.compile(
    r"(?:i\s+hope\s+this\s+(?:explanation\s+|answer\s+|solution\s+)?helps|"
    r"hope\s+this\s+helps|let\s+me\s+know\s+if\s+you\s+have|"
    r"feel\s+free\s+to\s+ask|if\s+you\s+have\s+any\s+(?:further\s+|other\s+|more\s+)?questions|"
    r"happy\s+to\s+help|please\s+let\s+me\s+know\s+if|"
    r"آمل\s+أن\s+يكون|أتمنى\s+أن\s+يكون|لا\s+تتردد\s+في\s+طرح|أخبرني\s+إذا\s+كنت|إذا\s+كان\s+لديك\s+أي\s+(?:سؤال|استفسار))",
    flags=re.IGNORECASE,
)

_EXPLICIT_CONCLUSION_START_RE = re.compile(
    r"^(?:\*{0,2})(?:therefore\b|thus\b|hence\b|consequently\b|so,\s|in\s+conclusion,\s+the\b|"
    r"final\s+answer\b|the\s+(?:final\s+|unique\s+|correct\s+)?(?:answer|solution|value|result)\s+is\b|"
    r"إذن\b|إذاً\b|لذلك\b|وبالتالي\b|وعليه\b|بناءً\s+على\s+ذلك\b|النتيجة\b|الإجابة\b|الجواب\b|خلاصة\s+القول\b)",
    flags=re.IGNORECASE | re.UNICODE,
)

_DOLCI_STEM_SOURCE_KEYWORDS: Tuple[str, ...] = (
    "math",
    "flan",
    "code",
    "python",
    "science",
    "qwq",
    "ultrafeedback",
)
_DOLCI_EXCLUDED_SOURCE_KEYWORDS: Tuple[str, ...] = (
    "wildjailbreak",
    "wildguard",
    "harm",
    "safety",
    "not-english",
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


def _prompts_semantically_consistent(
    p1: Optional[str],
    p2: Optional[str],
    min_overlap: float = 0.35,
) -> bool:
    """Verify two user prompt strings refer to the same underlying question."""
    s1 = (p1 or "").strip()
    s2 = (p2 or "").strip()
    if not s1 or not s2 or s1 == s2:
        return True
    w1 = set(_WORD_TOKEN_RE.findall(s1.lower()))
    w2 = set(_WORD_TOKEN_RE.findall(s2.lower()))
    if not w1 or not w2:
        return True
    overlap = len(w1 & w2) / float(max(1, min(len(w1), len(w2))))
    return overlap >= min_overlap


def _split_into_thinking_and_answer(
    text: str,
    min_reasoning_chars: int = 80,
    min_answer_chars: int = 20,
) -> Optional[Tuple[str, str]]:
    """Extract `(reasoning_trace, final_answer)` for thinking-mode DPO without stripping answer content.

    Accepts:
    1. Native `<think>...</think>` / `<|channel>thought...<channel|>` responses where `final_answer`
       is complete, free of raw `<think>` leftovers, and not a bare conversational sign-off filler.
    2. Step-by-step derivations WITHOUT `<think>` tags ONLY when the final paragraph begins with an
       explicit mathematical/logical conclusion marker (`Therefore`, `Thus`, `إذن`, `لذلك`, `وبالتالي`,
       etc.), contains zero conversational sign-off filler (`"I hope this helps!"`), and preserves
       numeric values whenever the derivation steps contain digits.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if _CJK_LEAK_RE.search(cleaned):
        return None

    explicit_r, explicit_a = parse_reasoning_and_content(cleaned)
    if explicit_r and explicit_a:
        if "<think>" in explicit_r or "</think>" in explicit_r or "<think>" in explicit_a or "</think>" in explicit_a:
            return None
        if len(explicit_a) < 160 and _SIGNOFF_FILLER_RE.search(explicit_a):
            return None
        if (
            len(explicit_r) >= min_reasoning_chars
            and len(explicit_a) >= min_answer_chars
            and _has_complete_terminal_ending(explicit_a)
        ):
            return explicit_r, explicit_a
        return None

    # Reject if raw unclosed `<think>` / `</think>` tags exist
    if "<think>" in cleaned or "</think>" in cleaned or "<|channel>" in cleaned:
        return None

    paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if len(paras) < 2:
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if len(lines) >= 3:
            paras = lines
        else:
            return None

    reasoning = "\n\n".join(paras[:-1]).strip()
    answer = paras[-1].strip()

    if len(reasoning) < min_reasoning_chars or len(answer) < min_answer_chars:
        return None
    if not _has_complete_terminal_ending(answer):
        return None
    if _SIGNOFF_FILLER_RE.search(answer):
        return None
    if not _EXPLICIT_CONCLUSION_START_RE.search(answer):
        return None
    if re.search(r"\d", reasoning) and not re.search(r"\d", answer):
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
    """Extract `(system_prompt, user_prompt, assistant_response)` strictly from a single-turn exchange.

    Multi-turn message lists (`len(non_system_turns) != 2`) are rejected (`(None, None, None)`) so a
    Turn-1 user prompt is NEVER paired with an unrelated Turn-2/Turn-3 assistant reply.
    """
    if isinstance(branch, str):
        return None, None, branch.strip()

    if isinstance(branch, list):
        sys_txt: Optional[str] = None
        non_sys: List[Tuple[str, str]] = []
        for msg in branch:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "").strip().lower()
            content = str(
                msg.get("content") if msg.get("content") is not None else (msg.get("value") or "")
            ).strip()
            if role in ("system", "developer"):
                if sys_txt is None:
                    sys_txt = content
            elif role in ("user", "human", "assistant", "gpt", "model"):
                non_sys.append((role, content))

        if len(non_sys) != 2:
            return None, None, None
        u_role, user_txt = non_sys[0]
        a_role, asst_txt = non_sys[1]
        if u_role not in ("user", "human") or a_role not in ("assistant", "gpt", "model"):
            return None, None, None
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
    if _CJK_LEAK_RE.search(unified_prompt + chosen_resp + rejected_resp):
        return None
    if _PERSIAN_ONLY_CHARS_RE.search(unified_prompt + chosen_resp):
        return None
    if "<think>" in chosen_resp or "</think>" in chosen_resp or "<think>" in rejected_resp:
        return None

    if strict_hygiene:
        if len(unified_prompt) < 12 or len(chosen_resp) < 40 or len(rejected_resp) < 25:
            return None
        if not _prompts_semantically_consistent(chosen_user_prompt, rejected_user_prompt, min_overlap=0.35):
            return None
        if arabic_char_ratio(unified_prompt) < 0.50 or arabic_char_ratio(chosen_resp) < 0.55:
            return None
        if not _has_complete_terminal_ending(chosen_resp) or not _has_complete_terminal_ending(rejected_resp):
            return None
        if _SIGNOFF_FILLER_RE.search(chosen_resp):
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
    if not _prompts_semantically_consistent(chosen_user_prompt, rejected_user_prompt, min_overlap=0.35):
        return None
    if has_english_meta_leak(unified_prompt) or has_english_meta_leak(chosen_resp):
        return None
    if _CJK_LEAK_RE.search(unified_prompt + chosen_resp + rejected_resp):
        return None
    if _PERSIAN_ONLY_CHARS_RE.search(unified_prompt + chosen_resp):
        return None
    if arabic_char_ratio(unified_prompt) < 0.50 or arabic_char_ratio(chosen_resp) < 0.55:
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
    _, r_prompt, rejected_resp = _extract_prompt_and_response(row.get("rejected"))
    prompt_txt = str(row.get("prompt") or c_prompt or "").strip()

    if len(prompt_txt) < 15 or not chosen_resp or not rejected_resp:
        return None
    if len(chosen_resp) < 30 or len(rejected_resp) < 15:
        return None
    if _CJK_LEAK_RE.search(prompt_txt + chosen_resp + rejected_resp):
        return None

    len_ratio = len(chosen_resp) / float(max(1, len(rejected_resp)))
    if strict_hygiene:
        if not _prompts_semantically_consistent(c_prompt, r_prompt, min_overlap=0.50):
            return None
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
# DPO 5: STEM/MCQ/Math Thinking CoT DPO (`allenai/Dolci-Think-DPO-7B` + `distilabel-math`)
# ---------------------------------------------------------------------------
def normalize_distilabel_math_thinking_dpo_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize genuine `<think>...</think>` STEM/MCQ/Math preference pairs into `<|channel>thought` DPO pairs."""
    ds_source = str(row.get("dataset_source") or "").strip().lower()
    if ds_source:
        if any(bad in ds_source for bad in _DOLCI_EXCLUDED_SOURCE_KEYWORDS):
            return None
        if not any(kw in ds_source for kw in _DOLCI_STEM_SOURCE_KEYWORDS):
            return None

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
    _, r_prompt, rejected_resp = _extract_prompt_and_response(raw_rejected)
    if not prompt_txt and c_prompt:
        prompt_txt = c_prompt.strip()

    if len(prompt_txt) < 15 or not chosen_resp or not rejected_resp:
        return None
    if not _prompts_semantically_consistent(c_prompt, r_prompt, min_overlap=0.50):
        return None
    if _CJK_LEAK_RE.search(prompt_txt + chosen_resp + rejected_resp):
        return None
    if _ARABIC_SCRIPT_RE.search(prompt_txt):
        return None
    if len(chosen_resp) > 6800 or len(rejected_resp) > 6800:
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


def _iter_fast_parquet_shard_if_supported(repo_id: str) -> Optional[Any]:
    """Read single-shard Parquet DPO sources directly via `hf_hub_download` + `pyarrow.parquet`."""
    shard_map = {
        "allenai/Dolci-Think-DPO-7B": "data/train-00000-of-00007.parquet",
        "arcee-globe/arabic-orpo-dpo-mix-40k-filtered": "data/train-00000-of-00001.parquet",
        "arcee-globe/filtered-arabic-distilabel-math-preference-orpo": "data/train-00000-of-00001.parquet",
        "argilla/ultrafeedback-binarized-preferences-cleaned": "data/train-00000-of-00001.parquet",
    }
    rel_file = shard_map.get(repo_id)
    if not rel_file:
        return None
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        local_pq = hf_hub_download(repo_id=repo_id, filename=rel_file, repo_type="dataset")
        pf = pq.ParquetFile(local_pq)

        def _row_gen():
            for batch in pf.iter_batches(batch_size=512):
                for rec in batch.to_pylist():
                    yield rec

        return _row_gen()
    except Exception:
        return None


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
        ds: Any = _iter_fast_parquet_shard_if_supported(active_repo) if streaming else None
        if ds is None:
            try:
                if active_repo == ARABIC_DIALECT_DPO_REPO:
                    validated_cfg = validate_dpo_load_config(active_repo, raw_cfg)
                    ds = load_dataset(
                        active_repo,
                        validated_cfg,
                        split=spec.default_split,
                        streaming=streaming,
                    )
                elif active_repo == "allenai/Dolci-Think-DPO-7B":
                    ds = load_dataset(
                        active_repo,
                        data_files="data/train-00000-of-00007.parquet",
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
