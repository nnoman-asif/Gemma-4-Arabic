"""Schema normalizers, quality filters, and 13-gram benchmark decontamination for Stage 1 Arabic SFT.

Covers all 9 curated public (`gated=False`) Arabic & Saudi SFT datasets (90,000 target rows):
1. `AdaMLLab/smolkalam-arabic-conversational-sft` (`Mixture_of_Thoughts_science_no_think`)
2. `AdaMLLab/smolkalam-arabic-conversational-sft` (`OpenThoughts3_50K` — verified `<think>...</think>`)
3. `AdaMLLab/smolkalam-arabic-conversational-sft` (`OpenHermes_2.5_no_think`)
4. `oddadmix/arabic-math-reasoning-synth` (`instruction` + `reasoning` + `answer`)
5. `FreedomIntelligence/alpaca-gpt4-arabic` (`conversations`: `human` / `gpt`)
6. `Gheras/Medical_Reasoning_Arabic_Dataset` (`Question` + `Complex_CoT` + `Response`, stripped of English notes)
7. `CohereLabs/aya_dataset` (human-annotated Arabic `arb`, `ars`, `arz`, `ajp`, `acq`, `ary`)
8. `VohoAI/voho-saudi-dialogues` (Najdi, Hijazi, Khaliji multi-turn dialogues)
9. `SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus` (`moj` & `boe`, active `ساري` statutes only)
"""

import hashlib
import json
import logging
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.chat_template import (
    format_gemma4_chat,
    make_uniform_message,
    parse_reasoning_and_content,
    verify_thinking_block_preserved,
)
from src.constants import (
    ARABIC_SYSTEM_PROMPT_POOL,
    DEFAULT_ARABIC_SYSTEM_PROMPT,
    SFT_ARABIC_DATASETS,
    SFT_MAX_TOKEN_LENGTH_FILTER,
    SYSTEM_PROMPT_INJECTION_PROB,
    load_dotenv_if_present,
)

logger = logging.getLogger(__name__)

_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:User|Assistant|Human|AI|Bot|System|المستخدم|المساعد|الإنسان|النظام)\s*[:：\-]\s*",
    flags=re.IGNORECASE,
)

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_WORD_NORM_RE = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)

_ENGLISH_META_LEAK_SUBSTRINGS: Tuple[str, ...] = (
    "this translation aims",
    "original text",
    "note on translation",
    "translated from english",
    "here is the translation",
    "as an ai language model",
    "in saudi dialect:",
    "generate a response in",
)

_SAUDI_LAW_QUESTION_TEMPLATES: Tuple[str, ...] = (
    "وضح النص النظامي والمضمون القانوني لـ {art} من {doc}.",
    "ماذا تنص عليه {art} في {doc}؟",
    "اذكر الحكم النظامي الوارد في {art} من {doc}.",
    "ما هي الأحكام المنصوص عليها في {art} ضمن {doc}؟",
    "اشرح مضمون {art} من {doc} وفق الأنظمة السعودية السارية.",
)


# ---------------------------------------------------------------------------
# Benchmark Decontamination Engine (Exact Normalized Text + 13-Gram Overlap)
# ---------------------------------------------------------------------------
class BenchmarkDecontaminator:
    """Fast O(L) exact + 13-word shingle decontamination filter against eval test sets."""

    def __init__(self, ngram_size: int = 13) -> None:
        self.ngram_size = ngram_size
        self.exact_hashes: Set[bytes] = set()
        self.ngram_hashes: Set[bytes] = set()
        self.loaded_benchmarks: Dict[str, int] = {}

    @staticmethod
    def _normalize_words(text: str) -> List[str]:
        cleaned = _WORD_NORM_RE.sub(" ", str(text or "").lower()).strip()
        return [w for w in cleaned.split() if w]

    @staticmethod
    def _digest(text: str) -> bytes:
        return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=12).digest()

    def add_benchmark_text(self, text: str) -> bool:
        """Index a single benchmark test question/prompt into exact and 13-gram sets."""
        words = self._normalize_words(text)
        if len(words) < 5:
            return False
        norm_joined = " ".join(words)
        self.exact_hashes.add(self._digest(norm_joined))
        n = self.ngram_size
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                shingle = " ".join(words[i : i + n])
                self.ngram_hashes.add(self._digest(shingle))
        return True

    def is_contaminated(self, text: str) -> bool:
        """Return True if `text` matches a benchmark test item exactly or on any 13-word shingle."""
        if not self.exact_hashes and not self.ngram_hashes:
            return False
        words = self._normalize_words(text)
        if len(words) < 5:
            return False
        norm_joined = " ".join(words)
        if self._digest(norm_joined) in self.exact_hashes:
            return True
        n = self.ngram_size
        if len(words) >= n and self.ngram_hashes:
            for i in range(len(words) - n + 1):
                if self._digest(" ".join(words[i : i + n])) in self.ngram_hashes:
                    return True
        return False

    def is_messages_contaminated(self, messages: Sequence[Dict[str, Any]]) -> bool:
        """Check all user turns in `messages` against the indexed benchmark test sets."""
        for msg in messages:
            if str(msg.get("role", "")).lower() == "user":
                if self.is_contaminated(str(msg.get("content") or "")):
                    return True
        return False

    @classmethod
    def from_eval_benchmarks(cls, load_live: bool = True) -> "BenchmarkDecontaminator":
        """Load public benchmark test prompts to guarantee 0.0% train-test overlap."""
        decontam = cls(ngram_size=13)
        if not load_live:
            return decontam

        load_dotenv_if_present()
        from datasets import get_dataset_config_names, load_dataset

        # (label, repo_id, config_name_or_ALL, split_candidates, text_fields)
        benchmark_specs: List[Tuple[str, str, Optional[str], Tuple[str, ...], Tuple[str, ...]]] = [
            ("MMLU-Pro", "TIGER-Lab/MMLU-Pro", None, ("test",), ("question",)),
            ("ArabicMMLU", "MBZUAI/ArabicMMLU", "__ALL__", ("test", "dev"), ("Question", "question")),
            ("MadinahQA", "MBZUAI/MadinahQA", "__ALL__", ("test", "dev"), ("Question", "question")),
            ("AraTrust", "asas-ai/AraTrust", None, ("test", "train"), ("Question", "question")),
            ("EXAMS-AR", "OALL/Arabic_EXAMS", None, ("test", "validation"), ("question", "Question")),
            (
                "AlGhafa",
                "OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
                "__ALL__",
                ("test",),
                ("query", "question"),
            ),
            (
                "LiveCodeBench",
                "livecodebench/code_generation_lite",
                None,
                ("test",),
                ("question_content",),
            ),
            ("AIME-2025", "MathArena/aime_2025", None, ("train", "test"), ("problem",)),
            ("AIME-2026", "MathArena/aime_2026", None, ("train", "test"), ("problem",)),
        ]

        for label, repo_id, cfg_spec, splits, fields in benchmark_specs:
            added = 0
            try:
                configs_to_load: List[Optional[str]]
                if cfg_spec == "__ALL__":
                    try:
                        configs_to_load = list(get_dataset_config_names(repo_id))
                    except Exception:
                        configs_to_load = [None]
                else:
                    configs_to_load = [cfg_spec]

                for cfg_name in configs_to_load:
                    for sp in splits:
                        try:
                            ds = (
                                load_dataset(repo_id, cfg_name, split=sp)
                                if cfg_name
                                else load_dataset(repo_id, split=sp)
                            )
                            for row in ds:
                                for f in fields:
                                    val = row.get(f)
                                    if isinstance(val, str) and len(val.strip()) >= 20:
                                        if decontam.add_benchmark_text(val):
                                            added += 1
                                        break
                            break
                        except Exception:
                            continue
            except Exception as exc:
                logger.debug("Optional decontamination source %s skipped: %s", repo_id, exc)
            if added > 0:
                decontam.loaded_benchmarks[label] = added
                logger.info(
                    "[Decontam] Indexed %d test prompts from %s (%s)",
                    added,
                    label,
                    repo_id,
                )

        logger.info(
            "[Decontam] Ready: %d exact hashes, %d 13-gram hashes across %d suites.",
            len(decontam.exact_hashes),
            len(decontam.ngram_hashes),
            len(decontam.loaded_benchmarks),
        )
        return decontam


# ---------------------------------------------------------------------------
# Token Length & Text Hygiene Helpers
# ---------------------------------------------------------------------------
def estimate_gemma4_token_count(text: str, tokenizer: Optional[Any] = None) -> int:
    """Return exact Gemma 4 token count when `tokenizer` is provided, or conservative upper-bound estimate."""
    if not text:
        return 0
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Conservative fallback: ~3.1 chars/token on mixed Arabic/English/LaTeX
    return max(1, int(len(text) / 3.0) + 16)


def strip_speaker_prefix(text: str) -> str:
    """Remove redundant leading speaker tags (`User:`, `Assistant:`, `المساعد:`) from text."""
    cleaned = (text or "").strip()
    while True:
        updated = _SPEAKER_PREFIX_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def drop_pil_image_columns(
    dataset: Any,
    columns_to_drop: Sequence[str] = ("image", "images"),
) -> Any:
    """Drop PIL image columns before text normalization when present."""
    col_names = getattr(dataset, "column_names", None)
    if col_names is None:
        return dataset
    present = [c for c in columns_to_drop if c in col_names]
    if present:
        return dataset.remove_columns(present)
    return dataset


def arabic_char_ratio(text: str) -> float:
    """Compute fraction of alphabetic characters that are Arabic script."""
    ar_count = len(_ARABIC_CHAR_RE.findall(text or ""))
    lat_count = len(_LATIN_CHAR_RE.findall(text or ""))
    total = ar_count + lat_count
    if total == 0:
        return 0.0
    return ar_count / float(total)


def has_english_meta_leak(text: str) -> bool:
    """Detect leaked English translator/prompt commentary inside Arabic dataset fields."""
    low = (text or "").lower()
    return any(marker in low for marker in _ENGLISH_META_LEAK_SUBSTRINGS)


def choose_optional_system_prompt(
    deterministic_key: str,
    pool: Sequence[str] = ARABIC_SYSTEM_PROMPT_POOL,
    prob: float = SYSTEM_PROMPT_INJECTION_PROB,
) -> Optional[str]:
    """Deterministically include a varied concise system prompt on `prob` (~30%) of examples."""
    if prob <= 0.0 or not pool:
        return None
    digest = hashlib.blake2b(deterministic_key.encode("utf-8", errors="ignore"), digest_size=8).digest()
    val = int.from_bytes(digest[:4], "little") / float(2**32)
    if val >= prob:
        return None
    idx = int.from_bytes(digest[4:], "little") % len(pool)
    return pool[idx]


def finalize_sft_sample(
    source: str,
    turns: Sequence[Dict[str, Any]],
    enable_thinking: bool,
    system_prompt: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    max_tokens: int = SFT_MAX_TOKEN_LENGTH_FILTER,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Validate, attach optional system prompt, check decontamination & length, and render Gemma 4 text."""
    messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append(make_uniform_message("system", system_prompt.strip()))

    has_user = False
    has_assistant = False
    for t in turns:
        u_msg = make_uniform_message(
            role=str(t.get("role", "user")),
            content=str(t.get("content", "")),
            reasoning_content=str(t.get("reasoning_content", "")),
        )
        if u_msg["role"] == "user":
            if not u_msg["content"]:
                return None
            has_user = True
        elif u_msg["role"] == "assistant":
            if not u_msg["content"]:
                return None
            has_assistant = True
        messages.append(u_msg)

    if not (has_user and has_assistant):
        return None
    if messages[-1]["role"] != "assistant":
        return None

    # Enforce strict thinking consistency: enable_thinking=True requires non-empty reasoning_content
    final_reasoning = messages[-1].get("reasoning_content", "").strip()
    if enable_thinking and not final_reasoning:
        return None
    if not enable_thinking and final_reasoning:
        # Strip reasoning if source was explicitly configured as no_think
        for m in messages:
            m["reasoning_content"] = ""

    # Benchmark decontamination gate
    if decontaminator is not None and decontaminator.is_messages_contaminated(messages):
        return None

    rendered_text = format_gemma4_chat(
        messages,
        tokenizer=None,
        enable_thinking=enable_thinking,
        add_generation_prompt=False,
        add_bos=True,
    )
    if not rendered_text or not verify_thinking_block_preserved(
        rendered_text, enable_thinking=bool(enable_thinking)
    ):
        return None

    # Fast character pre-check before exact tokenizer check
    if len(rendered_text) > max_tokens * 6:
        return None
    tok_len = estimate_gemma4_token_count(rendered_text, tokenizer=tokenizer)
    if tok_len > max_tokens:
        return None

    return {
        "source": source,
        "messages": messages,
        "enable_thinking": bool(enable_thinking),
        "text": rendered_text,
    }


# ---------------------------------------------------------------------------
# Normalizer 1–3: AdaMLLab/smolkalam-arabic-conversational-sft
# ---------------------------------------------------------------------------
def normalize_smolkalam_row(
    row: Dict[str, Any],
    source_key: str = "smolkalam_arabic_science",
    require_thinking: bool = False,
    min_lr: float = 0.85,
    min_scr: float = 0.90,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `AdaMLLab/smolkalam-arabic-conversational-sft`.

    Handles both `*_no_think` configs (`require_thinking=False`) and `OpenThoughts3_50K`
    (`require_thinking=True`), strictly rejecting malformed `OpenThoughts3_50K` rows that
    lack closed `<think>...</think>` tags or run past `max_tokens`.
    """
    lr_val = row.get("LR")
    scr_val = row.get("SCR")
    if isinstance(lr_val, (int, float)) and float(lr_val) < min_lr:
        return None
    if isinstance(scr_val, (int, float)) and float(scr_val) < min_scr:
        return None

    raw_msgs = row.get("messages")
    if not isinstance(raw_msgs, list) or len(raw_msgs) < 2:
        return None

    turns: List[Dict[str, str]] = []
    explicit_sys: Optional[str] = None

    for idx, m in enumerate(raw_msgs):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue

        if role == "system" and idx == 0:
            explicit_sys = content
            continue
        if role == "user":
            if has_english_meta_leak(content):
                return None
            turns.append({"role": "user", "content": content, "reasoning_content": ""})
        elif role in ("assistant", "model"):
            if require_thinking:
                if not (content.startswith("<think>") and "</think>" in content):
                    return None
                reasoning, answer = parse_reasoning_and_content(content)
                if len(reasoning) < 40 or len(answer) < 15:
                    return None
                if "<think>" in answer or "</think>" in answer:
                    return None
                if arabic_char_ratio(reasoning + " " + answer) < 0.45 or arabic_char_ratio(answer) < 0.25:
                    return None
                turns.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "reasoning_content": reasoning,
                    }
                )
            else:
                if "<think>" in content or "</think>" in content:
                    reasoning, answer = parse_reasoning_and_content(content)
                    content = answer if answer else ""
                if len(content) < 15 or arabic_char_ratio(content) < 0.50:
                    return None
                turns.append({"role": "assistant", "content": content, "reasoning_content": ""})

    if len(turns) < 2 or turns[-1]["role"] != "assistant":
        return None

    sys_prompt = explicit_sys or choose_optional_system_prompt(
        f"{source_key}:{turns[0]['content'][:120]}"
    )
    return finalize_sft_sample(
        source=source_key,
        turns=turns,
        enable_thinking=require_thinking,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Normalizer 4: oddadmix/arabic-math-reasoning-synth
# ---------------------------------------------------------------------------
def normalize_arabic_math_synth_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `oddadmix/arabic-math-reasoning-synth` (`instruction` + `reasoning` + `answer`)."""
    problem = str(row.get("instruction") or row.get("problem") or row.get("question") or "").strip()
    reasoning = str(row.get("reasoning") or "").strip()
    answer = str(row.get("answer") or row.get("solution") or "").strip()

    # Backwards compatibility for 2-column `{problem, solution}` inputs in unit tests
    if not reasoning and answer:
        parsed_r, parsed_a = parse_reasoning_and_content(answer)
        if parsed_r and parsed_a:
            reasoning, answer = parsed_r, parsed_a
        else:
            lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
            if len(lines) >= 2:
                reasoning = "\n".join(lines[:-1])
                answer = lines[-1]
            else:
                reasoning = answer

    if len(problem) < 10 or len(reasoning) < 15 or len(answer) < 2:
        return None
    if arabic_char_ratio(problem + " " + answer) < 0.45:
        return None

    sys_prompt = choose_optional_system_prompt(f"ar_math:{problem[:120]}")
    turns = [
        {"role": "user", "content": problem, "reasoning_content": ""},
        {"role": "assistant", "content": answer, "reasoning_content": reasoning},
    ]
    return finalize_sft_sample(
        source="arabic_math_reasoning_synth",
        turns=turns,
        enable_thinking=True,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# Alias preserved for backwards-compatible imports in existing tests
normalize_arabic_math_row = normalize_arabic_math_synth_row


# ---------------------------------------------------------------------------
# Normalizer 5: FreedomIntelligence/alpaca-gpt4-arabic (and ShareGPT Arabic)
# ---------------------------------------------------------------------------
def normalize_alpaca_gpt4_arabic_row(
    row: Dict[str, Any],
    source_key: str = "alpaca_gpt4_arabic",
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Convert ShareGPT `conversations` (`from`: `human`/`gpt`) into clean Arabic messages."""
    convs = row.get("conversations") or row.get("messages") or []
    if not isinstance(convs, list) or len(convs) < 2:
        return None

    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "model": "assistant",
        "system": "system",
    }
    turns: List[Dict[str, str]] = []
    explicit_sys: Optional[str] = None

    for idx, turn in enumerate(convs):
        if not isinstance(turn, dict):
            continue
        raw_role = str(turn.get("from") or turn.get("role") or "").strip().lower()
        mapped = role_map.get(raw_role)
        content = str(turn.get("value") if turn.get("value") is not None else (turn.get("content") or "")).strip()
        if not mapped or not content:
            continue
        if mapped == "system" and idx == 0:
            explicit_sys = content
            continue
        if mapped in ("user", "assistant"):
            turns.append({"role": mapped, "content": content, "reasoning_content": ""})

    if len(turns) < 2 or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return None
    if len(turns[-1]["content"]) < 15 or arabic_char_ratio(turns[-1]["content"]) < 0.50:
        return None
    if has_english_meta_leak(turns[-1]["content"]):
        return None

    sys_prompt = explicit_sys or choose_optional_system_prompt(f"{source_key}:{turns[0]['content'][:120]}")
    return finalize_sft_sample(
        source=source_key,
        turns=turns,
        enable_thinking=False,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


normalize_acegpt_v2_row = normalize_alpaca_gpt4_arabic_row


# ---------------------------------------------------------------------------
# Normalizer 6: Gheras/Medical_Reasoning_Arabic_Dataset
# ---------------------------------------------------------------------------
def normalize_medical_reasoning_arabic_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `Gheras/Medical_Reasoning_Arabic_Dataset` while rejecting English translation leaks."""
    question = str(row.get("Question") or row.get("question") or "").strip()
    cot = str(row.get("Complex_CoT") or row.get("reasoning") or "").strip()
    response = str(row.get("Response") or row.get("response") or "").strip()

    if len(question) < 15 or len(cot) < 40 or len(response) < 25:
        return None
    if has_english_meta_leak(question) or has_english_meta_leak(cot) or has_english_meta_leak(response):
        return None
    if arabic_char_ratio(question) < 0.70 or arabic_char_ratio(response) < 0.70 or arabic_char_ratio(cot) < 0.65:
        return None

    sys_prompt = choose_optional_system_prompt(f"med_ar:{question[:120]}")
    turns = [
        {"role": "user", "content": question, "reasoning_content": ""},
        {"role": "assistant", "content": response, "reasoning_content": cot},
    ]
    return finalize_sft_sample(
        source="medical_reasoning_arabic",
        turns=turns,
        enable_thinking=True,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Normalizer 7: CohereLabs/aya_dataset (Human-Annotated Arabic)
# ---------------------------------------------------------------------------
_AYA_ARABIC_CODES: Set[str] = {"arb", "ars", "arz", "ajp", "acq", "ary"}


def normalize_aya_dataset_arabic_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize human-annotated Arabic rows from `CohereLabs/aya_dataset`."""
    lang_code = str(row.get("language_code") or "").strip().lower()
    lang_name = str(row.get("language") or "").strip().lower()
    if lang_code and lang_code not in _AYA_ARABIC_CODES and "arabic" not in lang_name:
        return None

    user_input = str(row.get("inputs") or row.get("prompt") or "").strip()
    target_output = str(row.get("targets") or row.get("completion") or "").strip()
    if len(user_input) < 8 or len(target_output) < 15:
        return None

    tools_field = row.get("tools")
    tool_call_field = row.get("tool_call")
    if tools_field:
        tools_str = (
            tools_field
            if isinstance(tools_field, str)
            else json.dumps(tools_field, ensure_ascii=False)
        )
        user_input = f"{user_input}\n\nالأدوات المتاحة (JSON):\n{tools_str}"
    if tool_call_field:
        tc_str = (
            tool_call_field
            if isinstance(tool_call_field, str)
            else json.dumps(tool_call_field, ensure_ascii=False)
        )
        target_output = f"{tc_str}\n{target_output}".strip()

    if not tools_field and arabic_char_ratio(user_input + " " + target_output) < 0.55:
        return None

    sys_prompt = choose_optional_system_prompt(f"aya_ar:{user_input[:120]}")
    turns = [
        {"role": "user", "content": user_input, "reasoning_content": ""},
        {"role": "assistant", "content": target_output, "reasoning_content": ""},
    ]
    return finalize_sft_sample(
        source="aya_dataset_arabic",
        turns=turns,
        enable_thinking=False,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


normalize_aya_arabic_row = normalize_aya_dataset_arabic_row


# ---------------------------------------------------------------------------
# Normalizer 8: VohoAI/voho-saudi-dialogues
# ---------------------------------------------------------------------------
def normalize_voho_saudi_row(
    row: Dict[str, Any],
    system_prompt: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `VohoAI/voho-saudi-dialogues`."""
    raw_messages: Any = row.get("messages")
    if raw_messages is None and row.get("messages_json"):
        try:
            raw_messages = json.loads(row["messages_json"])
        except (TypeError, ValueError):
            raw_messages = None

    turns: List[Dict[str, str]] = []
    explicit_sys: Optional[str] = system_prompt

    if isinstance(raw_messages, list) and raw_messages:
        for idx, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role in ("human", "user"):
                norm_role = "user"
            elif role in ("assistant", "gpt", "model", "bot"):
                norm_role = "assistant"
            elif role == "system" and idx == 0:
                explicit_sys = strip_speaker_prefix(str(msg.get("content", ""))) or explicit_sys
                continue
            else:
                continue
            cleaned = strip_speaker_prefix(str(msg.get("content", "")))
            if cleaned:
                turns.append({"role": norm_role, "content": cleaned, "reasoning_content": ""})
    else:
        u_txt = strip_speaker_prefix(
            str(row.get("user") or row.get("prompt") or row.get("instruction") or row.get("question") or "")
        )
        a_txt = strip_speaker_prefix(
            str(row.get("assistant") or row.get("response") or row.get("output") or row.get("answer") or "")
        )
        if u_txt and a_txt:
            turns = [
                {"role": "user", "content": u_txt, "reasoning_content": ""},
                {"role": "assistant", "content": a_txt, "reasoning_content": ""},
            ]

    if len(turns) < 2 or turns[-1]["role"] != "assistant":
        return None
    if arabic_char_ratio(turns[-1]["content"]) < 0.50:
        return None

    chosen_sys = explicit_sys if explicit_sys is not None else choose_optional_system_prompt(
        f"voho:{turns[0]['content'][:120]}"
    )
    return finalize_sft_sample(
        source="voho_saudi_dialogues",
        turns=turns,
        enable_thinking=False,
        system_prompt=chosen_sys,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Normalizer 9: SaudiArabicLaws ("moj" & "boe", active statutes only)
# ---------------------------------------------------------------------------
def normalize_saudi_laws_row(
    row: Dict[str, Any],
    config_name: str = "moj",
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize an active statutory row from `SaudiArabicLaws` (`moj` or `boe`).

    Filters out repealed (`ملغي`) laws and short cross-reference stubs.
    """
    law_obj = row.get("law") if isinstance(row.get("law"), dict) else {}
    article_obj = row.get("article") if isinstance(row.get("article"), dict) else {}
    retrieval_obj = row.get("retrieval") if isinstance(row.get("retrieval"), dict) else {}

    # Reject repealed statutes when status metadata is present
    law_status = str(law_obj.get("status") or row.get("status") or "").strip()
    if law_status and ("ملغي" in law_status or "ملغى" in law_status):
        return None

    question = str(row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
    answer = str(row.get("answer") or row.get("response") or row.get("output") or "").strip()

    if not (question and answer):
        doc_title = str(
            row.get("law_name")
            or row.get("title")
            or row.get("doc_title")
            or law_obj.get("title_ar")
            or f"الأنظمة واللوائح السعودية ({config_name.upper()})"
        ).strip()
        article_heading = str(
            row.get("article_number")
            or row.get("article_title")
            or row.get("section")
            or article_obj.get("article_title_norm")
            or article_obj.get("heading_raw")
            or (f"المادة {article_obj['number_int']}" if article_obj.get("number_int") else "")
        ).strip()
        body = str(
            row.get("text")
            or row.get("article_text")
            or row.get("content")
            or article_obj.get("text")
            or retrieval_obj.get("contextual_text")
            or ""
        ).strip()

        if len(body) < 40:
            return None

        art_ref = article_heading if article_heading else "هذه المادة النظامية"
        tpl_idx = int.from_bytes(
            hashlib.blake2b(f"{doc_title}:{art_ref}".encode("utf-8", errors="ignore"), digest_size=2).digest(),
            "little",
        ) % len(_SAUDI_LAW_QUESTION_TEMPLATES)
        question = _SAUDI_LAW_QUESTION_TEMPLATES[tpl_idx].format(art=art_ref, doc=doc_title)
        answer = body

    if arabic_char_ratio(answer) < 0.60:
        return None

    sys_prompt = choose_optional_system_prompt(f"saudi_law_{config_name}:{question[:120]}")
    turns = [
        {"role": "user", "content": question, "reasoning_content": ""},
        {"role": "assistant", "content": answer, "reasoning_content": ""},
    ]
    return finalize_sft_sample(
        source=f"saudi_arabic_laws_{config_name}",
        turns=turns,
        enable_thinking=False,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Streaming Loader for All 9 Curated Arabic SFT Sources
# ---------------------------------------------------------------------------
def load_and_normalize_arabic_sft_source(
    source_key: str,
    max_samples: Optional[int] = None,
    streaming: bool = True,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
    seen_prompts: Optional[Set[bytes]] = None,
) -> List[Dict[str, Any]]:
    """Load and normalize rows for any of the 9 curated Arabic SFT dataset keys."""
    load_dotenv_if_present()
    from datasets import load_dataset

    if source_key not in SFT_ARABIC_DATASETS:
        raise KeyError(f"Unknown Arabic SFT dataset key: {source_key}")

    spec = SFT_ARABIC_DATASETS[source_key]
    limit = max_samples if max_samples is not None else spec.target_samples
    if limit <= 0:
        return []
    collected: List[Dict[str, Any]] = []
    dedup_set = seen_prompts if seen_prompts is not None else set()

    def _try_load(repo: str, cfg: Optional[str], split: str):
        if cfg:
            return load_dataset(repo, cfg, split=split, streaming=streaming)
        return load_dataset(repo, split=split, streaming=streaming)

    def _iter_saudi_laws_jsonl(repo: str, cfg: Optional[str]):
        from huggingface_hub import hf_hub_download

        rel_file = (
            "data/BOE/boe_legal_articles_corpus.jsonl"
            if str(cfg or "").lower() == "boe"
            else "data/MOJ/moj_saudi_legal_corpus.jsonl"
        )
        local_path = hf_hub_download(
            repo_id=repo,
            filename=rel_file,
            repo_type="dataset",
        )
        with open(local_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if raw_line:
                    yield json.loads(raw_line)

    for idx_cfg, cfg_name in enumerate(spec.configs):
        remaining_total = limit - len(collected)
        if remaining_total <= 0:
            break
        remaining_cfgs = max(1, len(spec.configs) - idx_cfg)
        target_for_this_cfg = (
            remaining_total
            if remaining_cfgs == 1
            else (remaining_total + remaining_cfgs - 1) // remaining_cfgs
        )

        if source_key == "saudi_arabic_laws":
            try:
                ds = list(_iter_saudi_laws_jsonl("SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus", cfg_name))
            except Exception as jsonl_exc:
                logger.warning(
                    "Direct JSONL download for SaudiArabicLaws (%s) failed (%s); falling back to load_dataset.",
                    cfg_name,
                    jsonl_exc,
                )
                ds = _try_load(spec.repo_id, cfg_name, spec.default_split)
        else:
            try:
                ds = _try_load(spec.repo_id, cfg_name, spec.default_split)
            except Exception as primary_exc:
                if spec.fallback_repo_id:
                    logger.warning(
                        "Primary repo '%s' unavailable (%s); falling back to '%s'.",
                        spec.repo_id,
                        primary_exc,
                        spec.fallback_repo_id,
                    )
                    ds = _try_load(spec.fallback_repo_id, cfg_name, spec.default_split)
                else:
                    raise

        count = 0
        for row in ds:
            item: Optional[Dict[str, Any]] = None
            if source_key == "smolkalam_arabic_science":
                item = normalize_smolkalam_row(
                    row,
                    source_key=source_key,
                    require_thinking=False,
                    min_lr=0.85,
                    min_scr=0.90,
                    tokenizer=tokenizer,
                    decontaminator=decontaminator,
                )
            elif source_key == "smolkalam_arabic_reasoning_cot":
                item = normalize_smolkalam_row(
                    row,
                    source_key=source_key,
                    require_thinking=True,
                    min_lr=0.78,
                    min_scr=0.80,
                    tokenizer=tokenizer,
                    decontaminator=decontaminator,
                )
            elif source_key == "smolkalam_arabic_openhermes":
                item = normalize_smolkalam_row(
                    row,
                    source_key=source_key,
                    require_thinking=False,
                    min_lr=0.88,
                    min_scr=0.93,
                    tokenizer=tokenizer,
                    decontaminator=decontaminator,
                )
            elif source_key == "arabic_math_reasoning_synth":
                item = normalize_arabic_math_synth_row(
                    row, tokenizer=tokenizer, decontaminator=decontaminator
                )
            elif source_key == "alpaca_gpt4_arabic":
                item = normalize_alpaca_gpt4_arabic_row(
                    row, source_key=source_key, tokenizer=tokenizer, decontaminator=decontaminator
                )
            elif source_key == "medical_reasoning_arabic":
                item = normalize_medical_reasoning_arabic_row(
                    row, tokenizer=tokenizer, decontaminator=decontaminator
                )
            elif source_key == "aya_dataset_arabic":
                item = normalize_aya_dataset_arabic_row(
                    row, tokenizer=tokenizer, decontaminator=decontaminator
                )
            elif source_key == "voho_saudi_dialogues":
                item = normalize_voho_saudi_row(
                    row, tokenizer=tokenizer, decontaminator=decontaminator
                )
            elif source_key == "saudi_arabic_laws":
                item = normalize_saudi_laws_row(
                    row,
                    config_name=cfg_name or "moj",
                    tokenizer=tokenizer,
                    decontaminator=decontaminator,
                )

            if item is None:
                continue

            # Deduplicate across the entire SFT corpus on first user turn
            first_user = next(
                (m["content"] for m in item["messages"] if m["role"] == "user"),
                "",
            )
            sig = hashlib.blake2b(
                first_user.strip().lower().encode("utf-8", errors="ignore"),
                digest_size=12,
            ).digest()
            if sig in dedup_set:
                continue
            dedup_set.add(sig)

            collected.append(item)
            count += 1
            if count >= target_for_this_cfg:
                break

        logger.info(
            "Collected %d / %d normalized rows from '%s' (config=%s).",
            count,
            target_for_this_cfg,
            source_key,
            cfg_name,
        )

    return collected
