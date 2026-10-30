"""Dataset schema normalizers for all 5 Stage 1 Arabic SFT datasets.

Handles dataset-specific schemas and preprocessing rules:
1. `VohoAI/voho-saudi-dialogues`:
   - Parses JSON string in `messages_json` (or `messages` list) and strips speaker prefixes.
2. `SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus`:
   - Supports both `name="moj"` (flat columns: `doc_title`, `article_title`, `text`, etc.)
     and `name="boe"` (nested dicts under `row["article"]["text"]` or
     `row["retrieval"]["contextual_text"]` + `row["law"]["title_ar"]`).
3. `FreedomIntelligence/AceGPT-v2-tuning`:
   - Maps `conversations` list (`from` / `value`) to standard `user` / `assistant` roles.
4. `Omartificial-Intelligence-Space/Arabic-Math-SFT`:
   - Drops the `image` (`PIL.Image`) column immediately via `.remove_columns(["image"])`
     and formats with `enable_thinking=True`.
5. `CohereForAI/aya_collection_language_split`:
   - Normalizes `inputs` and `targets` into chat turns with optional structured JSON
     tool-calling schemas.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.chat_template import format_gemma4_chat, format_thinking_response
from src.constants import (
    ARABIC_MATH_REASONING_SYSTEM_PROMPT,
    DEFAULT_MSA_SYSTEM_PROMPT,
    SAUDI_DIALECT_SYSTEM_PROMPT,
    SAUDI_LEGAL_SYSTEM_PROMPT,
    SFT_ARABIC_DATASETS,
    SOVEREIGN_TOOL_CALLING_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Regex to strip leading speaker prefixes in dialogue turns
SPEAKER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:user|assistant|human|gpt|bot|model|speaker\s*\d+|client|agent|"
    r"المستخدم|المساعد|العميل|الموظف|المتحدث\s*\d*|س|ج)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)


def strip_speaker_prefix(text: str) -> str:
    """Strip leading speaker labels (e.g. 'User:', 'المستخدم:') from a dialogue turn."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = SPEAKER_PREFIX_PATTERN.sub("", cleaned, count=1)
    return cleaned.strip()


def drop_pil_image_columns(
    dataset: Any,
    columns_to_drop: Sequence[str] = ("image",),
) -> Any:
    """Immediately drop `image` (`PIL.Image`) columns via `.remove_columns(["image"])`."""
    if dataset is None or not hasattr(dataset, "remove_columns"):
        return dataset

    col_names: Optional[Sequence[str]] = getattr(dataset, "column_names", None)
    if col_names is not None:
        present = [c for c in columns_to_drop if c in col_names]
        if present:
            return dataset.remove_columns(present)
        return dataset

    # For IterableDataset where column_names may be None until inspected
    try:
        return dataset.remove_columns(list(columns_to_drop))
    except Exception:
        return dataset


def normalize_voho_saudi_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    strip_prefixes: bool = True,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `VohoAI/voho-saudi-dialogues`.

    Supports both `messages_json` (JSON string) and `messages` (list of dicts),
    strips speaker prefixes, and formats with `enable_thinking=False`.
    """
    raw_messages: Any = row.get("messages_json")
    if isinstance(raw_messages, str) and raw_messages.strip():
        try:
            parsed = json.loads(raw_messages)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed messages_json in Voho row.")
            return None
    elif isinstance(raw_messages, list):
        parsed = raw_messages
    else:
        fallback = row.get("messages")
        if isinstance(fallback, str) and fallback.strip():
            try:
                parsed = json.loads(fallback)
            except json.JSONDecodeError:
                return None
        elif isinstance(fallback, list):
            parsed = fallback
        else:
            return None

    if not isinstance(parsed, list) or not parsed:
        return None

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SAUDI_DIALECT_SYSTEM_PROMPT}
    ]
    for idx, turn in enumerate(parsed):
        if not isinstance(turn, dict):
            continue
        raw_role = str(
            turn.get("role") or turn.get("from") or turn.get("speaker") or ""
        ).strip().lower()
        if raw_role in {"user", "human", "client", "speaker_1", "speaker 1", "المستخدم", "العميل"}:
            role = "user"
        elif raw_role in {"assistant", "gpt", "bot", "model", "agent", "speaker_2", "speaker 2", "المساعد", "الموظف"}:
            role = "assistant"
        elif raw_role == "system":
            role = "system"
        else:
            role = "user" if idx % 2 == 0 else "assistant"

        raw_content = str(turn.get("content") or turn.get("value") or turn.get("text") or "")
        content = strip_speaker_prefix(raw_content) if strip_prefixes else raw_content.strip()
        if not content:
            continue
        if role == "system":
            messages[0] = {"role": "system", "content": content}
        else:
            messages.append({"role": role, "content": content})

    if not any(m["role"] == "user" for m in messages) or not any(
        m["role"] == "assistant" for m in messages
    ):
        return None

    rendered = format_gemma4_chat(
        messages=messages,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {
        "source": "voho_saudi_dialogues",
        "messages": messages,
        "enable_thinking": False,
        "text": rendered,
    }


def normalize_saudi_laws_row(
    row: Dict[str, Any],
    config_name: str = "moj",
    tokenizer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus`.

    Supports both:
    - `config_name="moj"` (flat schema: `doc_title`, `article_title`, `article_topic`, `text`)
    - `config_name="boe"` (nested schema: `row["article"]["text"]` or
      `row["retrieval"]["contextual_text"]` + `row["law"]["title_ar"]`).
    """
    cfg = (config_name or "moj").strip().lower()

    law_title = ""
    article_title = ""
    article_text = ""

    if cfg == "boe" or isinstance(row.get("article"), dict) or isinstance(row.get("retrieval"), dict):
        law_dict = row.get("law") if isinstance(row.get("law"), dict) else {}
        article_dict = row.get("article") if isinstance(row.get("article"), dict) else {}
        retrieval_dict = row.get("retrieval") if isinstance(row.get("retrieval"), dict) else {}

        law_title = str(
            law_dict.get("title_ar")
            or law_dict.get("title")
            or row.get("doc_title")
            or "الأنظمة واللوائح السعودية"
        ).strip()
        article_title = str(
            article_dict.get("article_title_norm")
            or article_dict.get("article_title")
            or article_dict.get("number")
            or row.get("article_title")
            or "المادة النظامية"
        ).strip()
        article_text = str(
            article_dict.get("text")
            or retrieval_dict.get("contextual_text")
            or row.get("text")
            or ""
        ).strip()
    else:
        law_title = str(
            row.get("doc_title")
            or row.get("breadcrumb")
            or "الأنظمة واللوائح السعودية"
        ).strip()
        article_title = str(
            row.get("article_title")
            or row.get("article_topic")
            or "المادة النظامية"
        ).strip()
        article_text = str(
            row.get("text")
            or row.get("doc_summary")
            or ""
        ).strip()

    if not article_text:
        return None

    user_prompt = (
        f"بناءً على {law_title}، ما هو نص وحكم ({article_title})؟"
    )
    assistant_reply = (
        f"وفقاً لـ {law_title} ({article_title}):\n\n{article_text}"
    )

    messages = [
        {"role": "system", "content": SAUDI_LEGAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_reply},
    ]
    rendered = format_gemma4_chat(
        messages=messages,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {
        "source": f"saudi_arabic_laws_{cfg}",
        "messages": messages,
        "enable_thinking": False,
        "text": rendered,
    }


def normalize_acegpt_v2_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `FreedomIntelligence/AceGPT-v2-tuning`.

    Maps `conversations` list (`from` / `value`) to standard `user` / `assistant` roles.
    """
    conv = row.get("conversations") or row.get("messages")
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DEFAULT_MSA_SYSTEM_PROMPT}
    ]

    if isinstance(conv, list) and conv:
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            raw_from = str(turn.get("from") or turn.get("role") or "").strip().lower()
            val = str(turn.get("value") if "value" in turn else turn.get("content", "")).strip()
            if not val:
                continue
            if raw_from in {"human", "user", "instruction", "prompter"}:
                messages.append({"role": "user", "content": val})
            elif raw_from in {"gpt", "assistant", "bot", "model", "output"}:
                messages.append({"role": "assistant", "content": val})
            elif raw_from == "system":
                messages[0] = {"role": "system", "content": val}
    else:
        instruction = str(row.get("instruction") or row.get("question") or "").strip()
        inp = str(row.get("input") or "").strip()
        output = str(row.get("output") or row.get("response") or "").strip()
        if instruction and output:
            user_text = f"{instruction}\n\n{inp}".strip() if inp else instruction
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": output})

    if not any(m["role"] == "user" for m in messages) or not any(
        m["role"] == "assistant" for m in messages
    ):
        return None

    rendered = format_gemma4_chat(
        messages=messages,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {
        "source": "acegpt_v2_tuning",
        "messages": messages,
        "enable_thinking": False,
        "text": rendered,
    }


def normalize_arabic_math_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `Omartificial-Intelligence-Space/Arabic-Math-SFT`.

    Ignores/removes any `image` (`PIL.Image`) field and formats with `enable_thinking=True`.
    """
    clean_row = {k: v for k, v in row.items() if k != "image"}
    problem = str(
        clean_row.get("problem")
        or clean_row.get("question")
        or clean_row.get("instruction")
        or ""
    ).strip()
    solution = str(
        clean_row.get("solution")
        or clean_row.get("answer")
        or clean_row.get("output")
        or ""
    ).strip()
    reasoning = str(clean_row.get("reasoning") or clean_row.get("cot") or "").strip()

    if not problem or not solution:
        return None

    if not reasoning:
        lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
        if len(lines) > 1:
            reasoning = "\n".join(lines[:-1])
            final_answer = lines[-1]
        else:
            reasoning = f"تحليل المسألة وخطوات الحل:\n{solution}"
            final_answer = solution
    else:
        final_answer = solution

    assistant_content = format_thinking_response(
        reasoning=reasoning,
        final_answer=final_answer,
    )
    messages = [
        {"role": "system", "content": ARABIC_MATH_REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": assistant_content},
    ]
    rendered = format_gemma4_chat(
        messages=messages,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return {
        "source": "arabic_math_sft",
        "messages": messages,
        "enable_thinking": True,
        "text": rendered,
    }


def normalize_aya_arabic_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    enable_tool_calling_schema: bool = True,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `CohereForAI/aya_collection_language_split`.

    Normalizes `inputs` and `targets` into chat turns with optional structured
    JSON tool-calling schema formatting.
    """
    inputs_text = str(
        row.get("inputs") or row.get("instruction") or row.get("prompt") or ""
    ).strip()
    targets_text = str(
        row.get("targets") or row.get("output") or row.get("response") or ""
    ).strip()

    if not inputs_text or not targets_text:
        return None

    tool_schema = row.get("tools") or row.get("tool_schema")
    tool_call = row.get("tool_call") or row.get("function_call")

    system_prompt = DEFAULT_MSA_SYSTEM_PROMPT
    user_content = inputs_text
    assistant_content = targets_text

    if enable_tool_calling_schema and (tool_schema is not None or tool_call is not None):
        system_prompt = SOVEREIGN_TOOL_CALLING_SYSTEM_PROMPT
        if tool_schema is not None:
            schema_str = (
                json.dumps(tool_schema, ensure_ascii=False)
                if not isinstance(tool_schema, str)
                else tool_schema.strip()
            )
            user_content = f"الأدوات المتاحة (JSON Schema):\n{schema_str}\n\nالطلب:\n{inputs_text}"
        if tool_call is not None:
            call_str = (
                json.dumps(tool_call, ensure_ascii=False)
                if not isinstance(tool_call, str)
                else tool_call.strip()
            )
            assistant_content = f"{call_str}\n\n{targets_text}".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    rendered = format_gemma4_chat(
        messages=messages,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {
        "source": "aya_arabic_instruct",
        "messages": messages,
        "enable_thinking": False,
        "text": rendered,
    }


def normalize_arabic_sft_iterable(
    source_key: str,
    rows: Iterable[Dict[str, Any]],
    config_name: str = "default",
    tokenizer: Optional[Any] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Normalize an iterable of raw rows from any of the 5 Arabic SFT sources."""
    dispatch: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
        "voho_saudi_dialogues": lambda r: normalize_voho_saudi_row(r, tokenizer=tokenizer),
        "saudi_arabic_laws": lambda r: normalize_saudi_laws_row(
            r, config_name=config_name, tokenizer=tokenizer
        ),
        "acegpt_v2_tuning": lambda r: normalize_acegpt_v2_row(r, tokenizer=tokenizer),
        "arabic_math_sft": lambda r: normalize_arabic_math_row(r, tokenizer=tokenizer),
        "aya_arabic_instruct": lambda r: normalize_aya_arabic_row(r, tokenizer=tokenizer),
    }
    if source_key not in dispatch:
        raise KeyError(
            f"Unknown Arabic SFT source_key: {source_key!r}. "
            f"Valid keys: {sorted(dispatch.keys())}"
        )

    normalizer = dispatch[source_key]
    results: List[Dict[str, Any]] = []
    for row in rows:
        item = normalizer(row)
        if item is not None:
            results.append(item)
            if max_samples is not None and len(results) >= max_samples:
                break
    return results


LIVE_HUB_FALLBACK_SOURCES: Dict[str, List[Tuple[str, Optional[str]]]] = {
    "acegpt_v2_tuning": [
        ("FreedomIntelligence/Evol-Instruct-Arabic-GPT4", "default"),
        ("FreedomIntelligence/Alpaca-Arabic-GPT4", "default"),
    ],
    "aya_arabic_instruct": [
        (
            "parquet",
            "https://huggingface.co/datasets/CohereLabs/aya_collection_language_split/resolve/refs%2Fconvert%2Fparquet/standard_arabic/train/0000.parquet",
        ),
        (
            "parquet",
            "https://huggingface.co/datasets/CohereLabs/aya_collection_language_split/resolve/refs%2Fconvert%2Fparquet/najdi_arabic/train/0000.parquet",
        ),
    ],
}


def load_and_normalize_arabic_sft_source(
    source_key: str,
    max_samples: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    streaming: bool = True,
) -> List[Dict[str, Any]]:
    """Load one of the 5 Arabic SFT datasets from Hugging Face and normalize its schema."""
    from datasets import load_dataset

    if source_key not in SFT_ARABIC_DATASETS:
        raise KeyError(f"Unknown Arabic SFT source key: {source_key!r}")

    spec = SFT_ARABIC_DATASETS[source_key]

    if source_key in LIVE_HUB_FALLBACK_SOURCES:
        targets: List[Tuple[str, Optional[str]]] = LIVE_HUB_FALLBACK_SOURCES[source_key]
    else:
        targets = [(spec.repo_id, cfg_name) for cfg_name in spec.config_names]

    def _collect_from_targets(
        candidate_targets: List[Tuple[str, Optional[str]]]
    ) -> List[Dict[str, Any]]:
        chunk_results: List[Dict[str, Any]] = []
        per_target_limit = (
            max(1, max_samples // len(candidate_targets))
            if max_samples is not None
            else None
        )
        for repo_id, cfg_name in candidate_targets:
            if repo_id == "parquet" and cfg_name:
                ds = load_dataset(
                    "parquet",
                    data_files=cfg_name,
                    split="train",
                    streaming=streaming,
                )
            else:
                load_kwargs: Dict[str, Any] = {
                    "path": repo_id,
                    "split": spec.split,
                    "streaming": streaming,
                }
                if cfg_name and cfg_name != "default":
                    load_kwargs["name"] = cfg_name
                ds = load_dataset(**load_kwargs)

            if spec.drop_columns:
                ds = drop_pil_image_columns(ds, spec.drop_columns)

            normalized_chunk = normalize_arabic_sft_iterable(
                source_key=source_key,
                rows=ds,
                config_name=cfg_name or "default",
                tokenizer=tokenizer,
                max_samples=per_target_limit,
            )
            chunk_results.extend(normalized_chunk)
            if max_samples is not None and len(chunk_results) >= max_samples:
                return chunk_results[:max_samples]
        return chunk_results

    return _collect_from_targets(targets)
