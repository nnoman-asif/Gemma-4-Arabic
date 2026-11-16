"""Gemma 4 native chat template formatting, `<|channel>thought` CoT handling, and assistant loss masking.

Verified byte-for-byte against `google/gemma-4-E4B-it` and `google/gemma-4-26B-A4B-it`
`chat_template.jinja`:
- Special tokens:
    `<bos>` (2), `<eos>` (1), `<|turn>` (105), `<turn|>` (106),
    `<|think|>` (98), `<|channel>` (100), `<channel|>` (101).
- Non-thinking training (`enable_thinking=False`):
    * `E4B-it`:
        `<bos><|turn>user\n{U}<turn|>\n<|turn>model\n{A}<turn|>\n`
        (or `<bos><|turn>system\n{S}<turn|>\n...` when a system message is present)
    * `26B-A4B-it` (with `align_26b_empty_thought=True`, matching `add_generation_prompt=True`
      and Unsloth `get_chat_template("gemma-4")`):
        `<bos><|turn>user\n{U}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{A}<turn|>\n`
- Thinking training (`enable_thinking=True`):
    `<bos><|turn>system\n<|think|>\n{optional_S}<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n<|channel>thought\n{R}\n<channel|>{A}<turn|>\n`
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.constants import (
    BOS_TOKEN,
    CHANNEL_END_TOKEN,
    CHANNEL_START_TOKEN,
    EMPTY_THOUGHT_CHANNEL,
    FORBIDDEN_LEGACY_MARKERS,
    THINK_TOKEN,
    THOUGHT_CHANNEL_CLOSE,
    THOUGHT_CHANNEL_OPEN,
    TURN_END_TOKEN,
    TURN_START_TOKEN,
)

_THINK_TAG_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", flags=re.DOTALL)
_CHANNEL_TAG_RE = re.compile(
    r"^\s*<\|channel>thought\s*\r?\n?(.*?)\r?\n?<channel\|>\s*(.*)$",
    flags=re.DOTALL,
)
_LEGACY_PIPE_THINK_RE = re.compile(
    r"^\s*<\|think\|>\s*(.*?)\s*<\|think\|>\s*(.*)$",
    flags=re.DOTALL,
)


def parse_reasoning_and_content(raw_text: str) -> Tuple[str, str]:
    """Split embedded `<think>...</think>` or `<|channel>thought...<channel|>` into `(reasoning, content)`.

    If no closed reasoning tag is present, returns `("", stripped_text)`.
    """
    text = str(raw_text or "").strip()
    if not text:
        return ("", "")

    for pattern in (_THINK_TAG_RE, _CHANNEL_TAG_RE, _LEGACY_PIPE_THINK_RE):
        match = pattern.match(text)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return (reasoning, answer)

    return ("", text)


def make_uniform_message(
    role: str,
    content: str,
    reasoning_content: str = "",
) -> Dict[str, str]:
    """Create a Parquet/Arrow-uniform message dict `{"role", "content", "reasoning_content"}`."""
    clean_role = str(role or "user").strip().lower()
    if clean_role == "model":
        clean_role = "assistant"
    elif clean_role in ("human", "prompter"):
        clean_role = "user"
    elif clean_role in ("gpt", "bot"):
        clean_role = "assistant"

    clean_content = str(content or "").strip()
    clean_reasoning = str(reasoning_content or "").strip()

    if clean_role == "assistant" and not clean_reasoning:
        parsed_r, parsed_c = parse_reasoning_and_content(clean_content)
        if parsed_r and parsed_c:
            clean_reasoning = parsed_r
            clean_content = parsed_c

    return {
        "role": clean_role,
        "content": clean_content,
        "reasoning_content": clean_reasoning if clean_role == "assistant" else "",
    }


def normalize_messages_schema(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Normalize a sequence of message dicts into uniform `{"role", "content", "reasoning_content"}` dicts."""
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "user")
        content = str(msg.get("content") if msg.get("content") is not None else (msg.get("value") or ""))
        reasoning = str(
            msg.get("reasoning_content")
            or msg.get("reasoning")
            or msg.get("thinking")
            or ""
        )
        uniform = make_uniform_message(role=role, content=content, reasoning_content=reasoning)
        if uniform["content"] or uniform["reasoning_content"]:
            normalized.append(uniform)
    return normalized


def format_thinking_response(reasoning: str, final_answer: str) -> str:
    """Format an assistant turn body using Gemma 4's native `<|channel>thought` channel."""
    clean_reasoning = (reasoning or "").strip()
    clean_answer = (final_answer or "").strip()
    if not clean_reasoning:
        return clean_answer
    return f"{THOUGHT_CHANNEL_OPEN}{clean_reasoning}{THOUGHT_CHANNEL_CLOSE}{clean_answer}"


def strip_embedded_thinking_from_content(text: str) -> str:
    """Mirror Gemma 4 `strip_thinking` Jinja macro on assistant `content`."""
    s = str(text or "")
    if CHANNEL_END_TOKEN in s:
        parts = s.split(CHANNEL_END_TOKEN)
        out = ""
        for part in parts:
            if CHANNEL_START_TOKEN in part:
                out += part.split(CHANNEL_START_TOKEN)[0]
            else:
                out += part
        s = out
    elif CHANNEL_START_TOKEN in s:
        s = s.split(CHANNEL_START_TOKEN)[0]
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = re.sub(r"<\|think\|>.*?<\|think\|>", "", s, flags=re.DOTALL)
    return s.strip()


def has_nonempty_final_reasoning(messages: Sequence[Dict[str, Any]]) -> bool:
    """Return True iff the last assistant message has non-empty `reasoning_content`."""
    for msg in reversed(messages):
        role = str(msg.get("role", "")).strip().lower()
        if role in ("assistant", "model"):
            r = str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            if r:
                return True
            parsed_r, _ = parse_reasoning_and_content(str(msg.get("content") or ""))
            return bool(parsed_r)
    return False


def format_gemma4_chat(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    enable_thinking: bool = False,
    add_generation_prompt: bool = False,
    add_bos: bool = True,
    is_26b_moe: bool = False,
    align_26b_empty_thought: bool = False,
) -> str:
    """Render messages into native Gemma 4 `<|turn>...<turn|>` format.

    Invariants enforced:
    1. On completed training conversations (`add_generation_prompt=False`), if
       `enable_thinking=True` is requested without non-empty reasoning on the final
       assistant turn, `enable_thinking` is downgraded to `False` so `<|think|>` is
       never emitted in the system header without a matching `<|channel>thought` body.
    2. When `enable_thinking=True`, `<|turn>system\\n<|think|>\\n{optional_sys}<turn|>\\n`
       is emitted at the top of the sequence, and the final assistant turn renders
       `<|channel>thought\\n{reasoning}\\n<channel|>{answer}<turn|>\\n`.
    3. When `enable_thinking=False`, `<|think|>` is NOT emitted; on `26B-A4B-it`,
       `add_generation_prompt=True` (and completed assistant turns when
       `align_26b_empty_thought=True`) emits `<|channel>thought\\n<channel|>`.
    """
    norm_msgs = normalize_messages_schema(messages)
    if not norm_msgs:
        return ""

    effective_thinking = bool(enable_thinking)
    if not add_generation_prompt and effective_thinking and not has_nonempty_final_reasoning(norm_msgs):
        effective_thinking = False

    # Use real HF Gemma 4 tokenizer when provided and verified to support native `<|turn>`
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        if "<|turn>" in str(tokenizer.chat_template) and not (is_26b_moe and align_26b_empty_thought and not effective_thinking):
            rendered = tokenizer.apply_chat_template(
                norm_msgs,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=effective_thinking,
            )
            if not add_bos and rendered.startswith(BOS_TOKEN):
                rendered = rendered[len(BOS_TOKEN):]
            elif add_bos and not rendered.startswith(BOS_TOKEN):
                rendered = BOS_TOKEN + rendered
            return rendered

    chunks: List[str] = []
    if add_bos:
        chunks.append(BOS_TOKEN)

    loop_msgs = list(norm_msgs)
    has_system = bool(loop_msgs and loop_msgs[0]["role"] in ("system", "developer"))
    sys_text = ""
    if has_system:
        sys_text = loop_msgs[0]["content"].strip()
        loop_msgs = loop_msgs[1:]

    if effective_thinking or sys_text:
        sys_block = f"{TURN_START_TOKEN}system\n"
        if effective_thinking:
            sys_block += f"{THINK_TOKEN}\n"
        if sys_text:
            sys_block += sys_text
        sys_block += f"{TURN_END_TOKEN}\n"
        chunks.append(sys_block)

    last_user_idx = -1
    for idx, m in enumerate(loop_msgs):
        if m["role"] == "user":
            last_user_idx = idx

    for idx, m in enumerate(loop_msgs):
        role = m["role"]
        content = m["content"].strip()
        reasoning = m.get("reasoning_content", "").strip()

        if role == "user":
            chunks.append(f"{TURN_START_TOKEN}user\n{content}{TURN_END_TOKEN}\n")
        elif role == "assistant":
            clean_answer = strip_embedded_thinking_from_content(content)
            turn_body = f"{TURN_START_TOKEN}model\n"
            if effective_thinking and reasoning and idx > last_user_idx:
                turn_body += f"{THOUGHT_CHANNEL_OPEN}{reasoning}{THOUGHT_CHANNEL_CLOSE}{clean_answer}"
            elif not effective_thinking and is_26b_moe and align_26b_empty_thought:
                turn_body += f"{EMPTY_THOUGHT_CHANNEL}{clean_answer}"
            else:
                turn_body += clean_answer
            turn_body += f"{TURN_END_TOKEN}\n"
            chunks.append(turn_body)
        elif role in ("system", "developer") and content:
            chunks.append(f"{TURN_START_TOKEN}system\n{content}{TURN_END_TOKEN}\n")

    if add_generation_prompt:
        gen_prefix = f"{TURN_START_TOKEN}model\n"
        if is_26b_moe and not effective_thinking:
            gen_prefix += EMPTY_THOUGHT_CHANNEL
        chunks.append(gen_prefix)

    return "".join(chunks)


def build_tokenized_sft_Turn_masks(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool = False,
    max_seq_length: int = 4096,
    is_26b_moe: bool = False,
) -> Optional[Dict[str, List[int]]]:
    """Tokenize a conversation for Stage 1 SFT with exact response-only `assistant_masks`.

    Every `<bos>`, `<|turn>system...<turn|>\\n`, and `<|turn>user...<turn|>\\n<|turn>model\\n`
    token receives `assistant_masks = 0` (`labels = -100` inside `UnslothSFTTrainer` BEFORE
    `bfd` sequence packing). Only assistant completion tokens (`<|channel>thought...` +
    `{answer}<turn|>\\n`) receive `assistant_masks = 1`.

    Returns `None` if the conversation contains no assistant turn or exceeds `max_seq_length`
    without any valid assistant tokens.
    """
    norm_msgs = normalize_messages_schema(messages)
    if not norm_msgs:
        return None

    effective_thinking = bool(enable_thinking) and has_nonempty_final_reasoning(norm_msgs)

    # Separate optional leading system message
    loop_msgs = list(norm_msgs)
    sys_text = ""
    if loop_msgs and loop_msgs[0]["role"] in ("system", "developer"):
        sys_text = loop_msgs[0]["content"].strip()
        loop_msgs = loop_msgs[1:]

    last_user_idx = -1
    for idx, m in enumerate(loop_msgs):
        if m["role"] == "user":
            last_user_idx = idx

    input_ids: List[int] = []
    assistant_masks: List[int] = []

    def _encode_segment(seg_text: str) -> List[int]:
        return list(tokenizer.encode(seg_text, add_special_tokens=False))

    # 1. Leading <bos> + optional system block (always masked = 0)
    prefix_str = BOS_TOKEN
    if effective_thinking or sys_text:
        prefix_str += f"{TURN_START_TOKEN}system\n"
        if effective_thinking:
            prefix_str += f"{THINK_TOKEN}\n"
        if sys_text:
            prefix_str += sys_text
        prefix_str += f"{TURN_END_TOKEN}\n"

    prefix_ids = _encode_segment(prefix_str)
    input_ids.extend(prefix_ids)
    assistant_masks.extend([0] * len(prefix_ids))

    # 2. Encode each turn cleanly along exact `<|turn>` / `<turn|>` token boundaries
    for idx, m in enumerate(loop_msgs):
        role = m["role"]
        content = m["content"].strip()
        reasoning = m.get("reasoning_content", "").strip()

        if role == "user":
            user_seg = f"{TURN_START_TOKEN}user\n{content}{TURN_END_TOKEN}\n"
            u_ids = _encode_segment(user_seg)
            input_ids.extend(u_ids)
            assistant_masks.extend([0] * len(u_ids))
        elif role == "assistant":
            # Prompt header `<|turn>model\n` is masked (0)
            header_ids = _encode_segment(f"{TURN_START_TOKEN}model\n")
            input_ids.extend(header_ids)
            assistant_masks.extend([0] * len(header_ids))

            # Assistant completion body + `<turn|>\n` is supervised (1)
            clean_answer = strip_embedded_thinking_from_content(content)
            if effective_thinking and reasoning and idx > last_user_idx:
                comp_seg = f"{THOUGHT_CHANNEL_OPEN}{reasoning}{THOUGHT_CHANNEL_CLOSE}{clean_answer}{TURN_END_TOKEN}\n"
            elif not effective_thinking and is_26b_moe:
                comp_seg = f"{EMPTY_THOUGHT_CHANNEL}{clean_answer}{TURN_END_TOKEN}\n"
            else:
                comp_seg = f"{clean_answer}{TURN_END_TOKEN}\n"

            c_ids = _encode_segment(comp_seg)
            input_ids.extend(c_ids)
            assistant_masks.extend([1] * len(c_ids))

    if len(input_ids) > max_seq_length:
        if effective_thinking:
            # Never slice `<|channel>thought...<channel|>` mid-thought; drop over-length CoT rows
            return None
        eot_ids = _encode_segment(f"{TURN_END_TOKEN}\n")
        if max_seq_length <= len(eot_ids) + 16:
            return None
        keep_len = max_seq_length - len(eot_ids)
        input_ids = input_ids[:keep_len] + eot_ids
        assistant_masks = assistant_masks[:keep_len] + ([1] * len(eot_ids))
        if sum(assistant_masks) < 16:
            return None

    if sum(assistant_masks) < 2 or assistant_masks[0] != 0:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "assistant_masks": assistant_masks,
    }


build_tokenized_sft_turn_masks = build_tokenized_sft_Turn_masks


def verify_thinking_block_preserved(
    rendered_text: str,
    enable_thinking: bool = False,
) -> bool:
    """Assert that a rendered Gemma 4 conversation satisfies all native token invariants."""
    if not rendered_text:
        return False

    for bad_marker in FORBIDDEN_LEGACY_MARKERS:
        if bad_marker in rendered_text:
            return False

    if "<think>" in rendered_text or "</think>" in rendered_text:
        return False

    if f"{TURN_START_TOKEN}user\n" not in rendered_text:
        return False
    if f"{TURN_START_TOKEN}model\n" not in rendered_text:
        return False
    if TURN_END_TOKEN not in rendered_text:
        return False

    if enable_thinking:
        if f"{TURN_START_TOKEN}system\n{THINK_TOKEN}\n" not in rendered_text:
            return False
        if THOUGHT_CHANNEL_OPEN not in rendered_text:
            return False
        if CHANNEL_END_TOKEN not in rendered_text:
            return False
        # Verify non-empty reasoning between <|channel>thought\n and <channel|>
        thought_body = rendered_text.split(THOUGHT_CHANNEL_OPEN, 1)[1].split(CHANNEL_END_TOKEN, 1)[0]
        if not thought_body.strip():
            return False
    else:
        if THINK_TOKEN in rendered_text:
            return False

    return True
