"""Gemma 4 chat template wrapper with strict thinking-block token preservation.

Enforces the Gemma 4 Chat Template Rule:
1. Always format turns using:
   `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=...)`
2. Never strip the 4 hidden empty thinking-block tokens (`<|think|>\\n<|think|>\\n`)
   emitted when `enable_thinking=False`.
3. Support structured reasoning blocks (`<|think|>\\n...\\n<|think|>\\n...`) when
   `enable_thinking=True` (for Arabic-Math-SFT and OpenMathInstruct-2).
"""

import inspect
from typing import Any, Dict, List, Optional, Sequence

from src.constants import (
    ASSISTANT_TURN_PREFIX,
    BOS_TOKEN,
    EMPTY_THINKING_BLOCK,
    END_OF_TURN_TOKEN,
    START_OF_TURN_TOKEN,
    THINK_TOKEN,
)

ROLE_MAPPING: Dict[str, str] = {
    "user": "user",
    "human": "user",
    "prompter": "user",
    "instruction": "user",
    "assistant": "assistant",
    "model": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "output": "assistant",
    "system": "system",
}


def normalize_message_roles(
    messages: Sequence[Dict[str, Any]],
    default_system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Normalize conversation turns to standard `system`, `user`, and `assistant` roles."""
    normalized: List[Dict[str, str]] = []

    for msg in messages:
        raw_role = str(msg.get("role") or msg.get("from") or "user").strip().lower()
        role = ROLE_MAPPING.get(raw_role, "user")
        content = str(msg.get("content") if "content" in msg else msg.get("value", "")).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    if default_system_prompt and normalized:
        if normalized[0]["role"] != "system":
            normalized.insert(0, {"role": "system", "content": default_system_prompt.strip()})

    return normalized


def format_thinking_response(reasoning: str, final_answer: str) -> str:
    """Wrap a step-by-step reasoning trace and final response in Gemma 4 `<|think|>` tags."""
    clean_reasoning = (reasoning or "").strip()
    clean_answer = (final_answer or "").strip()
    if not clean_reasoning:
        return f"{EMPTY_THINKING_BLOCK}{clean_answer}"
    return f"{THINK_TOKEN}\n{clean_reasoning}\n{THINK_TOKEN}\n{clean_answer}"


def ensure_assistant_thinking_header(
    content: str,
    enable_thinking: bool = False,
    reasoning: Optional[str] = None,
) -> str:
    """Ensure assistant turn content contains the appropriate Gemma 4 thinking block."""
    stripped = (content or "").strip()
    if stripped.startswith(THINK_TOKEN):
        return stripped
    if enable_thinking:
        if reasoning:
            return format_thinking_response(reasoning, stripped)
        return f"{THINK_TOKEN}\n{stripped}\n{THINK_TOKEN}\n{stripped}"
    return f"{EMPTY_THINKING_BLOCK}{stripped}"


def verify_thinking_block_preserved(
    rendered_text: str,
    enable_thinking: bool = False,
) -> bool:
    """Verify that Gemma 4 thinking tokens were preserved and never stripped."""
    if not rendered_text or not isinstance(rendered_text, str):
        return False
    if enable_thinking:
        return rendered_text.count(THINK_TOKEN) >= 2
    return EMPTY_THINKING_BLOCK in rendered_text


def render_gemma4_template_deterministic(
    messages: Sequence[Dict[str, Any]],
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Deterministic Gemma 4 template renderer used when no HF tokenizer is passed."""
    norm_msgs = normalize_message_roles(messages)
    system_prefix = ""
    parts: List[str] = [BOS_TOKEN]

    idx = 0
    if norm_msgs and norm_msgs[0]["role"] == "system":
        system_prefix = norm_msgs[0]["content"].strip() + "\n\n"
        idx = 1

    first_user_handled = False
    for msg in norm_msgs[idx:]:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            if not first_user_handled and system_prefix:
                content = f"{system_prefix}{content}"
                first_user_handled = True
            parts.append(f"{START_OF_TURN_TOKEN}user\n{content}{END_OF_TURN_TOKEN}\n")
        elif role == "assistant":
            formatted_content = ensure_assistant_thinking_header(
                content=content,
                enable_thinking=enable_thinking,
            )
            parts.append(
                f"{START_OF_TURN_TOKEN}model\n{formatted_content}{END_OF_TURN_TOKEN}\n"
            )

    if add_generation_prompt:
        if enable_thinking:
            parts.append(f"{START_OF_TURN_TOKEN}model\n{THINK_TOKEN}\n")
        else:
            parts.append(f"{START_OF_TURN_TOKEN}model\n{EMPTY_THINKING_BLOCK}")

    return "".join(parts)


def _inject_missing_thinking_blocks(
    rendered: str,
    enable_thinking: bool,
) -> str:
    """Ensure `<start_of_turn>model\\n` turns include the mandatory `<|think|>` block."""
    if verify_thinking_block_preserved(rendered, enable_thinking=enable_thinking):
        return rendered

    marker = ASSISTANT_TURN_PREFIX
    if marker not in rendered:
        return rendered

    segments = rendered.split(marker)
    rebuilt = [segments[0]]
    for seg in segments[1:]:
        if seg.startswith(THINK_TOKEN):
            rebuilt.append(marker + seg)
        elif enable_thinking:
            rebuilt.append(f"{marker}{THINK_TOKEN}\n\n{THINK_TOKEN}\n{seg}")
        else:
            rebuilt.append(f"{marker}{EMPTY_THINKING_BLOCK}{seg}")
    return "".join(rebuilt)


def format_gemma4_chat(
    messages: Sequence[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    default_system_prompt: Optional[str] = None,
    enforce_thinking_verification: bool = True,
) -> str:
    """Format conversation messages using `tokenizer.apply_chat_template`.

    Always invokes `tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=..., enable_thinking=...)` when a tokenizer is supplied,
    and verifies that the 4 hidden empty thinking-block tokens (`<|think|>\\n<|think|>\\n`)
    for `enable_thinking=False` (or reasoning `<|think|>` tags for `enable_thinking=True`)
    are preserved.
    """
    norm_messages = normalize_message_roles(
        messages=messages,
        default_system_prompt=default_system_prompt,
    )
    if not norm_messages:
        raise ValueError("Cannot format an empty message list.")

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            supports_enable_thinking = (
                "enable_thinking" in sig.parameters
                or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
            )
        except (TypeError, ValueError):
            supports_enable_thinking = True

        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if supports_enable_thinking:
            kwargs["enable_thinking"] = enable_thinking

        rendered = tokenizer.apply_chat_template(norm_messages, **kwargs)
        rendered = _inject_missing_thinking_blocks(
            rendered=rendered,
            enable_thinking=enable_thinking,
        )
    else:
        rendered = render_gemma4_template_deterministic(
            messages=norm_messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    if enforce_thinking_verification and any(
        m["role"] == "assistant" for m in norm_messages
    ):
        if not verify_thinking_block_preserved(rendered, enable_thinking=enable_thinking):
            raise ValueError(
                f"Gemma 4 thinking block verification failed (enable_thinking={enable_thinking}). "
                "The 4 hidden thinking tokens must never be stripped."
            )

    return rendered
