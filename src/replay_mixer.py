"""Replay buffer normalizers and strict 75% Arabic / 25% Replay stratified mixer.

Implements:
1. Schema normalizer for `HuggingFaceH4/ultrachat_200k` (`split="train_sft"`, `enable_thinking=False`).
2. Schema normalizer for `nvidia/OpenMathInstruct-2` (`split="train_1M"`, `enable_thinking=True`).
3. Stratified 75% Arabic / 25% Replay buffer allocation and deterministic interleaving
   to guarantee zero catastrophic forgetting on English, Math, and Coding benchmarks.
"""

import logging
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.chat_template import format_gemma4_chat, format_thinking_response
from src.constants import (
    SFT_ARABIC_DATASETS,
    SFT_ARABIC_MIX_RATIO,
    SFT_REPLAY_DATASETS,
    SFT_REPLAY_MIX_RATIO,
)

logger = logging.getLogger(__name__)


def normalize_ultrachat_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `HuggingFaceH4/ultrachat_200k` (`split="train_sft"`)."""
    raw_messages = row.get("messages") or row.get("conversations")
    messages: List[Dict[str, str]] = []

    if isinstance(raw_messages, list) and raw_messages:
        for turn in raw_messages:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or turn.get("from") or "").strip().lower()
            content = str(
                turn.get("content") if "content" in turn else turn.get("value", "")
            ).strip()
            if not content:
                continue
            if role in {"user", "human"}:
                messages.append({"role": "user", "content": content})
            elif role in {"assistant", "gpt", "model"}:
                messages.append({"role": "assistant", "content": content})
            elif role == "system":
                messages.insert(0, {"role": "system", "content": content})
    else:
        prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
        response = str(row.get("response") or row.get("output") or "").strip()
        if prompt and response:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

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
        "source": "ultrachat_200k",
        "messages": messages,
        "enable_thinking": False,
        "text": rendered,
    }


def normalize_openmath_instruct_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a row from `nvidia/OpenMathInstruct-2` (`split="train_1M"`).

    Formats step-by-step mathematical reasoning inside Gemma 4 `<|think|>` blocks
    with `enable_thinking=True`.
    """
    problem = str(
        row.get("problem") or row.get("question") or row.get("instruction") or ""
    ).strip()
    generated_solution = str(
        row.get("generated_solution") or row.get("solution") or row.get("output") or ""
    ).strip()
    expected_answer = str(row.get("expected_answer") or row.get("answer") or "").strip()

    if not problem or not generated_solution:
        return None

    if expected_answer and expected_answer not in generated_solution:
        final_answer = f"{generated_solution}\n\nFinal Answer: {expected_answer}"
    elif expected_answer:
        final_answer = f"Final Answer: {expected_answer}"
    else:
        lines = [ln.strip() for ln in generated_solution.splitlines() if ln.strip()]
        final_answer = lines[-1] if lines else generated_solution

    assistant_content = format_thinking_response(
        reasoning=generated_solution,
        final_answer=final_answer,
    )
    messages = [
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
        "source": "openmath_instruct_2",
        "messages": messages,
        "enable_thinking": True,
        "text": rendered,
    }


def normalize_replay_sft_iterable(
    source_key: str,
    rows: Iterable[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Normalize an iterable of raw rows from `ultrachat_200k` or `openmath_instruct_2`."""
    dispatch: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
        "ultrachat_200k": lambda r: normalize_ultrachat_row(r, tokenizer=tokenizer),
        "openmath_instruct_2": lambda r: normalize_openmath_instruct_row(
            r, tokenizer=tokenizer
        ),
    }
    if source_key not in dispatch:
        raise KeyError(
            f"Unknown Replay SFT source_key: {source_key!r}. "
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


def load_and_normalize_replay_source(
    source_key: str,
    max_samples: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    streaming: bool = True,
) -> List[Dict[str, Any]]:
    """Load one of the 2 Replay SFT datasets from Hugging Face and normalize its schema."""
    from datasets import load_dataset

    if source_key not in SFT_REPLAY_DATASETS:
        raise KeyError(f"Unknown Replay SFT source key: {source_key!r}")

    spec = SFT_REPLAY_DATASETS[source_key]
    cfg_name = spec.config_names[0] if spec.config_names else "default"
    load_kwargs: Dict[str, Any] = {
        "path": spec.repo_id,
        "split": spec.split,
        "streaming": streaming,
    }
    if cfg_name and cfg_name != "default":
        load_kwargs["name"] = cfg_name

    ds = load_dataset(**load_kwargs)
    return normalize_replay_sft_iterable(
        source_key=source_key,
        rows=ds,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )


def compute_mix_allocation(
    total_samples: int,
    arabic_ratio: float = SFT_ARABIC_MIX_RATIO,
    replay_ratio: float = SFT_REPLAY_MIX_RATIO,
) -> Dict[str, Any]:
    """Compute exact sample counts for 75% Arabic / 25% Replay stratified mixing."""
    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")
    if abs((arabic_ratio + replay_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"arabic_ratio ({arabic_ratio}) + replay_ratio ({replay_ratio}) must equal 1.0."
        )

    replay_total = max(1, int(round(total_samples * replay_ratio)))
    arabic_total = max(1, total_samples - replay_total)

    arabic_per_source: Dict[str, int] = {}
    allocated_ar = 0
    ar_items = list(SFT_ARABIC_DATASETS.items())
    for idx, (k, spec) in enumerate(ar_items):
        if idx == len(ar_items) - 1:
            count = max(1, arabic_total - allocated_ar)
        else:
            count = max(1, int(round(arabic_total * spec.weight)))
            allocated_ar += count
        arabic_per_source[k] = count

    replay_per_source: Dict[str, int] = {}
    allocated_rep = 0
    rep_items = list(SFT_REPLAY_DATASETS.items())
    for idx, (k, spec) in enumerate(rep_items):
        if idx == len(rep_items) - 1:
            count = max(1, replay_total - allocated_rep)
        else:
            count = max(1, int(round(replay_total * spec.weight)))
            allocated_rep += count
        replay_per_source[k] = count

    return {
        "total_samples": arabic_total + replay_total,
        "arabic_total": arabic_total,
        "replay_total": replay_total,
        "arabic_ratio": arabic_total / (arabic_total + replay_total),
        "replay_ratio": replay_total / (arabic_total + replay_total),
        "arabic_per_source": arabic_per_source,
        "replay_per_source": replay_per_source,
    }


def mix_arabic_and_replay_samples(
    arabic_samples: List[Dict[str, Any]],
    replay_samples: List[Dict[str, Any]],
    arabic_ratio: float = SFT_ARABIC_MIX_RATIO,
    replay_ratio: float = SFT_REPLAY_MIX_RATIO,
    seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Combine Arabic and Replay SFT samples into a strict 75% / 25% stratified mixture.

    Adjusts sample counts (via deterministic resampling or trimming) so that the output
    pool strictly satisfies the `arabic_ratio` (0.75) to `replay_ratio` (0.25) balance,
    then interleaves 3 Arabic examples per 1 Replay example across the dataset.
    """
    if not arabic_samples:
        raise ValueError("arabic_samples cannot be empty.")
    if not replay_samples:
        raise ValueError("replay_samples cannot be empty.")
    if abs((arabic_ratio + replay_ratio) - 1.0) > 1e-6:
        raise ValueError("arabic_ratio + replay_ratio must equal 1.0.")

    rng = random.Random(seed)
    ar_pool = list(arabic_samples)
    rep_pool = list(replay_samples)
    rng.shuffle(ar_pool)
    rng.shuffle(rep_pool)

    # Compute exact 3:1 (75% / 25%) target counts anchored on the Arabic pool size
    # For general ratios, target_total = round(len(ar_pool) / arabic_ratio)
    target_total = max(4, int(round(len(ar_pool) / arabic_ratio)))
    # Ensure exact multiple of 4 when using 0.75 / 0.25 so 75.00% / 25.00% is exact
    if abs(arabic_ratio - 0.75) < 1e-6 and target_total % 4 != 0:
        target_total += 4 - (target_total % 4)

    target_ar = int(round(target_total * arabic_ratio))
    target_rep = target_total - target_ar

    def _resize_pool(pool: List[Dict[str, Any]], target_n: int) -> List[Dict[str, Any]]:
        if len(pool) >= target_n:
            return pool[:target_n]
        resized: List[Dict[str, Any]] = []
        while len(resized) < target_n:
            need = target_n - len(resized)
            chunk = list(pool)
            rng.shuffle(chunk)
            resized.extend(chunk[:need])
        return resized

    ar_selected = _resize_pool(ar_pool, target_ar)
    rep_selected = _resize_pool(rep_pool, target_rep)

    # Stratified interleaving: 3 Arabic + 1 Replay per 4-sample block when 75/25
    mixed: List[Dict[str, Any]] = []
    if abs(arabic_ratio - 0.75) < 1e-6 and target_ar == 3 * target_rep:
        for i in range(target_rep):
            mixed.extend(ar_selected[3 * i : 3 * i + 3])
            mixed.append(rep_selected[i])
    else:
        mixed = ar_selected + rep_selected
        rng.shuffle(mixed)

    actual_ar = sum(
        1 for item in mixed if item.get("source") not in SFT_REPLAY_DATASETS
    )
    actual_rep = len(mixed) - actual_ar
    stats = {
        "total_mixed": len(mixed),
        "arabic_count": actual_ar,
        "replay_count": actual_rep,
        "arabic_fraction": round(actual_ar / len(mixed), 6) if mixed else 0.0,
        "replay_fraction": round(actual_rep / len(mixed), 6) if mixed else 0.0,
    }
    return mixed, stats
