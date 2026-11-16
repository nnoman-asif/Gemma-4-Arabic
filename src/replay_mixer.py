"""25% Continual-Learning Replay Buffer normalizers and 75/25 stratified mixer.

Protects all 4 Gemma 4 Official Retention Benchmarks against catastrophic forgetting:
1. `nvidia/OpenCodeReasoning` (`split_0`/`split_0` — `<think>` CoT + Python code -> protects `LiveCodeBench-v6`)
2. `open-r1/codeforces-cots` (`solutions_py_decontaminated` — decontaminated competitive Python CoT -> protects `LiveCodeBench-v6`)
3. `m-a-p/CodeFeedback-Filtered-Instruction` (clean Python/C++/SQL instruction pairs)
4. `nvidia/OpenMathInstruct-2` (`train_1M` — step-by-step CoT in `reasoning_content` + `\boxed{expected_answer}` in `content` -> protects `AIME-2026`)
5. `nvidia/OpenScienceReasoning-2` (`train` — graduate STEM `<think>` CoT + `\boxed{}` -> protects `GPQA-Diamond` & `MMLU-Pro`)
6. `HuggingFaceH4/ultrachat_200k` (`train_sft` — multi-turn English instruction following -> protects `MMLU-Pro`)
"""

import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.chat_template import parse_reasoning_and_content
from src.constants import (
    ENGLISH_SYSTEM_PROMPT_POOL,
    SFT_ARABIC_DATASETS,
    SFT_ARABIC_RATIO,
    SFT_MAX_TOKEN_LENGTH_FILTER,
    SFT_REPLAY_DATASETS,
    SFT_REPLAY_RATIO,
    load_dotenv_if_present,
)
from src.data_normalizer import (
    BenchmarkDecontaminator,
    choose_optional_system_prompt,
    finalize_sft_sample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay 1: nvidia/OpenCodeReasoning (`config="split_0"`, `split="split_0"`)
# ---------------------------------------------------------------------------
def normalize_opencode_reasoning_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `nvidia/OpenCodeReasoning` (`input` + `output` containing `<think>...</think>` + Python solution)."""
    prompt = str(row.get("input") or row.get("prompt") or "").strip()
    raw_output = str(row.get("output") or "").strip()
    if len(prompt) < 20 or len(raw_output) < 40:
        return None
    if not (raw_output.startswith("<think>") and "</think>" in raw_output):
        return None

    reasoning, answer = parse_reasoning_and_content(raw_output)
    if len(reasoning) < 40 or len(answer) < 20:
        return None
    if "<think>" in answer or "</think>" in answer:
        return None

    sys_prompt = choose_optional_system_prompt(
        f"ocr:{prompt[:120]}", pool=ENGLISH_SYSTEM_PROMPT_POOL
    )
    turns = [
        {"role": "user", "content": prompt, "reasoning_content": ""},
        {"role": "assistant", "content": answer, "reasoning_content": reasoning},
    ]
    return finalize_sft_sample(
        source="opencode_reasoning_cot",
        turns=turns,
        enable_thinking=True,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        max_tokens=SFT_MAX_TOKEN_LENGTH_FILTER,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Replay 2: open-r1/codeforces-cots (`solutions_py_decontaminated`)
# ---------------------------------------------------------------------------
def normalize_codeforces_cot_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `open-r1/codeforces-cots` (`finish_reason == 'stop'`, `<think>...</think>` + Python)."""
    finish_reason = str(row.get("finish_reason") or "stop").strip().lower()
    if finish_reason and finish_reason != "stop":
        return None

    prompt = str(row.get("prompt") or "").strip()
    gen = str(row.get("generation") or "").strip()
    if not (prompt and gen) and isinstance(row.get("messages"), list) and len(row["messages"]) >= 2:
        prompt = str(row["messages"][0].get("content") or "").strip()
        gen = str(row["messages"][1].get("content") or "").strip()

    if len(prompt) < 20 or len(gen) < 40:
        return None
    if not (gen.startswith("<think>") and "</think>" in gen):
        return None

    reasoning, answer = parse_reasoning_and_content(gen)
    if len(reasoning) < 40 or len(answer) < 20:
        return None

    sys_prompt = choose_optional_system_prompt(
        f"cf_cot:{prompt[:120]}", pool=ENGLISH_SYSTEM_PROMPT_POOL
    )
    turns = [
        {"role": "user", "content": prompt, "reasoning_content": ""},
        {"role": "assistant", "content": answer, "reasoning_content": reasoning},
    ]
    return finalize_sft_sample(
        source="codeforces_python_cot",
        turns=turns,
        enable_thinking=True,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        max_tokens=SFT_MAX_TOKEN_LENGTH_FILTER,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Replay 3: m-a-p/CodeFeedback-Filtered-Instruction
# ---------------------------------------------------------------------------
_ALLOWED_CODE_LANGS: Set[str] = {"python", "cpp", "javascript", "java", "sql", "shell"}


def normalize_codefeedback_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `m-a-p/CodeFeedback-Filtered-Instruction` (`query` + `answer`)."""
    lang = str(row.get("lang") or "").strip().lower()
    if lang and lang not in _ALLOWED_CODE_LANGS:
        return None

    query = str(row.get("query") or row.get("instruction") or "").strip()
    answer = str(row.get("answer") or row.get("output") or "").strip()
    if len(query) < 15 or len(answer) < 30:
        return None

    sys_prompt = choose_optional_system_prompt(
        f"codefb:{query[:120]}", pool=ENGLISH_SYSTEM_PROMPT_POOL
    )
    turns = [
        {"role": "user", "content": query, "reasoning_content": ""},
        {"role": "assistant", "content": answer, "reasoning_content": ""},
    ]
    return finalize_sft_sample(
        source="codefeedback_instruction",
        turns=turns,
        enable_thinking=False,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        max_tokens=SFT_MAX_TOKEN_LENGTH_FILTER,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Replay 4: nvidia/OpenMathInstruct-2 (FIXED: full derivation + \boxed{} answer)
# ---------------------------------------------------------------------------
def normalize_openmath_instruct_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `nvidia/OpenMathInstruct-2` preserving BOTH `generated_solution` CoT AND `\boxed{}` final answer."""
    problem = str(row.get("problem") or row.get("question") or "").strip()
    gen_sol = str(row.get("generated_solution") or row.get("solution") or "").strip()
    expected = str(row.get("expected_answer") or "").strip()

    if len(problem) < 10 or len(gen_sol) < 15:
        return None

    # Strip any embedded <think> tags if present
    parsed_r, parsed_a = parse_reasoning_and_content(gen_sol)
    reasoning = parsed_r if parsed_r else gen_sol

    if expected:
        boxed_str = expected if "\\boxed" in expected else f"\\boxed{{{expected}}}"
        # Build a clean post-thought final response summarizing the result with \boxed{}
        last_para = [p.strip() for p in (parsed_a or gen_sol).split("\n\n") if p.strip()]
        concise_tail = last_para[-1] if last_para and len(last_para[-1]) <= 320 else ""
        if concise_tail and "\\boxed" in concise_tail:
            final_answer = concise_tail
        elif concise_tail:
            final_answer = f"{concise_tail}\n\nFinal Answer: {boxed_str}"
        else:
            final_answer = f"The final answer is {boxed_str}."
    else:
        if parsed_a:
            final_answer = parsed_a
        else:
            paras = [p.strip() for p in gen_sol.split("\n\n") if p.strip()]
            if len(paras) >= 2:
                reasoning = "\n\n".join(paras[:-1])
                final_answer = paras[-1]
            else:
                final_answer = gen_sol

    sys_prompt = choose_optional_system_prompt(
        f"openmath2:{problem[:120]}", pool=ENGLISH_SYSTEM_PROMPT_POOL
    )
    turns = [
        {"role": "user", "content": problem, "reasoning_content": ""},
        {"role": "assistant", "content": final_answer, "reasoning_content": reasoning},
    ]
    return finalize_sft_sample(
        source="openmath_instruct2_cot",
        turns=turns,
        enable_thinking=True,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        max_tokens=3600,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Replay 5: nvidia/OpenScienceReasoning-2 (`<think>...</think>` + STEM answer)
# ---------------------------------------------------------------------------
def normalize_openscience_reasoning_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize `nvidia/OpenScienceReasoning-2` (`input` + `output` with `<think>...</think>`)."""
    prompt = str(row.get("input") or row.get("question") or "").strip()
    raw_output = str(row.get("output") or "").strip()
    if len(prompt) < 15 or len(raw_output) < 40:
        return None
    if not (raw_output.startswith("<think>") and "</think>" in raw_output):
        return None

    reasoning, answer = parse_reasoning_and_content(raw_output)
    if len(reasoning) < 40 or len(answer) < 10:
        return None

    sys_prompt = choose_optional_system_prompt(
        f"osr2:{prompt[:120]}", pool=ENGLISH_SYSTEM_PROMPT_POOL
    )
    turns = [
        {"role": "user", "content": prompt, "reasoning_content": ""},
        {"role": "assistant", "content": answer, "reasoning_content": reasoning},
    ]
    return finalize_sft_sample(
        source="openscience_reasoning2_cot",
        turns=turns,
        enable_thinking=True,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        max_tokens=SFT_MAX_TOKEN_LENGTH_FILTER,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Replay 6: HuggingFaceH4/ultrachat_200k
# ---------------------------------------------------------------------------
def normalize_ultrachat_row(
    row: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize multi-turn English instruction rows from `HuggingFaceH4/ultrachat_200k`."""
    raw_msgs = row.get("messages")
    if not isinstance(raw_msgs, list) or len(raw_msgs) < 2:
        return None

    turns: List[Dict[str, str]] = []
    explicit_sys: Optional[str] = None
    for idx, msg in enumerate(raw_msgs):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system" and idx == 0:
            explicit_sys = content
            continue
        if role in ("user", "assistant"):
            turns.append({"role": role, "content": content, "reasoning_content": ""})

    # Trim trailing user message if conversation ended mid-turn
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()

    if len(turns) < 2 or turns[0]["role"] != "user":
        return None

    sys_prompt = explicit_sys or choose_optional_system_prompt(
        f"ultrachat:{turns[0]['content'][:120]}", pool=ENGLISH_SYSTEM_PROMPT_POOL
    )
    return finalize_sft_sample(
        source="ultrachat_english_replay",
        turns=turns,
        enable_thinking=False,
        system_prompt=sys_prompt,
        tokenizer=tokenizer,
        max_tokens=SFT_MAX_TOKEN_LENGTH_FILTER,
        decontaminator=decontaminator,
    )


# ---------------------------------------------------------------------------
# Streaming Loader & 75% / 25% Stratified Mixer
# ---------------------------------------------------------------------------
def load_and_normalize_replay_source(
    source_key: str,
    max_samples: Optional[int] = None,
    streaming: bool = True,
    tokenizer: Optional[Any] = None,
    decontaminator: Optional[BenchmarkDecontaminator] = None,
    seen_prompts: Optional[Set[bytes]] = None,
) -> List[Dict[str, Any]]:
    """Stream and normalize rows for any of the 6 Replay Buffer dataset keys."""
    load_dotenv_if_present()
    from datasets import load_dataset

    if source_key not in SFT_REPLAY_DATASETS:
        raise KeyError(f"Unknown Replay dataset key: {source_key}")

    spec = SFT_REPLAY_DATASETS[source_key]
    limit = max_samples if max_samples is not None else spec.target_samples
    cfg_name = spec.configs[0] if spec.configs else None
    ds = (
        load_dataset(spec.repo_id, cfg_name, split=spec.default_split, streaming=streaming)
        if cfg_name
        else load_dataset(spec.repo_id, split=spec.default_split, streaming=streaming)
    )

    dedup_set = seen_prompts if seen_prompts is not None else set()
    collected: List[Dict[str, Any]] = []

    for row in ds:
        item: Optional[Dict[str, Any]] = None
        if source_key == "opencode_reasoning_cot":
            item = normalize_opencode_reasoning_row(row, tokenizer=tokenizer, decontaminator=decontaminator)
        elif source_key == "codeforces_python_cot":
            item = normalize_codeforces_cot_row(row, tokenizer=tokenizer, decontaminator=decontaminator)
        elif source_key == "codefeedback_instruction":
            item = normalize_codefeedback_row(row, tokenizer=tokenizer, decontaminator=decontaminator)
        elif source_key == "openmath_instruct2_cot":
            item = normalize_openmath_instruct_row(row, tokenizer=tokenizer, decontaminator=decontaminator)
        elif source_key == "openscience_reasoning2_cot":
            item = normalize_openscience_reasoning_row(row, tokenizer=tokenizer, decontaminator=decontaminator)
        elif source_key == "ultrachat_english_replay":
            item = normalize_ultrachat_row(row, tokenizer=tokenizer, decontaminator=decontaminator)

        if item is None:
            continue

        first_user = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        sig = hashlib.blake2b(
            first_user.strip().lower().encode("utf-8", errors="ignore"),
            digest_size=12,
        ).digest()
        if sig in dedup_set:
            continue
        dedup_set.add(sig)

        collected.append(item)
        if len(collected) >= limit:
            break

    logger.info(
        "Collected %d / %d normalized replay rows from '%s'.",
        len(collected),
        limit,
        source_key,
    )
    return collected


def compute_mix_allocation(
    total_samples: int,
    arabic_ratio: float = SFT_ARABIC_RATIO,
    replay_ratio: float = SFT_REPLAY_RATIO,
) -> Dict[str, Any]:
    """Compute exact per-dataset sample allocations enforcing the 75% Arabic / 25% Replay split."""
    if abs((arabic_ratio + replay_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"arabic_ratio ({arabic_ratio}) + replay_ratio ({replay_ratio}) must equal 1.0"
        )
    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    arabic_total = int(round(total_samples * arabic_ratio))
    replay_total = total_samples - arabic_total

    ar_base_sum = sum(s.target_samples for s in SFT_ARABIC_DATASETS.values())
    rep_base_sum = sum(s.target_samples for s in SFT_REPLAY_DATASETS.values())

    ar_per_source: Dict[str, int] = {}
    allocated_ar = 0
    ar_keys = list(SFT_ARABIC_DATASETS.keys())
    for idx, k in enumerate(ar_keys):
        if idx == len(ar_keys) - 1:
            ar_per_source[k] = max(0, arabic_total - allocated_ar)
        else:
            share = int(round(arabic_total * (SFT_ARABIC_DATASETS[k].target_samples / ar_base_sum)))
            ar_per_source[k] = share
            allocated_ar += share

    rep_per_source: Dict[str, int] = {}
    allocated_rep = 0
    rep_keys = list(SFT_REPLAY_DATASETS.keys())
    for idx, k in enumerate(rep_keys):
        if idx == len(rep_keys) - 1:
            rep_per_source[k] = max(0, replay_total - allocated_rep)
        else:
            share = int(round(replay_total * (SFT_REPLAY_DATASETS[k].target_samples / rep_base_sum)))
            rep_per_source[k] = share
            allocated_rep += share

    return {
        "total_samples": total_samples,
        "arabic_total": arabic_total,
        "replay_total": replay_total,
        "arabic_ratio": arabic_ratio,
        "replay_ratio": replay_ratio,
        "arabic_per_source": ar_per_source,
        "replay_per_source": rep_per_source,
    }


def mix_arabic_and_replay_samples(
    arabic_samples: Sequence[Dict[str, Any]],
    replay_samples: Sequence[Dict[str, Any]],
    arabic_ratio: float = SFT_ARABIC_RATIO,
    replay_ratio: float = SFT_REPLAY_RATIO,
    seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Combine and deterministically shuffle Arabic and Replay samples at exact 75% / 25% ratio."""
    if not arabic_samples:
        raise ValueError("arabic_samples pool cannot be empty.")
    if not replay_samples:
        raise ValueError("replay_samples pool cannot be empty.")

    max_total_by_ar = int(len(arabic_samples) / arabic_ratio)
    max_total_by_rep = int(len(replay_samples) / replay_ratio)
    effective_total = min(max_total_by_ar, max_total_by_rep)

    if effective_total < 4:
        combined = list(arabic_samples) + list(replay_samples)
        rng = random.Random(seed)
        rng.shuffle(combined)
        stats = {
            "arabic_count": len(arabic_samples),
            "replay_count": len(replay_samples),
            "total_count": len(combined),
            "arabic_fraction": round(len(arabic_samples) / len(combined), 4),
            "replay_fraction": round(len(replay_samples) / len(combined), 4),
        }
        return combined, stats

    target_ar = int(round(effective_total * arabic_ratio))
    target_rep = effective_total - target_ar

    rng = random.Random(seed)
    ar_pool = list(arabic_samples)
    rep_pool = list(replay_samples)
    rng.shuffle(ar_pool)
    rng.shuffle(rep_pool)

    selected_ar = ar_pool[:target_ar]
    selected_rep = rep_pool[:target_rep]
    mixed = selected_ar + selected_rep
    rng.shuffle(mixed)

    stats = {
        "arabic_count": len(selected_ar),
        "replay_count": len(selected_rep),
        "total_count": len(mixed),
        "arabic_fraction": round(len(selected_ar) / len(mixed), 6),
        "replay_fraction": round(len(selected_rep) / len(mixed), 6),
    }
    return mixed, stats
