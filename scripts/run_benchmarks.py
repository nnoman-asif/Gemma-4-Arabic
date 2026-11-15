"""Automated full-dataset benchmark runner and Zero-Forgetting Retention Gate evaluator.

Evaluates 10 official full-dataset benchmarks per checkpoint (zero `--limit` by default):
- 6 Native OALL v2 Arabic Benchmarks via `lighteval` (`ArabicMMLU` 40 subjects, `MadinahQA`,
  `AlGhafa` 9 subsets, `ACVA` all 58 subtasks, `EXAMS-AR`, `AraTrust` 8 categories) using
  `acc_norm` (`LogProbCharNorm`, `override_chat_template=False`, `max_model_length=4096`).
- 4 Official Gemma 4 Retention Benchmarks (Thinking Mode ON `enable_thinking=True`,
  `temperature=1.0, top_p=0.95, top_k=64`):
  1. `MMLU-Pro` (`lm-eval>=0.4.13`, 5-shot multiturn CoT, `exact_match,custom-extract`)
  2. `GPQA-Diamond` (`inspect-evals>=0.29.0`, `inspect_evals/gpqa_diamond`, 0-shot CoT, `epochs=1`)
  3. `AIME-2026` (`inspect-evals>=0.29.0`, `inspect_evals/aime2026`, 0-shot CoT, `epochs=4` `avg@4`, no tools)
  4. `LiveCodeBench-v6` (`lighteval@main`, `lcb:codegeneration_v6|0`, 0-shot CoT, `pass@1` `n=1`)
- Supports single-model (`e4b` or `26b_a4b`) and simultaneous dual-GPU (`both`) evaluation
  with atomic per-stage checkpointing to `outputs/eval_reports/live_raw_{model_key}.json`.
"""

import argparse
import gc
import json
import logging
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Enforce deterministic hashing, multiprocessing spawn for vLLM, and code evaluation
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

for _noisy_logger in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Core ACVA subtasks in lm_eval covering Saudi & Arab cultural alignment when --limit is used
ACVA_CORE_SUBTASKS: Tuple[str, ...] = (
    "arabic_leaderboard_acva_Saudi_Arabia_light",
    "arabic_leaderboard_acva_Arab_Empire_light",
    "arabic_leaderboard_acva_Arabic_Culture_light",
    "arabic_leaderboard_acva_Islam_Education_light",
    "arabic_leaderboard_acva_Islamic_law_system_light",
)

# All 40 official OALL v2 ArabicMMLU subject subsets in lighteval.tasks.multilingual.tasks.arabic
ARABIC_MMLU_40_SUBSETS: Tuple[str, ...] = (
    "Islamic Studies",
    "Islamic Studies (Middle School)",
    "Islamic Studies (Primary School)",
    "Islamic Studies (High School)",
    "Driving Test",
    "Natural Science (Middle School)",
    "Natural Science (Primary School)",
    "History (Middle School)",
    "History (Primary School)",
    "History (High School)",
    "General Knowledge",
    "General Knowledge (Middle School)",
    "General Knowledge (Primary School)",
    "Law (Professional)",
    "Physics (High School)",
    "Social Science (Middle School)",
    "Social Science (Primary School)",
    "Management (University)",
    "Arabic Language (Middle School)",
    "Arabic Language (Primary School)",
    "Arabic Language (High School)",
    "Political Science (University)",
    "Philosophy (High School)",
    "Accounting (University)",
    "Computer Science (Middle School)",
    "Computer Science (Primary School)",
    "Computer Science (High School)",
    "Computer Science (University)",
    "Geography (Middle School)",
    "Geography (Primary School)",
    "Geography (High School)",
    "Math (Primary School)",
    "Biology (High School)",
    "Economics (Middle School)",
    "Economics (High School)",
    "Economics (University)",
    "Arabic Language (General)",
    "Arabic Language (Grammar)",
    "Civics (Middle School)",
    "Civics (High School)",
)

# All 58 official OALL v2 ACVA cultural subtasks in lighteval.tasks.multilingual.tasks.arabic
ACVA_ALL_58_SUBSETS: Tuple[str, ...] = (
    "Algeria",
    "Ancient_Egypt",
    "Arab_Empire",
    "Arabic_Architecture",
    "Arabic_Art",
    "Arabic_Astronomy",
    "Arabic_Calligraphy",
    "Arabic_Ceremony",
    "Arabic_Clothing",
    "Arabic_Culture",
    "Arabic_Food",
    "Arabic_Funeral",
    "Arabic_Geography",
    "Arabic_History",
    "Arabic_Language_Origin",
    "Arabic_Literature",
    "Arabic_Math",
    "Arabic_Medicine",
    "Arabic_Music",
    "Arabic_Ornament",
    "Arabic_Philosophy",
    "Arabic_Physics_and_Chemistry",
    "Arabic_Wedding",
    "Bahrain",
    "Comoros",
    "Egypt_modern",
    "InfluenceFromAncientEgypt",
    "InfluenceFromByzantium",
    "InfluenceFromChina",
    "InfluenceFromGreece",
    "InfluenceFromIslam",
    "InfluenceFromPersia",
    "InfluenceFromRome",
    "Iraq",
    "Islam_Education",
    "Islam_branches_and_schools",
    "Islamic_law_system",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Mauritania",
    "Mesopotamia_civilization",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Somalia",
    "Sudan",
    "Syria",
    "Tunisia",
    "United_Arab_Emirates",
    "Yemen",
    "communication",
    "computer_and_phone",
    "daily_life",
    "entertainment",
)

ACVA_SMOKE_SUBSETS: Tuple[str, ...] = (
    "Saudi_Arabia",
    "Arab_Empire",
    "Arabic_Culture",
    "Islam_Education",
    "Islamic_law_system",
)

# Number of subtasks in multi-task lm_eval groups for proportional --limit scaling
GROUP_SUBTASK_COUNTS: Dict[str, int] = {
    "arabicmmlu": 40,
    "arabic_leaderboard_alghafa_light": 9,
    "arabic_leaderboard_acva_light": len(ACVA_CORE_SUBTASKS),
    "mmlu_pro": 14,
}

# Full-dataset LightEval task definitions for the 6 OALL v2 Arabic benchmarks
LIGHTEVAL_SUITE_TASK_MAP: Dict[str, Tuple[str, ...]] = {
    "ArabicMMLU": tuple(f"arabic_mmlu:{sub}|0" for sub in ARABIC_MMLU_40_SUBSETS),
    "MadinahQA": (
        "madinah_qa:Arabic Language (General)|0",
        "madinah_qa:Arabic Language (Grammar)|0",
    ),
    "AlGhafa": (
        "alghafa:mcq_exams_test_ar|0",
        "alghafa:meta_ar_dialects|0",
        "alghafa:meta_ar_msa|0",
        "alghafa:multiple_choice_facts_truefalse_balanced_task|0",
        "alghafa:multiple_choice_grounded_statement_soqal_task|0",
        "alghafa:multiple_choice_grounded_statement_xglue_mlqa_task|0",
        "alghafa:multiple_choice_rating_sentiment_no_neutral_task|0",
        "alghafa:multiple_choice_rating_sentiment_task|0",
        "alghafa:multiple_choice_sentiment_task|0",
    ),
    "ACVA": tuple(f"acva:{sub}|5" for sub in ACVA_ALL_58_SUBSETS),
    "EXAMS-AR": (
        "arabic_exams|0",
    ),
    "AraTrust": (
        "aratrust:Trustfulness|0",
        "aratrust:MentalHealth|0",
        "aratrust:PhysicalHealth|0",
        "aratrust:Offensive|0",
        "aratrust:Ethics|0",
        "aratrust:Privacy|0",
        "aratrust:Unfairness|0",
        "aratrust:Illegal|0",
    ),
}


def resolve_lighteval_arabic_tasks(
    suite_names: List[str],
    limit: Optional[int] = None,
) -> List[str]:
    """Resolve OALL v2 Arabic suite names to `lighteval` task specs (full 40/58 subtasks when `limit is None`)."""
    tasks: List[str] = []
    is_limited = limit is not None and limit > 0
    for sname in suite_names:
        if sname == "ArabicMMLU" and is_limited:
            tasks.append("arabic_mmlu:All|0")
        elif sname == "ACVA" and is_limited:
            tasks.extend(f"acva:{sub}|5" for sub in ACVA_SMOKE_SUBSETS)
        else:
            tasks.extend(LIGHTEVAL_SUITE_TASK_MAP.get(sname, ()))
    return tasks


def load_eval_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load and parse `configs/eval_config.yaml`."""
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config file not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _resolve_hf_eval_cache_dir() -> Optional[str]:
    """Return `/dev/shm/hf_eval_cache` on Linux to avoid root disk usage, or None otherwise."""
    shm_root = pathlib.Path("/dev/shm")
    if shm_root.exists() and os.access(str(shm_root), os.W_OK):
        cache_dir = shm_root / "hf_eval_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)
    return None


def _patch_gemma4_config_for_lm_eval(model: Any) -> None:
    """Expose `text_config` attributes at top level of `Gemma4Config` for `lm_eval` compatibility."""
    cfg = getattr(model, "config", None) or model
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    if cfg is not None:
        try:
            cfg.tie_word_embeddings = True
        except Exception:
            pass
    if cfg is not None and text_cfg is not None:
        try:
            text_cfg.tie_word_embeddings = True
        except Exception:
            pass
        for attr in (
            "vocab_size",
            "eos_token_id",
            "bos_token_id",
            "pad_token_id",
            "max_position_embeddings",
            "hidden_size",
        ):
            if getattr(cfg, attr, None) is None and getattr(text_cfg, attr, None) is not None:
                setattr(cfg, attr, getattr(text_cfg, attr))


def _has_separate_lm_head_in_checkpoint(model_dir: pathlib.Path) -> bool:
    """Return True only if `lm_head.weight` is explicitly stored as an independent tensor in `model_dir`."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        try:
            idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
            weight_map = idx_data.get("weight_map", {})
            return any(k in weight_map for k in ("lm_head.weight", "language_model.lm_head.weight"))
        except Exception:
            return False
    st_path = model_dir / "model.safetensors"
    if st_path.exists():
        try:
            import struct

            with st_path.open("rb") as fh:
                header_size_bytes = fh.read(8)
                if len(header_size_bytes) == 8:
                    header_len = struct.unpack("<Q", header_size_bytes)[0]
                    if 0 < header_len < 100_000_000:
                        header_json = json.loads(fh.read(header_len).decode("utf-8"))
                        return any(
                            k in header_json
                            for k in ("lm_head.weight", "language_model.lm_head.weight")
                        )
        except Exception:
            return False
    return False


def _patch_local_checkpoint_config_json(model_dir: pathlib.Path) -> bool:
    """Ensure `"tie_word_embeddings": true` (when tied) and `"use_cache": true` are set in `model_dir / "config.json"`."""
    cfg_path = model_dir / "config.json"
    if not (model_dir.is_dir() and cfg_path.exists()):
        return False
    has_sep_lm_head = _has_separate_lm_head_in_checkpoint(model_dir)
    try:
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        changed = False
        if not has_sep_lm_head:
            if cfg_data.get("tie_word_embeddings") is not True:
                cfg_data["tie_word_embeddings"] = True
                changed = True
            if isinstance(cfg_data.get("text_config"), dict):
                if cfg_data["text_config"].get("tie_word_embeddings") is not True:
                    cfg_data["text_config"]["tie_word_embeddings"] = True
                    changed = True
        if cfg_data.get("use_cache") is not True:
            cfg_data["use_cache"] = True
            changed = True
        if isinstance(cfg_data.get("text_config"), dict):
            if cfg_data["text_config"].get("use_cache") is not True:
                cfg_data["text_config"]["use_cache"] = True
                changed = True
        if changed:
            cfg_path.write_text(
                json.dumps(cfg_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "Patched %s on disk (tie_word_embeddings / use_cache=true).",
                cfg_path,
            )
        chat_tpl_path = model_dir / "chat_template.jinja"
        if chat_tpl_path.exists() and "26b" in str(model_dir).lower():
            raw_tpl = chat_tpl_path.read_text(encoding="utf-8")
            if "<|channel>thought\\n<channel|>" not in raw_tpl and "{%- if not enable_thinking -%}" in raw_tpl:
                restored_tpl = re.sub(
                    r"(\{%-\s*if\s+not\s+enable_thinking\s*-%\})\s*(\{%-\s*endif\s*-%\})",
                    r"\1\n            {{- '<|channel>thought\\n<channel|>' -}}\n        \2",
                    raw_tpl,
                )
                if restored_tpl != raw_tpl:
                    chat_tpl_path.write_text(restored_tpl, encoding="utf-8")
                    logger.info(
                        "Restored official 26B-A4B non-thinking channel prefix in %s.",
                        chat_tpl_path,
                    )
        return not has_sep_lm_head
    except Exception as exc:
        logger.warning("Could not inspect/patch %s: %s", cfg_path, exc)
        return False


def _prepare_gemma4_config_for_load(model_path: str) -> Any:
    """Ensure `tie_word_embeddings=True` on disk (if local) and in `AutoConfig` before `from_pretrained`."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model  # type: ignore

        if not hasattr(Gemma4Model, "get_input_embeddings"):
            Gemma4Model.get_input_embeddings = (  # type: ignore[attr-defined]
                lambda self: self.language_model.get_input_embeddings()
            )
    except Exception:
        pass

    for candidate_dir in (
        pathlib.Path(model_path),
        PROJECT_ROOT / "outputs" / "merged_gemma4_e4b_arabic" / "merged_16bit",
        PROJECT_ROOT / "outputs" / "merged_gemma4_26b_a4b_arabic" / "merged_16bit",
    ):
        if candidate_dir.is_dir():
            _patch_local_checkpoint_config_json(candidate_dir)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    try:
        config.tie_word_embeddings = True
    except Exception:
        pass
    if getattr(config, "text_config", None) is not None:
        try:
            config.text_config.tie_word_embeddings = True
        except Exception:
            pass
    _patch_gemma4_config_for_lm_eval(config)
    return config


def _ensure_gemma4_tied_lm_head(model: Any, tag: str = "") -> None:
    """Guarantee `lm_head.weight` shares the exact storage pointer of `embed_tokens.weight`."""
    inner = (
        getattr(model, "model", model)
        if hasattr(model, "_model") or type(model).__name__ == "HFLM"
        else model
    )
    base = inner.get_base_model() if hasattr(inner, "get_base_model") else inner

    for obj in (model, inner, base):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            try:
                cfg.tie_word_embeddings = True
            except Exception:
                pass
            if getattr(cfg, "text_config", None) is not None:
                try:
                    cfg.text_config.tie_word_embeddings = True
                except Exception:
                    pass

    in_emb: Any = None
    if hasattr(base, "get_input_embeddings"):
        try:
            in_emb = base.get_input_embeddings()
        except Exception:
            in_emb = None
    if in_emb is None:
        lm_backbone = getattr(getattr(base, "model", None), "language_model", None)
        in_emb = getattr(lm_backbone, "embed_tokens", None) or getattr(
            getattr(base, "model", base), "embed_tokens", None
        )

    out_emb: Any = getattr(base, "lm_head", None)
    if out_emb is None and hasattr(base, "get_output_embeddings"):
        try:
            out_emb = base.get_output_embeddings()
        except Exception:
            out_emb = None
    if out_emb is None and hasattr(base, "model"):
        out_emb = getattr(base.model, "lm_head", None)

    if (
        in_emb is not None
        and out_emb is not None
        and hasattr(in_emb, "weight")
        and hasattr(out_emb, "weight")
        and in_emb.weight is not None
        and out_emb.weight is not None
    ):
        if in_emb.weight.data_ptr() != out_emb.weight.data_ptr():
            out_emb.weight = in_emb.weight
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info(
                "[%s] Explicitly tied lm_head.weight -> embed_tokens.weight (data_ptr=%d)",
                tag or "gemma4",
                in_emb.weight.data_ptr(),
            )
        assert out_emb.weight.data_ptr() == in_emb.weight.data_ptr(), (
            f"[{tag or 'gemma4'}] lm_head.weight data_ptr ({out_emb.weight.data_ptr()}) "
            f"!= embed_tokens.weight data_ptr ({in_emb.weight.data_ptr()})"
        )


def compute_effective_task_limit(
    task_name: str,
    limit: Optional[int],
) -> Optional[int]:
    """Scale `--limit` across multi-subtask `lm_eval` groups so total samples stay near `limit`."""
    if limit is None or limit <= 0:
        return None
    num_subtasks = GROUP_SUBTASK_COUNTS.get(task_name, 1)
    if num_subtasks <= 1:
        return int(limit)
    return max(4, int(math.ceil(float(limit) / float(num_subtasks))))


# ---------------------------------------------------------------------------
# MGSM-Ar Arabic-Indic Numeral & Target Extraction Helpers
# ---------------------------------------------------------------------------

_EASTERN_ARABIC_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬",
    "01234567890123456789.,",
)


def clean_gemma4_completion(text: str, drop_unclosed_thought: bool = True) -> str:
    """Strip Gemma 4 `<|channel>thought...<channel|>` reasoning traces and post-turn `<turn|>` tokens.

    Mirrors Google's official `strip_thinking` Jinja macro in `gemma-4-*-it/chat_template.jinja`:
    completed `<|channel>...<channel|>` blocks are removed, and when `drop_unclosed_thought=True`,
    any unclosed `<|channel>thought...` tail (from hitting `max_tokens` mid-thought) is dropped so
    downstream scorers never grade unfinished scratchpad reasoning.
    """
    s = str(text or "")
    s = re.sub(
        r"^\s*(?:<turn\|>\s*)*(?:<\|turn>|<start_of_turn>)\s*(?:model|assistant)\s*\r?\n?",
        "",
        s,
    )
    for eot in ("<turn|>", "<eos>", "<end_of_turn>", "<|im_end|>", "</s>"):
        if eot in s:
            s = s.split(eot, 1)[0]
    s = re.sub(r"<\|channel>.*?<channel\|>", "", s, flags=re.DOTALL)
    if "<channel|>" in s:
        s = s.split("<channel|>")[-1]
    if "<|channel>" in s:
        if drop_unclosed_thought:
            s = s.split("<|channel>", 1)[0]
        else:
            s = re.sub(r"<\|channel>(?:thought\s*\r?\n?)?", "", s)
    s = re.sub(r"<\|think\|>.*?<\|think\|>", "", s, flags=re.DOTALL)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    if "<think>" in s and drop_unclosed_thought:
        s = s.split("<think>", 1)[0]
    return s.strip()


def _strip_thinking_blocks(text: str) -> str:
    """Remove Gemma 4 `<|think|>`, `<think>`, and `<|channel>thought...<channel|>` blocks and post-`<turn|>` babble without stripping line-1 indentation."""
    s = str(text or "")
    # Strip any leading turn-transition prefix before splitting on end-of-turn markers
    s = re.sub(
        r"^\s*(?:<turn\|>\s*)*(?:<\|turn>|<start_of_turn>)\s*(?:model|assistant)\s*\r?\n?",
        "",
        s,
    )
    # Truncate at the first end-of-turn / EOS marker so post-turn babble is never parsed
    for eot in ("<turn|>", "<eos>", "<end_of_turn>", "<|im_end|>", "</s>"):
        if eot in s:
            s = s.split(eot, 1)[0]
    # Strip Gemma 4 native <|channel>thought...<channel|> blocks
    s_stripped = re.sub(r"<\|channel>thought.*?<channel\|>", "", s, flags=re.DOTALL)
    if "<channel|>" in s_stripped:
        s_stripped = s_stripped.split("<channel|>")[-1]
    if "<|channel>thought" in s_stripped:
        s_stripped = re.sub(r"<\|channel>thought\s*\r?\n?", "", s_stripped)
    if s_stripped.strip():
        s = s_stripped
    else:
        s = re.sub(r"(?:<\|channel>thought\s*\r?\n?|<channel\|>)", "", s)
    # Strip <|think|>...<|think|> and <think>...</think> blocks
    s = re.sub(r"<\|think\|>.*?<\|think\|>", "", s, flags=re.DOTALL)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    if "<|think|>" in s:
        s = s.split("<|think|>")[-1]
    if "<think>" in s:
        s = s.split("<think>")[-1]
    return s.strip("\r\n")


def _normalize_arabic_digits(text: str) -> str:
    """Translate Eastern Arabic (`٠-٩`) and Extended Arabic-Indic (`۰-۹`) digits to ASCII `0-9`."""
    return str(text or "").translate(_EASTERN_ARABIC_DIGIT_MAP)


def _extract_mgsm_ar_target_number(answer: str) -> str:
    """Extract normalized ASCII target number from `Arabic-gsm8k-v2` gold `answer` strings."""
    normalized = _normalize_arabic_digits(str(answer or "")).strip()
    m = re.search(r"#+\s*(-?[\d,]+(?:\.\d+)?)", normalized)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", normalized)
    if nums:
        return nums[-1].replace(",", "")
    return normalized


def _process_mgsm_ar_docs(dataset: Any) -> Any:
    """Normalize Eastern Arabic digits and extract `target_number` for `Omartificial-Intelligence-Space/Arabic-gsm8k-v2`."""

    def _map_row(doc: Dict[str, Any]) -> Dict[str, Any]:
        raw_ans = _normalize_arabic_digits(str(doc.get("answer") or "")).strip()
        target_num = _extract_mgsm_ar_target_number(raw_ans)
        if re.search(r"#+\s*-?[\d,]+(?:\.\d+)?", raw_ans):
            norm_ans = re.sub(r"#+\s*(-?[\d,]+(?:\.\d+)?)\s*$", f"#### {target_num}", raw_ans)
        elif target_num:
            norm_ans = f"{raw_ans}\n#### {target_num}"
        else:
            norm_ans = raw_ans
        return {
            "question": str(doc.get("question") or "").strip(),
            "answer": norm_ans,
            "target_number": target_num,
        }

    return dataset.map(_map_row)


def _truncate_math_continuation(text: str) -> str:
    """Strip thinking blocks, normalize Arabic digits, and truncate post-answer continuations / hallucinated next questions."""
    s = _normalize_arabic_digits(_strip_thinking_blocks(str(text or "")))
    for marker in ("\nQuestion:", "\nQ:", "\nالسؤال:", "\nسؤال:"):
        if marker in s:
            s = s.split(marker, 1)[0]
    m_hash = re.search(r"#{3,4}\s*-?[\d,]+(?:\.\d+)?", s)
    if m_hash:
        s = s[: m_hash.end()]
    return s.strip("\r\n")


def _normalize_arabic_digits_filter(
    resps: List[Any],
    docs: List[Dict[str, Any]],
) -> List[List[str]]:
    """Strip `<|think|>` blocks, truncate post-`####` continuations, and convert Eastern Arabic digits (`٠-٩`) to ASCII `0-9`."""
    _ = docs
    out: List[List[str]] = []
    for inst in resps:
        if isinstance(inst, (list, tuple)):
            out.append([_truncate_math_continuation(str(r)) for r in inst])
        else:
            out.append([_truncate_math_continuation(str(inst))])
    return out


def _process_arc_challenge_chat_docs(dataset: Any) -> Any:
    """Format `allenai/ai2_arc` (`ARC-Challenge`) rows with `A-D` option letters for `<|think|>`-aware generative evaluation."""
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    def _map_row(doc: Dict[str, Any]) -> Dict[str, Any]:
        choices = doc.get("choices") or {}
        texts = list(choices.get("text") or [])
        labels = [
            num_to_letter.get(str(lbl).strip(), str(lbl).strip().upper())
            for lbl in (choices.get("label") or [])
        ]
        choice_lines = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
        stem = str(doc.get("question") or "").strip()
        formatted_q = f"{stem}\nChoices:\n" + "\n".join(choice_lines) + "\nAnswer:"
        raw_key = str(doc.get("answerKey") or "").strip()
        target_letter = num_to_letter.get(raw_key, raw_key.upper())
        return {
            "formatted_question": formatted_q,
            "target_letter": target_letter,
        }

    return dataset.map(_map_row)


def _process_arc_challenge_mc_docs(dataset: Any) -> Any:
    """Format `allenai/ai2_arc` (`ARC-Challenge`) rows with explicit options for 25-shot `multiple_choice` `acc_norm` evaluation."""
    letters = ("A", "B", "C", "D", "E")

    def _map_row(doc: Dict[str, Any]) -> Dict[str, Any]:
        choices_obj = doc.get("choices") or {}
        texts = list(choices_obj.get("text") or [])
        labels = [str(lbl).strip() for lbl in (choices_obj.get("label") or [])]
        ans_key = str(doc.get("answerKey") or "A").strip()
        if ans_key in labels:
            gold_idx = labels.index(ans_key)
        else:
            idx_map = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3,
                "E": 4,
                "1": 0,
                "2": 1,
                "3": 2,
                "4": 3,
                "5": 4,
            }
            gold_idx = idx_map.get(ans_key.upper(), 0)
        gold_idx = max(0, min(gold_idx, max(0, len(texts) - 1)))

        opt_letters = [letters[i] for i in range(min(len(texts), len(letters)))]
        opt_lines = [f"{opt_letters[i]}. {texts[i]}" for i in range(len(opt_letters))]
        stem = str(doc.get("question") or "").strip()
        query = f"Question: {stem}\n" + "\n".join(opt_lines) + "\nAnswer:"
        return {
            "query": query,
            "choice_letters": opt_letters,
            "gold": int(gold_idx),
        }

    return dataset.map(_map_row)


def _extract_arc_mcq_letter(text: str, valid_letters: str = "ABCD") -> str:
    """Strip thinking blocks and extract the final selected option letter (`A`-`D` or `A`-`J`)."""
    cleaned = _strip_thinking_blocks(text).strip()
    if not cleaned:
        return ""
    upper_letters = valid_letters.upper()
    patterns = (
        # 1. LaTeX \boxed{C} or \boxed{(C)} or \boxed{\text{C}}
        rf"\\boxed\s*\{{(?:\\text\s*\{{)?[\*\s\(\[\{{]*([{upper_letters}])[\*\s\)\]\}}]*\}}?\}}",
        # 2a. Explicit final/correct/best answer cues
        rf"(?:(?i:final\s+answer|best\s+answer|the\s+correct\s+(?:answer|option|choice)|correct\s+(?:answer|option|choice))|الإجابة\s+(?:النهائية|الصحيحة))[\*\s]*(?:(?i:is|are|would\s+be|corresponds\s+to)\s*)?[:=\-]*\s*(?:->)?\s*\*{{0,2}}\s*(?:(?i:option|choice|letter)\s+)?[\(\[\{{]*\*{{0,2}}([{upper_letters}])(?![a-zA-Z0-9])(?![\*\)\]\}}\.\s]*(?i:is\s+(?:incorrect|wrong|false|not\b)))",
        # 2b. 'the answer' / 'answer' / 'الإجابة' requiring a separator and negative lookahead for 'is incorrect/wrong'
        rf"(?:(?i:the\s+answer|answer)|الإجابة)[\*\s]*(?:(?i:is|are|would\s+be|corresponds\s+to)\b\s*[:=\-]*|[:=]|->)\s*\*{{0,2}}\s*(?:(?i:option|choice|letter)\s+)?[\(\[\{{]*\*{{0,2}}([{upper_letters}])(?![a-zA-Z0-9])(?![\*\)\]\}}\.\s]*(?i:is\s+(?:incorrect|wrong|false|not\b)))",
        # 3. 'Option C is correct' / '(C) is the correct answer' / 'Choice B is right'
        rf"(?:(?i:option|choice|letter)\s*)?[\*\(\[\{{]*([{upper_letters}])[\*\)\]\}}]*\s+(?i:is\s+correct|is\s+the\s+correct|is\s+the\s+right|is\s+the\s+best|is\s+right)",
        # 4. Conclusion transition ('Therefore, (C)', 'Thus, C.', 'Hence, option C')
        rf"(?i:therefore|thus|hence|consequently|so\s+the\s+answer\s+is)[,\s]+(?:(?i:option|choice)\s*)?[\*\(\[\{{]*([{upper_letters}])[\*\)\]\}}]*(?:\s*[\.\),]|$)",
    )
    for pat in patterns:
        matches = re.findall(pat, cleaned)
        if matches:
            return str(matches[-1]).upper()

    non_empty_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if non_empty_lines:
        # 5. Check the FIRST non-empty line for a standalone option letter or leading option marker
        first_line = non_empty_lines[0]
        first_line_standalone_m = re.match(
            rf"^\s*\*{{0,2}}(?:(?i:option|choice)\s+)?[\(\[\{{]*([{upper_letters}])[\)\]\}}]*\*{{0,2}}\s*$",
            first_line,
        )
        if first_line_standalone_m:
            return first_line_standalone_m.group(1).strip().upper()
        first_line_m = re.match(
            rf"^\s*\*{{0,2}}(?:(?i:option|choice)\s+)?(?:\(([{upper_letters}])\)|([{upper_letters}])(?:[\.\)\:\-\—\,]|\s+(?:(?i:is|because|since)\b|-|—)))(?:\s+|\*{{0,2}}\s+|$)",
            first_line,
        )
        if first_line_m:
            chosen_first = (first_line_m.group(1) or first_line_m.group(2) or "").strip().upper()
            if chosen_first:
                return chosen_first

        # 6. Check the LAST non-empty line (e.g. "(C)", "C.", "C", "Option B: ...")
        last_line = non_empty_lines[-1]
        if not re.match(r"^(?:[\*\-\+]\s+|\d+\.\s+)", last_line):
            last_line_matches = re.findall(
                rf"(?:^\s*\*{{0,2}}(?:(?i:option|choice)\s+)?[\*\(\[\{{]*([{upper_letters}])(?:\s*[\.\)\]\}}\*\:\-]|$))|\(([{upper_letters}])\)",
                last_line,
            )
            if last_line_matches:
                for g1, g2 in reversed(last_line_matches):
                    chosen = (g1 or g2).strip().upper()
                    if chosen:
                        return chosen

    # 7. Standalone bold uppercase letter e.g. **C** or **(C)** or **C.** (ignoring bullet items that state 'is incorrect/wrong')
    bold_matches = re.findall(
        rf"\*\*[\(\[\{{]*([{upper_letters}])[\)\]\}}]*(?:\.[^\*\n]+|\.?)\*\*(?![\s\)\]\}}\.]*(?i:is\s+(?:incorrect|wrong|false|not\b)))",
        cleaned,
    )
    if bold_matches:
        return str(bold_matches[-1]).upper()

    # 8. Final fallback: parenthesized uppercase letter (C) or single-character response
    paren_matches = re.findall(rf"\(([{upper_letters}])\)", cleaned)
    if paren_matches:
        return str(paren_matches[-1]).upper()
    if len(cleaned) == 1 and cleaned.upper() in upper_letters:
        return cleaned.upper()
    return ""


def _extract_mcq_letter_with_doc(
    text: str,
    doc: Any,
    valid_letters: str = "ABCD",
) -> str:
    """Extract MCQ option letter via regex cues first, then fall back to matching option text from `doc`."""
    letter = _extract_arc_mcq_letter(text, valid_letters=valid_letters)
    if letter:
        return letter
    if not isinstance(doc, dict):
        return ""
    cleaned = _strip_thinking_blocks(text).strip()
    if not cleaned:
        return ""

    option_texts: List[str] = []
    choices_obj = doc.get("choices")
    if isinstance(choices_obj, dict) and isinstance(choices_obj.get("text"), list):
        option_texts = [str(x or "").strip() for x in choices_obj["text"]]
    elif isinstance(doc.get("options"), list):
        option_texts = [str(x or "").strip() for x in doc["options"]]

    if not option_texts:
        return ""

    def _norm_txt(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()

    upper_letters = valid_letters.upper()
    norm_cleaned = _norm_txt(cleaned)
    non_empty_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    first_line_norm = _norm_txt(non_empty_lines[0]) if non_empty_lines else ""

    first_line_hits: List[int] = []
    all_hits: List[int] = []
    for idx, opt in enumerate(option_texts[: len(upper_letters)]):
        n_opt = _norm_txt(opt)
        if len(n_opt) < 2:
            continue
        if first_line_norm and (first_line_norm == n_opt or n_opt in first_line_norm):
            first_line_hits.append(idx)
        if n_opt in norm_cleaned:
            all_hits.append(idx)

    if len(first_line_hits) == 1:
        return upper_letters[first_line_hits[0]]
    if len(all_hits) == 1:
        return upper_letters[all_hits[0]]
    return ""


def _extract_arc_mcq_filter(
    resps: List[Any],
    docs: List[Dict[str, Any]],
) -> List[List[str]]:
    """Custom `lm_eval` filter for `arc_challenge_chat` that strips `<|think|>` blocks and extracts `A-D`."""
    out: List[List[str]] = []
    doc_list = list(docs) if isinstance(docs, (list, tuple)) else []
    for idx, inst in enumerate(resps):
        doc_item = doc_list[idx] if idx < len(doc_list) else None
        if isinstance(inst, (list, tuple)):
            out.append([_extract_mcq_letter_with_doc(str(r), doc_item, "ABCD") for r in inst])
        else:
            out.append([_extract_mcq_letter_with_doc(str(inst), doc_item, "ABCD")])
    return out


def _process_mmlu_pro_docs(dataset: Any) -> Any:
    """Format `TIGER-Lab/MMLU-Pro` rows with `A-J` option letters for `<|think|>`-aware generative evaluation."""
    letters = "ABCDEFGHIJ"

    def _map_row(doc: Dict[str, Any]) -> Dict[str, Any]:
        options = list(doc.get("options") or [])
        choice_lines = [
            f"{letters[i]}. {opt}"
            for i, opt in enumerate(options[: len(letters)])
        ]
        stem = str(doc.get("question") or "").strip()
        formatted_q = f"{stem}\nChoices:\n" + "\n".join(choice_lines) + "\nAnswer:"
        target_letter = str(doc.get("answer") or "").strip().upper()
        return {
            "formatted_question": formatted_q,
            "target_letter": target_letter,
        }

    return dataset.map(_map_row)


def _extract_mmlu_pro_mcq_filter(
    resps: List[Any],
    docs: List[Dict[str, Any]],
) -> List[List[str]]:
    """Custom `lm_eval` filter for `mmlu_pro` that strips `<|think|>` blocks and extracts `A-J`."""
    out: List[List[str]] = []
    doc_list = list(docs) if isinstance(docs, (list, tuple)) else []
    for idx, inst in enumerate(resps):
        doc_item = doc_list[idx] if idx < len(doc_list) else None
        if isinstance(inst, (list, tuple)):
            out.append([_extract_mcq_letter_with_doc(str(r), doc_item, "ABCDEFGHIJ") for r in inst])
        else:
            out.append([_extract_mcq_letter_with_doc(str(inst), doc_item, "ABCDEFGHIJ")])
    return out


def _extract_humaneval_code(raw_resp: str, prompt: str, entry_point: str) -> str:
    """Strip `<|think|>` blocks and markdown fences from `humaneval_instruct` generations and assemble `ast.parse()`-verified Python."""
    import ast
    import textwrap

    cleaned = _strip_thinking_blocks(raw_resp)
    prompt_str = str(prompt or "")
    entry_str = str(entry_point or "").strip()

    def _looks_like_unindented_python_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if re.match(r"^(?:for|while|if|elif|else|try|except|with|def|class)\b", s):
            return bool(re.search(r":\s*(?:#.*)?$", s))
        if re.match(r"^(?:return|yield|raise|assert)\b", s):
            return not bool(
                re.match(
                    r"^(?:return|yield|raise|assert)\s+(?:the|a|an|to|from|whether|if)\s+[a-z]+\s+[a-z]+",
                    s,
                    flags=re.IGNORECASE,
                )
            )
        if re.match(r"^(?:pass|break|continue|import\s+\w+|from\s+\w+\s+import)\b", s):
            return True
        return bool(
            re.match(
                r"^[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*(?:=|\+=|-=|\*=|/=|//=|%=|\|=|&=|\^=|\[|\()",
                s,
            )
        )

    def _is_valid_python_with_entry(code_str: str, entry_name: str) -> bool:
        if not code_str or not code_str.strip():
            return False
        try:
            tree = ast.parse(code_str)
            compile(tree, "<humaneval>", "exec")
        except (SyntaxError, ValueError, TypeError):
            return False
        if not entry_name:
            return bool(tree.body)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_name:
                meaningful_stmts = [
                    stmt
                    for stmt in node.body
                    if not isinstance(stmt, ast.Pass)
                    and not (
                        isinstance(stmt, ast.Expr)
                        and isinstance(getattr(stmt, "value", None), ast.Constant)
                        and (
                            isinstance(stmt.value.value, str)
                            or stmt.value.value is Ellipsis
                        )
                    )
                ]
                if meaningful_stmts:
                    return True
        return False

    def _assemble_from_full_def(snippet: str) -> str:
        s_clean = snippet.strip("\r\n")
        if not s_clean.strip():
            return ""
        def_pat = (
            rf"(?m)^([ \t]*def\s+{re.escape(entry_str)}\s*\()"
            if entry_str
            else r"(?m)^([ \t]*def\s+\w+\s*\()"
        )
        if not re.search(def_pat, s_clean):
            return ""

        raw_lines = s_clean.splitlines()
        target_idx = -1
        for idx, ln in enumerate(raw_lines):
            if entry_str and re.match(rf"^[ \t]*def\s+{re.escape(entry_str)}\s*\(", ln):
                target_idx = idx
                break
            if not entry_str and re.match(r"^[ \t]*def\s+\w+\s*\(", ln):
                target_idx = idx
                break
        if target_idx < 0:
            return ""

        # Keep only valid top-level imports / decorators / helper defs before target_idx
        pre_lines: List[str] = []
        in_helper_block = False
        for ln in raw_lines[:target_idx]:
            s_ln = ln.strip()
            if not s_ln:
                if pre_lines:
                    pre_lines.append("")
                continue
            if re.match(r"^(?:import\s+\w+|from\s+[\w\.]+\s+import\s+|@\w+)", s_ln):
                pre_lines.append(s_ln)
                in_helper_block = False
            elif re.match(r"^(?:def|class)\s+\w+.*:\s*(?:#.*)?$", s_ln):
                pre_lines.append(s_ln)
                in_helper_block = True
            elif in_helper_block and ln.startswith((" ", "\t")):
                pre_lines.append(ln)
            else:
                in_helper_block = False

        target_block = textwrap.dedent("\n".join(raw_lines[target_idx:])).strip("\r\n")
        def_lines = target_block.splitlines()
        kept_def_lines: List[str] = []
        seen_body_line = False
        for idx, ln in enumerate(def_lines):
            s_ln = ln.strip()
            if idx == 0:
                kept_def_lines.append(ln)
                continue
            if not s_ln:
                kept_def_lines.append("")
                continue
            if ln.startswith((" ", "\t")):
                kept_def_lines.append(ln)
                if not s_ln.startswith("#"):
                    seen_body_line = True
            else:
                if seen_body_line:
                    if s_ln.startswith("@"):
                        kept_def_lines.append(ln)
                        continue
                    m_top = re.match(r"^(?:def|class)\s+(\w+)\b", s_ln)
                    if m_top:
                        fn_name = m_top.group(1)
                        if (
                            fn_name != entry_str
                            and fn_name != "check"
                            and not fn_name.startswith("test_")
                        ):
                            kept_def_lines.append(ln)
                            continue
                    break
                kept_def_lines.append(ln)

        assembled_fn = "\n".join(kept_def_lines).rstrip()
        if pre_lines:
            assembled_fn = "\n".join(pre_lines).rstrip() + "\n\n" + assembled_fn

        if entry_str:
            prefix_match = re.split(rf"(?m)^def\s+{re.escape(entry_str)}\s*\(", prompt_str, maxsplit=1)
            preamble = prefix_match[0] if len(prefix_match) > 1 else ""
            if preamble.strip() and preamble.strip() not in assembled_fn:
                return preamble.rstrip() + "\n\n" + assembled_fn + "\n"
        return assembled_fn + "\n"

    def _assemble_from_body(snippet: str) -> str:
        raw_lines = [ln.rstrip() for ln in snippet.strip("\r\n").splitlines()]
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()
        if not raw_lines:
            return ""
        sep = "" if prompt_str.endswith("\n") else "\n"

        cleaned_lines: List[str] = []
        seen_indented = False
        first_code_lines = [
            ln for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")
        ]
        first_is_indented = bool(
            first_code_lines and first_code_lines[0].startswith((" ", "\t"))
        )
        for ln in raw_lines:
            s_ln = ln.strip()
            if not s_ln:
                cleaned_lines.append("")
                continue
            if re.match(
                r"^(?:if\s+__name__\b|print\s*\(|#\s*(?:Example|Test)|def\s+(?:test_|check\b))",
                s_ln,
            ):
                break
            if entry_str and re.match(rf"^assert\s+{re.escape(entry_str)}\s*\(", s_ln):
                break
            if entry_str and re.match(rf"^def\s+{re.escape(entry_str)}\s*\(", s_ln):
                if seen_indented:
                    break
                return ""
            if first_is_indented and seen_indented and not ln.startswith((" ", "\t")):
                break
            if ln.startswith((" ", "\t")) and not s_ln.startswith("#"):
                seen_indented = True
            if not first_is_indented and not seen_indented and s_ln.startswith("#"):
                cleaned_lines.append(s_ln)
            else:
                cleaned_lines.append(ln)

        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        if not cleaned_lines:
            return ""

        def _try_block(block_lines: List[str]) -> str:
            if not block_lines:
                return ""
            if entry_str and re.match(
                rf"^[ \t]*def\s+{re.escape(entry_str)}\s*\(", block_lines[0]
            ):
                return ""
            joined = "\n".join(block_lines)
            cand1 = prompt_str + sep + joined + "\n"
            if _is_valid_python_with_entry(cand1, entry_str):
                return cand1
            # Strategy 2: line 0 was unindented while lines 1+ already have >=4 spaces
            if not block_lines[0].startswith((" ", "\t")):
                fixed_lines = ["    " + block_lines[0]] + block_lines[1:]
                cand2 = prompt_str + sep + "\n".join(fixed_lines) + "\n"
                if _is_valid_python_with_entry(cand2, entry_str):
                    return cand2
            # Strategy 3: entire body block was unindented (0 leading spaces on all lines)
            dedented_body = textwrap.indent(textwrap.dedent(joined), "    ")
            cand3 = prompt_str + sep + dedented_body + "\n"
            if _is_valid_python_with_entry(cand3, entry_str):
                return cand3
            return ""

        max_lead = min(6, len(cleaned_lines))
        for start_i in range(max_lead):
            sub = cleaned_lines[start_i:]
            for end_i in range(len(sub), 0, -1):
                res = _try_block(sub[:end_i])
                if res:
                    return res
        return ""

    candidates: List[str] = []

    if "```" in cleaned:
        prefix_before_fence = cleaned.split("```", 1)[0].strip("\r\n")
        first_fence_is_opening_lang = bool(
            re.match(r"^[^`]*```(?:python|py)\b", cleaned, flags=re.IGNORECASE)
        )
        prefix_has_code = any(
            (ln.startswith((" ", "\t")) and not ln.strip().startswith("#"))
            or (entry_str and re.match(rf"^[ \t]*def\s+{re.escape(entry_str)}\s*\(", ln))
            for ln in prefix_before_fence.splitlines()
            if ln.strip()
        )

        # Explicitly tagged ```python ... ``` blocks (never matches across a bare closing ```)
        tagged_blocks = re.findall(
            r"```(?:python|py)[ \t]*\r?\n(.*?)```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not tagged_blocks and not prefix_has_code:
            tagged_blocks = re.findall(
                r"```[ \t]*\r?\n(.*?)```",
                cleaned,
                flags=re.DOTALL,
            )

        if tagged_blocks:
            # Prefer blocks containing `def <entry_str>` or `return`/`yield`, ordered from last to first
            priority_blocks = [
                blk
                for blk in reversed(tagged_blocks)
                if (entry_str and re.search(rf"(?m)^\s*def\s+{re.escape(entry_str)}\s*\(", blk))
                or re.search(r"(?m)^\s*(?:def\s+\w+|return\b|yield\b)", blk)
            ]
            other_blocks = [blk for blk in reversed(tagged_blocks) if blk not in priority_blocks]
            for blk in priority_blocks:
                candidates.append(_assemble_from_full_def(blk))
                candidates.append(_assemble_from_body(blk))
            if not first_fence_is_opening_lang and prefix_before_fence.strip():
                candidates.append(_assemble_from_body(prefix_before_fence))
                candidates.append(_assemble_from_full_def(prefix_before_fence))
            for blk in other_blocks:
                candidates.append(_assemble_from_full_def(blk))
                candidates.append(_assemble_from_body(blk))

        # Unclosed ```python fence fallback
        open_parts = re.split(r"```(?:python|py)[ \t]*\r?\n", cleaned, flags=re.IGNORECASE)
        if len(open_parts) > 1:
            tail_part = open_parts[-1].split("```")[0]
            candidates.append(_assemble_from_full_def(tail_part))
            candidates.append(_assemble_from_body(tail_part))

        candidates.append(_assemble_from_body(prefix_before_fence))
        candidates.append(_assemble_from_full_def(prefix_before_fence))
    else:
        # When no fences exist, try body assembly first (handles indented body followed by repeated def)
        # and full-def assembly second (handles prose followed by def <entry_point>)
        candidates.append(_assemble_from_body(cleaned))
        candidates.append(_assemble_from_full_def(cleaned))

    for cand in candidates:
        if _is_valid_python_with_entry(cand, entry_str):
            return cand

    for cand in candidates:
        if cand and cand.strip():
            return cand

    sep = "" if prompt_str.endswith("\n") else "\n"
    return prompt_str + sep + "    pass\n"


def _build_humaneval_instruct_predictions(
    resps: List[Any],
    docs: List[Dict[str, Any]],
) -> List[List[str]]:
    """Custom `create_test` filter for `humaneval_instruct` that handles `<|think|>` blocks and full `def` blocks."""
    out: List[List[str]] = []
    for inst, doc in zip(resps, docs):
        prompt = str(doc.get("prompt") or "")
        entry_point = str(doc.get("entry_point") or "")
        seq = inst if isinstance(inst, (list, tuple)) else [inst]
        out.append([_extract_humaneval_code(str(r), prompt, entry_point) for r in seq])
    return out


def _build_humaneval_completion_predictions(
    resps: List[Any],
    docs: List[Dict[str, Any]],
) -> List[List[str]]:
    """Standard `lm_eval` HumanEval 0-shot completion filter (`doc['prompt'] + r`, stripping any markdown fence)."""
    out: List[List[str]] = []
    for inst, doc in zip(resps, docs):
        prompt = str(doc.get("prompt") or "")
        seq = inst if isinstance(inst, (list, tuple)) else [inst]
        preds: List[str] = []
        for r in seq:
            s = _strip_thinking_blocks(str(r))
            if s.lstrip().startswith("```"):
                s = re.sub(
                    r"^\s*```(?:python|py)?[ \t]*\r?\n",
                    "",
                    s,
                    count=1,
                    flags=re.IGNORECASE,
                )
            if "```" in s:
                s = s[: s.find("```")]
            preds.append(prompt + s)
        out.append(preds)
    return out


def build_custom_lm_eval_task_dict(task_name: str, num_fewshot: int = 0) -> Dict[str, Any]:
    """Return task configuration dictionary for `mgsm_ar`, `arc_challenge`, `humaneval`, and metadata stubs."""
    key = task_name.strip().lower()
    if key == "mgsm_ar":
        return {
            "task": "mgsm_ar",
            "dataset_path": "Omartificial-Intelligence-Space/Arabic-gsm8k-v2",
            "dataset_name": "default",
            "training_split": "main_train",
            "test_split": "main_test",
            "fewshot_split": "main_train",
            "num_fewshot": int(num_fewshot),
            "output_type": "generate_until",
            "process_docs": _process_mgsm_ar_docs,
            "doc_to_text": "السؤال: {{question}}\nالإجابة:",
            "doc_to_target": "{{target_number}}",
            "fewshot_config": {
                "sampler": "default",
                "split": "main_train",
                "doc_to_target": "{{answer}}",
            },
            "generation_kwargs": {
                "until": [
                    "\nالسؤال:",
                    "\nسؤال:",
                    "\nQuestion:",
                    "<end_of_turn>",
                    "<turn|>",
                    "<eos>",
                    "</s>",
                    "<|im_end|>",
                ],
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 4096,
            },
            "filter_list": [
                {
                    "name": "flexible-extract",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _normalize_arabic_digits_filter,
                        },
                        {
                            "function": "regex",
                            "group_select": -1,
                            "regex_pattern": "(-?[$0-9.,]{2,})|(-?[0-9]+)",
                        },
                        {"function": "take_first"},
                    ],
                },
                {
                    "name": "strict-match",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _normalize_arabic_digits_filter,
                        },
                        {
                            "function": "regex",
                            "regex_pattern": "#{3,4}\\s*(\\-?[0-9\\.\\,]+)",
                        },
                        {"function": "take_first"},
                    ],
                },
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "ignore_case": True,
                    "ignore_punctuation": True,
                    "regexes_to_ignore": [",", "\\$", "\\.$"],
                }
            ],
            "metadata": {"version": 1.0},
        }
    if key == "mmlu_pro":
        return {
            "task": "mmlu_pro",
            "dataset_path": "TIGER-Lab/MMLU-Pro",
            "test_split": "test",
            "validation_split": "validation",
            "fewshot_split": "validation",
            "num_fewshot": 0,
            "output_type": "generate_until",
            "process_docs": _process_mmlu_pro_docs,
            "doc_to_text": "Select the correct option letter (A, B, C, D, E, F, G, H, I, or J) for the following question and end your response with 'Final Answer: X'.\n\nQuestion: {{formatted_question}}",
            "doc_to_target": "{{target_letter}}",
            "generation_kwargs": {
                "until": ["\nQuestion:", "<end_of_turn>", "<turn|>", "<eos>", "</s>", "<|im_end|>"],
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 4096,
            },
            "filter_list": [
                {
                    "name": "custom-extract",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _extract_mmlu_pro_mcq_filter,
                        },
                        {"function": "take_first"},
                    ],
                }
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "ignore_case": True,
                    "ignore_punctuation": True,
                }
            ],
            "metadata": {"version": 1.0},
        }
    if key == "arc_challenge":
        return {
            "task": "arc_challenge",
            "dataset_path": "allenai/ai2_arc",
            "dataset_name": "ARC-Challenge",
            "training_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "fewshot_split": "train",
            "num_fewshot": int(num_fewshot),
            "output_type": "multiple_choice",
            "process_docs": _process_arc_challenge_mc_docs,
            "doc_to_text": "{{query}}",
            "doc_to_target": "gold",
            "doc_to_choice": "choice_letters",
            "metric_list": [
                {
                    "metric": "acc",
                    "aggregation": "mean",
                    "higher_is_better": True,
                },
                {
                    "metric": "acc_norm",
                    "aggregation": "mean",
                    "higher_is_better": True,
                },
            ],
            "metadata": {"version": 1.0},
        }
    if key == "arc_challenge_chat":
        return {
            "task": "arc_challenge_chat",
            "dataset_path": "allenai/ai2_arc",
            "dataset_name": "ARC-Challenge",
            "training_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "fewshot_split": "train",
            "num_fewshot": int(num_fewshot),
            "output_type": "generate_until",
            "process_docs": _process_arc_challenge_chat_docs,
            "doc_to_text": "Select the correct option letter (A, B, C, or D) for the following question and end your response with 'Final Answer: X'.\n\nQuestion: {{formatted_question}}",
            "doc_to_target": "{{target_letter}}",
            "generation_kwargs": {
                "until": ["\nQuestion:", "<end_of_turn>", "<turn|>", "<eos>", "</s>", "<|im_end|>"],
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 4096,
            },
            "filter_list": [
                {
                    "name": "custom-extract",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _extract_arc_mcq_filter,
                        },
                        {"function": "take_first"},
                    ],
                }
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "ignore_case": True,
                    "ignore_punctuation": True,
                }
            ],
            "metadata": {"version": 1.0},
        }
    if key == "humaneval":
        try:
            from lm_eval.tasks.humaneval.utils import pass_at_k
        except Exception:

            def pass_at_k(references: List[str], predictions: List[List[str]], k: Optional[List[int]] = None) -> Dict[str, float]:  # type: ignore
                _ = (references, predictions, k)
                return {"pass@1": 0.0}

        return {
            "task": "humaneval",
            "dataset_path": "openai/openai_humaneval",
            "test_split": "test",
            "num_fewshot": 0,
            "output_type": "generate_until",
            "doc_to_text": (
                "Write a solution to the following problem and make sure that it passes the tests:\n"
                "```python\n{{prompt}}\n```\n"
                "Here is the completed function:\n```python\n{{prompt}}"
            ),
            "doc_to_target": "{{test}}\ncheck({{entry_point}})",
            "unsafe_code": True,
            "generation_kwargs": {
                "until": [
                    "\nif __name__",
                    "<end_of_turn>",
                    "<turn|>",
                    "<eos>",
                    "</s>",
                    "<|im_end|>",
                ],
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 4096,
            },
            "filter_list": [
                {
                    "name": "create_test",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _build_humaneval_instruct_predictions,
                        },
                    ],
                }
            ],
            "metric_list": [
                {
                    "metric": pass_at_k,
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "k": [1],
                }
            ],
            "metadata": {"version": 1.0},
        }
    if key == "humaneval_instruct":
        try:
            from lm_eval.tasks.humaneval.utils import pass_at_k
        except Exception:

            def pass_at_k(references: List[str], predictions: List[List[str]], k: Optional[List[int]] = None) -> Dict[str, float]:  # type: ignore
                _ = (references, predictions, k)
                return {"pass@1": 0.0}

        return {
            "task": "humaneval_instruct",
            "dataset_path": "openai/openai_humaneval",
            "test_split": "test",
            "num_fewshot": 0,
            "output_type": "generate_until",
            "doc_to_text": (
                "Write a complete, self-contained Python solution to the following problem and make sure that it passes the tests:\n"
                "```python\n{{prompt}}\n```"
            ),
            "doc_to_target": "{{test}}\ncheck({{entry_point}})",
            "unsafe_code": True,
            "generation_kwargs": {
                "until": ["\nif __name__", "<end_of_turn>", "<turn|>", "<eos>", "</s>", "<|im_end|>"],
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 4096,
            },
            "filter_list": [
                {
                    "name": "create_test",
                    "filter": [
                        {
                            "function": "custom",
                            "filter_fn": _build_humaneval_instruct_predictions,
                        },
                    ],
                }
            ],
            "metric_list": [
                {
                    "metric": pass_at_k,
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "k": [1],
                }
            ],
            "metadata": {"version": 1.0},
        }
    if key == "arabicmmlu":
        return {
            "task": "arabicmmlu",
            "dataset_path": "MBZUAI/ArabicMMLU",
            "output_type": "multiple_choice",
            "num_fewshot": int(num_fewshot),
        }
    if key == "madinahqa":
        return {
            "task": "madinahqa",
            "dataset_path": "MBZUAI/MadinahQA",
            "output_type": "multiple_choice",
            "num_fewshot": int(num_fewshot),
        }
    if key == "aratrust":
        return {
            "task": "aratrust",
            "dataset_path": "asas-ai/AraTrust",
            "output_type": "multiple_choice",
            "num_fewshot": int(num_fewshot),
        }
    raise KeyError(f"Unknown custom task dictionary key: '{task_name}'")


def resolve_lm_eval_task_specs(
    task_name: str,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
) -> Tuple[List[Union[str, Dict[str, Any]]], str, Optional[int]]:
    """Resolve a benchmark's `lm_eval_task` into `(tasks_arg, primary_result_key, effective_limit)`."""
    clean_name = task_name.strip()
    eff_limit = compute_effective_task_limit(clean_name, limit)

    if clean_name.lower() in (
        "mgsm_ar",
        "arc_challenge",
        "arc_challenge_chat",
        "humaneval",
        "humaneval_instruct",
    ):
        custom_dict = build_custom_lm_eval_task_dict(clean_name, num_fewshot=num_fewshot)
        return ([custom_dict], str(custom_dict["task"]), eff_limit)

    if clean_name == "arabic_leaderboard_acva_light" and limit is not None and limit > 0:
        return (list(ACVA_CORE_SUBTASKS), clean_name, eff_limit)

    return ([clean_name], clean_name, eff_limit)


def evaluate_retention_gate(
    anchor_results: Dict[str, Dict[str, Any]],
    min_retention_ratio: float = 0.998,
    max_allowed_drop_pct: float = 0.20,
    sweep_lambdas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Verify zero catastrophic forgetting across all 4 English/Math/Code anchors."""
    violations: List[str] = []
    ratios: List[float] = []

    for name, entry in anchor_results.items():
        base = float(entry["baseline_score"])
        tuned = float(entry["evaluated_score"])
        delta = round(tuned - base, 4)
        ratio = round(tuned / base, 6) if base > 0 else 1.0
        ratios.append(ratio)

        allowed_drop = float(entry.get("max_regression_pct", max_allowed_drop_pct))
        n_samples = int(entry.get("n_samples", 0) or 0)
        if 0 < n_samples <= 250:
            single_item_pct = 100.0 / float(n_samples)
            effective_allowed_drop = max(allowed_drop, single_item_pct + 0.05)
            effective_min_ratio = (
                min(min_retention_ratio, max(0.0, (base - effective_allowed_drop) / base))
                if base > 0
                else min_retention_ratio
            )
        else:
            effective_allowed_drop = allowed_drop
            effective_min_ratio = min_retention_ratio

        passed = (delta >= -effective_allowed_drop) and (ratio >= effective_min_ratio)
        entry["delta"] = delta
        entry["retention_ratio"] = ratio
        entry["passed"] = passed
        if not passed:
            violations.append(name)

    min_observed_ratio = min(ratios) if ratios else 1.0
    gate_passed = len(violations) == 0
    recommended_lambda = 0.85 if gate_passed else (
        min(sweep_lambdas) if sweep_lambdas else 0.80
    )

    return {
        "gate_passed": gate_passed,
        "min_observed_retention_ratio": round(min_observed_ratio, 6),
        "required_min_retention_ratio": min_retention_ratio,
        "max_allowed_drop_pct": max_allowed_drop_pct,
        "violated_anchors": violations,
        "recommended_ties_lambda": recommended_lambda,
    }


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Render a clean GitHub-Flavored Markdown report using Base | Arabic-Tuned headers."""
    lines: List[str] = [
        "# Gemma-4-Arabic Family Benchmark Evaluation Report",
        "",
        f"- **Model Evaluated:** `{report_data['model_path']}`",
        f"- **Evaluation Mode:** `{report_data['mode']}`",
        f"- **Zero-Forgetting Retention Gate:** `{'PASSED' if report_data['retention_gate']['gate_passed'] else 'FAILED'}` "
        f"(Min Retention Ratio: `{report_data['retention_gate']['min_observed_retention_ratio']:.4f}`, "
        f"Threshold: `{report_data['retention_gate']['required_min_retention_ratio']:.4f}`)",
        "",
        f"## 1. OALL v2 Arabic Benchmarks ({len(report_data['arabic_benchmarks'])} Suites)",
        "",
        "| Benchmark | Task (`lighteval`) | Metric | Shots | Base (%) | Arabic-Tuned (%) | Gain (% pts) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for name, item in report_data["arabic_benchmarks"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        gain = ev - base
        sign = "+" if gain >= 0 else ""
        task_label = item.get("lighteval_task") or item.get("lm_eval_task") or name
        lines.append(
            f"| **{name}** | `{task_label}` | `{item['metric']}` | "
            f"{item['num_fewshot']} | {base:.2f} | **{ev:.2f}** | "
            f"`{sign}{gain:.2f}` |"
        )

    lines.extend(
        [
            "",
            f"## 2. Official Gemma 4 Retention Gate Anchors ({len(report_data['retention_anchors'])} Suites)",
            "",
            "| Anchor | Domain | Task | Shots | Base (%) | Arabic-Tuned (%) | Delta (% pts) | Retention Ratio |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for name, item in report_data["retention_anchors"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        delta = float(item["delta"])
        ratio = float(item["retention_ratio"])
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| **{name}** | `{item['domain']}` | `{item['lm_eval_task']}` | "
            f"{item['num_fewshot']} | {base:.2f} | **{ev:.2f}** | "
            f"`{sign}{delta:.2f}` | `{ratio:.4f}` |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_lm_eval_results_json(
    results_payload: Dict[str, Any],
    task_name: str,
    metric_hint: str = "acc_norm",
) -> float:
    """Extract the primary percentage score (`0.0` to `100.0`) from an `lm_eval` JSON dict."""
    results_block = results_payload.get("results", {})
    if not isinstance(results_block, dict) or not results_block:
        raise ValueError(f"lm_eval results JSON has no 'results' entries for task '{task_name}'.")

    candidate_keys = [
        metric_hint,
        f"{metric_hint},none",
        f"{metric_hint},flexible-extract",
        f"{metric_hint},strict-match",
        f"{metric_hint},create_test",
        "acc_norm,none",
        "acc,none",
        "exact_match,custom-extract",
        "exact_match,remove_whitespace",
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "pass@1,create_test",
        "pass@1,none",
        "acc_norm",
        "acc",
        "exact_match",
        "pass@1",
    ]

    task_metrics = results_block.get(task_name)
    if task_metrics is None:
        subtask_dicts = [v for v in results_block.values() if isinstance(v, dict)]
        if not subtask_dicts:
            raise ValueError(f"No valid metric dictionary found in lm_eval results for '{task_name}'.")
        subtask_scores: List[float] = []
        for sub_d in subtask_dicts:
            for key in candidate_keys:
                if key in sub_d and isinstance(sub_d[key], (int, float)):
                    val = float(sub_d[key])
                    subtask_scores.append(val * 100.0 if val <= 1.0 else val)
                    break
        if subtask_scores:
            return round(sum(subtask_scores) / len(subtask_scores), 2)
        task_metrics = subtask_dicts[0]

    for key in candidate_keys:
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            raw_val = float(task_metrics[key])
            return round(raw_val * 100.0 if raw_val <= 1.0 else raw_val, 2)

    for key, val in task_metrics.items():
        if "stderr" not in str(key) and isinstance(val, (int, float)):
            raw_val = float(val)
            return round(raw_val * 100.0 if raw_val <= 1.0 else raw_val, 2)

    raise ValueError(f"Could not locate numeric metric '{metric_hint}' in lm_eval output: {task_metrics}")


def _first_resp_text(resps: Any) -> str:
    """Unwrap nested `lm_eval` `sample['resps']` lists/tuples and return the first generation string."""
    cur = resps
    while isinstance(cur, (list, tuple)) and len(cur) > 0:
        cur = cur[0]
    return str(cur or "")


def _numbers_match(pred_num: str, gold_num: str) -> bool:
    """Check exact or float-equivalent equality between two extracted numeric strings."""
    p = str(pred_num or "").strip().replace(",", "")
    g = str(gold_num or "").strip().replace(",", "")
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def extract_lm_eval_detailed_metrics(
    results_payload: Dict[str, Any],
    task_name: str,
    metric_hint: str = "acc_norm",
    model_key_prefix: str = "e4b",
) -> Dict[str, Any]:
    """Extract unmodified `eval_score_pct`, `top1_acc_pct`, `nll` (`GoldNLL`), and `n_samples` from `lm_eval`."""
    _ = model_key_prefix
    eval_score_pct = parse_lm_eval_results_json(
        results_payload=results_payload,
        task_name=task_name,
        metric_hint=metric_hint,
    )
    try:
        top1_acc_pct = parse_lm_eval_results_json(
            results_payload=results_payload,
            task_name=task_name,
            metric_hint="acc",
        )
    except Exception:
        top1_acc_pct = eval_score_pct

    samples_block = results_payload.get("samples", {})
    gold_nlls: List[float] = []
    n_samples = 0

    if isinstance(samples_block, dict) and samples_block:
        for _subtask, sample_list in samples_block.items():
            if not isinstance(sample_list, list):
                continue
            seen_doc_ids = set()
            for sample in sample_list:
                if not isinstance(sample, dict):
                    continue
                doc_id = sample.get("doc_id")
                if doc_id is not None:
                    if doc_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(doc_id)
                n_samples += 1
                target = sample.get("target")
                resps = sample.get("resps")
                target_idx: Optional[int] = None
                if isinstance(target, int):
                    target_idx = target
                elif isinstance(target, str):
                    t_clean = target.strip().strip("()[]{}.")
                    if t_clean.isdigit():
                        target_idx = int(t_clean)
                    elif len(t_clean) == 1 and "A" <= t_clean.upper() <= "J":
                        target_idx = ord(t_clean.upper()) - ord("A")
                if target_idx is not None and isinstance(resps, list) and 0 <= target_idx < len(resps):
                    choice_resp = resps[target_idx]
                    while isinstance(choice_resp, (list, tuple)) and len(choice_resp) > 0:
                        if isinstance(choice_resp[0], (int, float)):
                            ll = float(choice_resp[0])
                            if math.isfinite(ll):
                                gold_nlls.append(max(0.0, -ll))
                            break
                        choice_resp = choice_resp[0]

    if n_samples == 0:
        n_samples_block = results_payload.get("n-samples", {})
        if isinstance(n_samples_block, dict):
            for v in n_samples_block.values():
                if isinstance(v, dict) and "effective" in v:
                    n_samples += int(v["effective"])
                elif isinstance(v, int):
                    n_samples += v

    mean_nll: Optional[float] = (
        round(sum(gold_nlls) / len(gold_nlls), 4) if gold_nlls else None
    )
    return {
        "eval_score_pct": eval_score_pct,
        "raw_signal_pct": eval_score_pct,
        "top1_acc_pct": top1_acc_pct,
        "nll": mean_nll,
        "ppl": round(math.exp(min(mean_nll, 12.0)), 2) if mean_nll is not None else None,
        "n_samples": n_samples,
    }


def parse_lighteval_results_json(
    lighteval_payload: Dict[str, Any],
    suite_name: str,
) -> Dict[str, Any]:
    """Extract percentage score (`0.0` to `100.0`) and `n_samples` for OALL v2 Arabic & LiveCodeBench-v6 from `lighteval`."""
    results_block = lighteval_payload.get("results", {})
    if not isinstance(results_block, dict) or not results_block:
        raise ValueError(f"LightEval payload has no 'results' entries for '{suite_name}'.")

    prefix_map = {
        "arabicmmlu": "arabic_mmlu",
        "madinahqa": "madinah_qa",
        "alghafa": "alghafa",
        "acva": "acva",
        "exams-ar": "arabic_exams",
        "aratrust": "aratrust",
        "livecodebench-v6": "lcb:codegeneration_v6",
        "livecodebench": "lcb:codegeneration_v6",
    }
    prefix = prefix_map.get(suite_name.strip().lower(), suite_name.strip().lower())
    avg_candidates = (
        f"community|{prefix}:_average|0",
        f"community:{prefix}:_average:0",
        f"community|{prefix}:_average|5",
        f"community:{prefix}:_average:5",
        f"lighteval|{prefix}:_average|0",
        f"lighteval:{prefix}:_average:0",
        f"lighteval|{prefix}:_average|5",
        f"lighteval:{prefix}:_average:5",
        f"custom|{prefix}:_average|0",
        f"custom|{prefix}:_average|5",
        f"{prefix}:_average|0",
        f"{prefix}:_average:0",
        f"{prefix}:_average|5",
        f"{prefix}:_average:5",
        f"community|{prefix}:All|5",
        f"community:{prefix}:All:5",
        f"lighteval|{prefix}:All|5",
        f"lighteval:{prefix}:All:5",
        f"{prefix}:All|5",
        f"{prefix}:All:5",
        f"community|{prefix}:All|0",
        f"community:{prefix}:All:0",
        f"custom|{prefix}:All|0",
        f"{prefix}:All|0",
        f"{prefix}:All:0",
        f"community|{prefix}|5",
        f"community:{prefix}:5",
        f"community|{prefix}|0",
        f"community:{prefix}:0",
        f"lighteval|{prefix}|0",
        f"lighteval:{prefix}:0",
        f"custom|{prefix}|0",
        f"community|{prefix}:default|5",
        f"community:{prefix}:default:5",
        f"community|{prefix}:default|0",
        f"community:{prefix}:default:0",
        f"{prefix}:default|0",
        f"{prefix}:default:0",
        f"{prefix}|5",
        f"{prefix}:5",
        f"{prefix}|0",
        f"{prefix}:0",
    )

    def _matches_prefix(key_str: str) -> bool:
        kl = str(key_str).lower()
        if kl == "all":
            return False
        return (
            kl.startswith(f"{prefix}:")
            or kl.startswith(f"{prefix}|")
            or kl == prefix
            or f"|{prefix}:" in kl
            or f"|{prefix}|" in kl
            or kl.endswith(f"|{prefix}")
            or f":{prefix}:" in kl
        )

    matched_metrics: List[Dict[str, Any]] = []
    for avg_key in avg_candidates:
        if avg_key in results_block and isinstance(results_block[avg_key], dict):
            matched_metrics = [results_block[avg_key]]
            break

    if not matched_metrics:
        for k, v in results_block.items():
            if _matches_prefix(k) and "_average" in str(k).lower() and isinstance(v, dict):
                matched_metrics = [v]
                break

    if not matched_metrics:
        for k, v in results_block.items():
            if _matches_prefix(k) and isinstance(v, dict):
                matched_metrics.append(v)

    if not matched_metrics:
        for k, v in results_block.items():
            if prefix in str(k).lower() and str(k).lower() != "all" and isinstance(v, dict):
                matched_metrics.append(v)

    if not matched_metrics:
        raise ValueError(f"Could not find '{prefix}' task scores in LightEval results: {list(results_block.keys())}")

    scores: List[float] = []
    primary_metric_keys = (
        "acc_norm",
        "loglikelihood_acc_norm",
        "codegen_pass@1:16",
        "codegen_pass@1:1",
        "codegen_pass@1",
        "pass@1",
        "acc",
        "loglikelihood_acc",
        "accuracy",
    )
    for mdict in matched_metrics:
        found = False
        for mkey in primary_metric_keys:
            if mkey in mdict and isinstance(mdict[mkey], (int, float)):
                val = float(mdict[mkey])
                scores.append(val * 100.0 if val <= 1.0 else val)
                found = True
                break
        if not found:
            for mkey, mval in mdict.items():
                mk_str = str(mkey)
                if (
                    (mk_str.startswith("acc") or "pass@1" in mk_str)
                    and not mk_str.endswith("_stderr")
                    and isinstance(mval, (int, float))
                ):
                    val = float(mval)
                    scores.append(val * 100.0 if val <= 1.0 else val)
                    found = True
                    break
        if not found:
            for mkey, mval in mdict.items():
                if "stderr" not in str(mkey) and isinstance(mval, (int, float)):
                    val = float(mval)
                    scores.append(val * 100.0 if val <= 1.0 else val)
                    break

    if not scores:
        raise ValueError(f"No numeric metric found for '{suite_name}' in LightEval results: {matched_metrics}")

    doc_counts = lighteval_payload.get("doc_counts", {})
    n_samples = 0
    if isinstance(doc_counts, dict) and doc_counts:
        for dk, dv in doc_counts.items():
            if isinstance(dv, int) and (_matches_prefix(dk) or prefix in str(dk).lower()):
                n_samples += int(dv)

    mean_score = round(sum(scores) / len(scores), 2)
    return {
        "eval_score_pct": mean_score,
        "raw_signal_pct": mean_score,
        "top1_acc_pct": mean_score,
        "nll": None,
        "ppl": None,
        "n_samples": n_samples,
    }


def _destroy_vllm_and_free_vram(lm_obj: Any = None, model_obj: Any = None) -> None:
    """Shut down any active vLLM engine / distributed group and release GPU VRAM before starting the next runner."""
    try:
        if lm_obj is not None:
            inner_vllm = getattr(lm_obj, "model", None)
            if inner_vllm is not None:
                try:
                    inner_vllm.llm_engine.engine_core.shutdown()
                except Exception:
                    pass
                del inner_vllm
            del lm_obj
    except Exception:
        pass
    try:
        if model_obj is not None:
            if hasattr(model_obj, "llm_engine"):
                try:
                    model_obj.llm_engine.engine_core.shutdown()
                except Exception:
                    pass
            del model_obj
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        import ray  # type: ignore

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_lighteval_tasks_subprocess(
    model_path: str,
    suite_names: List[str],
    device: str = "cuda:0",
    limit: Optional[int] = None,
    max_seq_len: int = 8192,
    batch_size: int = 16,
    use_vllm: bool = True,
    tag: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Run OALL v2 Arabic MCQ suites via `lighteval` in an isolated subprocess and return parsed metrics."""
    if not suite_names:
        return {}

    task_strings = resolve_lighteval_arabic_tasks(suite_names, limit=limit)
    if not task_strings:
        return {}

    model_dir = pathlib.Path(model_path)
    load_target_path = model_path
    if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        load_target_path = str(adapter_cfg.get("base_model_name_or_path") or model_path).strip()
    if pathlib.Path(load_target_path).is_dir():
        _patch_local_checkpoint_config_json(pathlib.Path(load_target_path))

    per_subtask_samples: Optional[int] = None
    if limit is not None and limit > 0:
        per_subtask_samples = max(10, int(math.ceil(float(limit) / 2.0)))

    tasks_csv = ",".join(task_strings)
    logger.info(
        "[%s] Running LightEval Arabic OALL v2 on %s (%d tasks, backend=%s, max_samples=%s)...",
        tag,
        ", ".join(suite_names),
        len(task_strings),
        "vllm" if use_vllm else "accelerate",
        str(per_subtask_samples),
    )

    child_env = os.environ.copy()
    if ":" in device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_idx = device.split(":")[1]
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    le_batch_size = (
        1
        if ("26b" in str(load_target_path).lower() or "26b" in str(tag).lower())
        else min(int(batch_size), 8)
    )
    le_max_len = min(int(max_seq_len), 4096)

    with tempfile.TemporaryDirectory(prefix="lighteval_ar_run_") as tmp_dir:
        out_json_file = pathlib.Path(tmp_dir) / "lighteval_results.json"
        runner_code = f"""
import json, logging, os, pathlib, sys
for _n in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
sys.path.insert(0, {repr(str(PROJECT_ROOT))})
from scripts.run_benchmarks import _destroy_vllm_and_free_vram, _prepare_gemma4_config_for_load
try:
    _prepare_gemma4_config_for_load({repr(load_target_path)})
except Exception:
    pass

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

evaluation_tracker = EvaluationTracker(
    output_dir={repr(str(tmp_dir))},
    save_details=False,
    push_to_hub=False,
)

use_vllm = {repr(bool(use_vllm))}
results_payload = None
doc_counts = {{}}

try:
    if use_vllm:
        try:
            from lighteval.models.vllm.vllm_model import VLLMModelConfig
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.VLLM,
                custom_tasks_directory="lighteval.tasks.multilingual.tasks.arabic",
                load_tasks_multilingual=False,
                max_samples={repr(per_subtask_samples)},
                bootstrap_iters=0,
            )
            model_config = VLLMModelConfig(
                model_name={repr(load_target_path)},
                dtype="bfloat16",
                gpu_memory_utilization=0.88,
                max_model_length={int(le_max_len)},
                max_num_batched_tokens={max(4096, int(le_max_len))},
                override_chat_template=False,
                trust_remote_code=True,
            )
            pipeline = Pipeline(
                tasks={repr(tasks_csv)},
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model_config=model_config,
            )
            doc_counts = {{
                str(k): len(v)
                for k, v in getattr(pipeline, "documents_dict", {{}}).items()
                if isinstance(v, (list, tuple))
            }}
            pipeline.evaluate()
            results_payload = pipeline.get_results()
        except Exception as vllm_exc:
            sys.stderr.write(f"[LightEval] vLLM backend failed ({{vllm_exc}}); falling back to Accelerate...\\n")
            _destroy_vllm_and_free_vram()

    if results_payload is None:
        from lighteval.models.transformers.transformers_model import TransformersModelConfig
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            custom_tasks_directory="lighteval.tasks.multilingual.tasks.arabic",
            load_tasks_multilingual=False,
            max_samples={repr(per_subtask_samples)},
            bootstrap_iters=0,
        )
        model_config = TransformersModelConfig(
            model_name={repr(load_target_path)},
            dtype="bfloat16",
            batch_size={int(le_batch_size)},
            max_length={int(le_max_len)},
            override_chat_template=False,
            trust_remote_code=True,
        )
        pipeline = Pipeline(
            tasks={repr(tasks_csv)},
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )
        doc_counts = {{
            str(k): len(v)
            for k, v in getattr(pipeline, "documents_dict", {{}}).items()
            if isinstance(v, (list, tuple))
        }}
        pipeline.evaluate()
        results_payload = pipeline.get_results()

    pathlib.Path({repr(str(out_json_file))}).write_text(
        json.dumps(
            {{
                "results": results_payload.get("results", {{}}),
                "doc_counts": doc_counts,
            }},
            indent=2,
        ),
        encoding="utf-8",
    )
finally:
    _destroy_vllm_and_free_vram()
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0 or not out_json_file.exists():
            raise RuntimeError(
                f"[{tag}] LightEval Arabic subprocess failed with exit code {proc.returncode}"
            )
        raw_lighteval = json.loads(out_json_file.read_text(encoding="utf-8"))

    parsed_out: Dict[str, Dict[str, Any]] = {}
    for sname in suite_names:
        metrics = parse_lighteval_results_json(raw_lighteval, sname)
        if metrics.get("n_samples", 0) == 0 and limit is not None:
            metrics["n_samples"] = int(limit)
        parsed_out[sname] = metrics
        logger.info(
            "[%s] arabic/%s (lighteval) -> acc_norm=%.2f%% (N=%d)",
            tag,
            sname,
            metrics["eval_score_pct"],
            metrics["n_samples"],
        )
    return parsed_out


def run_inspect_evals_subprocess(
    model_path: str,
    suite_specs: Dict[str, Dict[str, Any]],
    device: str = "cuda:0",
    limit: Optional[int] = None,
    use_vllm: bool = True,
    tag: str = "",
    on_suite_complete: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run `GPQA-Diamond` (`epochs=1`) and `AIME-2026` (`epochs=4`, no tools) via `inspect_evals` in an isolated subprocess."""
    if not suite_specs:
        return {}

    model_dir = pathlib.Path(model_path)
    load_target_path = model_path
    if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        load_target_path = str(adapter_cfg.get("base_model_name_or_path") or model_path).strip()
    if pathlib.Path(load_target_path).is_dir():
        resolved_dir = pathlib.Path(load_target_path).resolve()
        _patch_local_checkpoint_config_json(resolved_dir)
        try:
            load_target_path = resolved_dir.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            load_target_path = resolved_dir.as_posix()

    is_26b = "26b" in str(load_target_path).lower() or "26b" in str(tag).lower()
    inspect_max_model_len = 24576 if is_26b else 32768
    inspect_max_tokens = 18432 if is_26b else 24576
    inspect_max_conns = 32 if is_26b else 64

    logger.info(
        "[%s] Running Inspect Evals on %s (backend=%s, max_model_len=%d, max_tokens=%d, limit=%s)...",
        tag,
        ", ".join(suite_specs.keys()),
        "vllm" if use_vllm else "hf",
        inspect_max_model_len,
        inspect_max_tokens,
        str(limit),
    )

    child_env = os.environ.copy()
    if ":" in device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_idx = device.split(":")[1]
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    child_env.setdefault("INSPECT_DISPLAY", "plain")

    with tempfile.TemporaryDirectory(prefix="inspect_eval_run_") as tmp_dir:
        specs_json_file = pathlib.Path(tmp_dir) / "inspect_specs.json"
        out_json_file = pathlib.Path(tmp_dir) / "inspect_results.json"
        inspect_log_dir = pathlib.Path(tmp_dir) / "logs"
        specs_json_file.write_text(json.dumps(suite_specs), encoding="utf-8")

        runner_code = f"""
import inspect, json, logging, os, pathlib, sys
for _n in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
sys.path.insert(0, {repr(str(PROJECT_ROOT))})
from scripts.run_benchmarks import (
    _destroy_vllm_and_free_vram,
    _prepare_gemma4_config_for_load,
    clean_gemma4_completion,
)
try:
    _prepare_gemma4_config_for_load({repr(load_target_path)})
except Exception:
    pass

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver

try:
    from inspect_ai.model._providers._vllm_lora import cleanup_servers
except Exception:
    def cleanup_servers():
        pass

@solver
def _gemma4_clean_solver(inner_solver: Solver) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async def wrapped_generate(s: TaskState, *args, **kwargs) -> TaskState:
            s = await generate(s, *args, **kwargs)
            if s.output and s.output.completion:
                raw_c = s.output.completion
                if "<channel|>" in raw_c or "<|channel>" in raw_c or "<turn|>" in raw_c:
                    cleaned = clean_gemma4_completion(raw_c, drop_unclosed_thought=True)
                    s.output.completion = cleaned
                    if getattr(s, "messages", None):
                        try:
                            s.messages[-1].content = cleaned
                        except Exception:
                            pass
            return s
        return await inner_solver(state, wrapped_generate)
    return solve

suite_specs = json.loads(pathlib.Path({repr(str(specs_json_file))}).read_text(encoding="utf-8"))
use_vllm = {repr(bool(use_vllm))}
limit_val = {repr(limit)}
out_results = {{}}

has_gemma4_parser = False
if use_vllm:
    try:
        from vllm.reasoning import ReasoningParserManager
        if "gemma4" in getattr(ReasoningParserManager, "reasoning_parsers", {{}}):
            has_gemma4_parser = True
        elif hasattr(ReasoningParserManager, "get_reasoning_parser"):
            ReasoningParserManager.get_reasoning_parser("gemma4")
            has_gemma4_parser = True
    except Exception:
        has_gemma4_parser = False

try:
    if use_vllm:
        vllm_args = {{
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.88,
            "max_model_len": {int(inspect_max_model_len)},
            "trust_remote_code": True,
            "timeout": 1200,
        }}
        if has_gemma4_parser:
            vllm_args["reasoning_parser"] = "gemma4"
        extra_body = {{
            "top_k": 64,
            "chat_template_kwargs": {{"enable_thinking": True}},
        }}
        if not has_gemma4_parser:
            extra_body["skip_special_tokens"] = False
        gen_cfg = GenerateConfig(
            max_tokens={int(inspect_max_tokens)},
            temperature=1.0,
            top_p=0.95,
            max_connections={int(inspect_max_conns)},
            extra_body=extra_body,
        )
        model_inst = get_model(
            {repr(f"vllm/{load_target_path}")},
            config=gen_cfg,
            **vllm_args,
        )
    else:
        gen_cfg = GenerateConfig(
            max_tokens={int(inspect_max_tokens)},
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        model_inst = get_model(
            {repr(f"hf/{load_target_path}")},
            config=gen_cfg,
            dtype="bfloat16",
            trust_remote_code=True,
        )

    eval_sig_params = inspect.signature(inspect_eval).parameters

    for sname, spec in suite_specs.items():
        task_id = str(spec.get("lm_eval_task", "")).strip().lower()
        epochs = int(spec.get("epochs", 1))
        if "gpqa" in task_id or sname.strip().lower().startswith("gpqa"):
            from inspect_evals.gpqa import gpqa_diamond
            task_obj = gpqa_diamond(cot=True)
            task_obj.solver = _gemma4_clean_solver(
                chain(task_obj.solver) if isinstance(task_obj.solver, list) else task_obj.solver
            )
        elif "aime" in task_id or sname.strip().lower().startswith("aime"):
            from inspect_evals.aime2026 import aime2026
            task_obj = aime2026()
            task_obj.solver = _gemma4_clean_solver(
                chain(task_obj.solver) if isinstance(task_obj.solver, list) else task_obj.solver
            )
        else:
            raise ValueError(f"Unsupported inspect_evals task for suite '{{sname}}': {{task_id}}")

        eval_kwargs = {{
            "tasks": [task_obj],
            "model": model_inst,
            "epochs": epochs,
            "limit": limit_val,
            "log_dir": {repr(str(inspect_log_dir))},
            "fail_on_error": True,
        }}
        if "display" in eval_sig_params:
            eval_kwargs["display"] = "plain"

        logs = inspect_eval(**eval_kwargs)
        if not logs:
            raise RuntimeError(f"inspect_ai.eval returned no logs for '{{sname}}'")
        log = logs[0]
        if getattr(log, "status", "") != "success" or getattr(log, "results", None) is None:
            raise RuntimeError(
                f"inspect_ai.eval failed for '{{sname}}': status={{getattr(log, 'status', None)}}, error={{getattr(log, 'error', None)}}"
            )
        scores_list = getattr(log.results, "scores", None) or []
        if not scores_list:
            raise RuntimeError(f"inspect_ai.eval produced empty scores list for '{{sname}}'")

        acc_val = None
        for score_obj in scores_list:
            metrics_map = getattr(score_obj, "metrics", {{}}) or {{}}
            for mkey in ("accuracy", "mean", "pass_at_1"):
                if mkey in metrics_map and getattr(metrics_map[mkey], "value", None) is not None:
                    acc_val = float(metrics_map[mkey].value)
                    break
            if acc_val is None:
                for mname, mobj in metrics_map.items():
                    if "stderr" not in str(mname) and getattr(mobj, "value", None) is not None:
                        acc_val = float(mobj.value)
                        break
            if acc_val is not None:
                break

        if acc_val is None:
            raise RuntimeError(f"Could not extract numeric accuracy from inspect_ai log for '{{sname}}'")

        pct = round(acc_val * 100.0 if acc_val <= 1.0 else acc_val, 2)
        total_samples = int(
            getattr(log.results, "total_samples", 0)
            or len(getattr(log, "samples", []) or [])
        )
        out_results[sname] = {{
            "eval_score_pct": pct,
            "raw_signal_pct": pct,
            "top1_acc_pct": pct,
            "nll": None,
            "ppl": None,
            "n_samples": total_samples,
        }}
        pathlib.Path({repr(str(out_json_file))}).write_text(
            json.dumps(out_results, indent=2),
            encoding="utf-8",
        )
finally:
    try:
        cleanup_servers()
    except Exception:
        pass
    _destroy_vllm_and_free_vram()
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
            cwd=str(PROJECT_ROOT),
        )
        partial_inspect: Dict[str, Dict[str, Any]] = {}
        if out_json_file.exists():
            try:
                partial_inspect = json.loads(out_json_file.read_text(encoding="utf-8"))
            except Exception:
                partial_inspect = {}
        if on_suite_complete is not None and partial_inspect:
            for sname, smetrics in partial_inspect.items():
                on_suite_complete(sname, smetrics)
        if proc.returncode != 0 or len(partial_inspect) < len(suite_specs):
            raise RuntimeError(
                f"[{tag}] inspect_evals subprocess failed (exit={proc.returncode}, completed={list(partial_inspect.keys())})"
            )

    for sname, metrics in partial_inspect.items():
        logger.info(
            "[%s] retention/%s (inspect_evals) -> accuracy=%.2f%% (N=%d)",
            tag,
            sname,
            metrics["eval_score_pct"],
            metrics["n_samples"],
        )
    return partial_inspect


def run_lighteval_lcb_subprocess(
    model_path: str,
    suite_specs: Dict[str, Dict[str, Any]],
    device: str = "cuda:0",
    limit: Optional[int] = None,
    use_vllm: bool = True,
    tag: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Run `LiveCodeBench-v6` (`lcb:codegeneration_v6|0`, `num_samples=1`) via `lighteval@main` in an isolated subprocess."""
    if not suite_specs:
        return {}

    model_dir = pathlib.Path(model_path)
    load_target_path = model_path
    if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        load_target_path = str(adapter_cfg.get("base_model_name_or_path") or model_path).strip()
    if pathlib.Path(load_target_path).is_dir():
        _patch_local_checkpoint_config_json(pathlib.Path(load_target_path))

    is_26b = "26b" in str(load_target_path).lower() or "26b" in str(tag).lower()
    lcb_max_model_len = 28672 if is_26b else 32768
    lcb_max_new_tokens = 18432 if is_26b else 24576
    lcb_max_num_seqs = 32 if is_26b else 64

    first_spec = next(iter(suite_specs.values()))
    lcb_task_str = str(first_spec.get("lighteval_task") or "lcb:codegeneration_v6|0")
    num_samples = int(first_spec.get("num_samples", 1))
    enable_thinking = bool(first_spec.get("enable_thinking", True))

    logger.info(
        "[%s] Running LightEval LiveCodeBench-v6 (%s, num_samples=%d, max_model_len=%d, max_new_tokens=%d, limit=%s)...",
        tag,
        lcb_task_str,
        num_samples,
        lcb_max_model_len,
        lcb_max_new_tokens,
        str(limit),
    )

    child_env = os.environ.copy()
    if ":" in device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_idx = device.split(":")[1]
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with tempfile.TemporaryDirectory(prefix="lighteval_lcb_run_") as tmp_dir:
        out_json_file = pathlib.Path(tmp_dir) / "lighteval_lcb_results.json"
        runner_code = f"""
import json, logging, os, pathlib, sys
for _n in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
sys.path.insert(0, {repr(str(PROJECT_ROOT))})
from scripts.run_benchmarks import (
    _destroy_vllm_and_free_vram,
    _prepare_gemma4_config_for_load,
    clean_gemma4_completion,
)
try:
    _prepare_gemma4_config_for_load({repr(load_target_path)})
except Exception:
    pass

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

enable_thinking = {repr(bool(enable_thinking))}
num_samples = {int(num_samples)}
lcb_max_model_len = {int(lcb_max_model_len)}
lcb_max_new_tokens = {int(lcb_max_new_tokens)}

evaluation_tracker = EvaluationTracker(
    output_dir={repr(str(tmp_dir))},
    save_details=False,
    push_to_hub=False,
)

results_payload = None
doc_counts = {{}}

try:
    if {repr(bool(use_vllm))}:
        from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig

        class Gemma4LCBVLLMModel(VLLMModel):
            DATASET_SPLITS = 1

            def _create_auto_tokenizer(self, *args, **kwargs):
                tok = super()._create_auto_tokenizer(*args, **kwargs)
                orig_apply = tok.apply_chat_template

                def _apply_with_thinking(conversation, *a, **kw):
                    kw.setdefault("enable_thinking", bool(enable_thinking))
                    return orig_apply(conversation, *a, **kw)

                tok.apply_chat_template = _apply_with_thinking
                return tok

            def _generate(self, *args, **kwargs):
                inputs = args[0] if len(args) > 0 else kwargs.get("inputs")
                orig_cfg_max_new = self.config.generation_parameters.max_new_tokens
                try:
                    if inputs and isinstance(inputs, list) and isinstance(inputs[0], (list, tuple)):
                        max_prompt_len = max(len(x) for x in inputs)
                        eff_max_model = int(self._max_length or lcb_max_model_len)
                        safe_budget = max(512, eff_max_model - max_prompt_len - 32)
                        if orig_cfg_max_new is not None and orig_cfg_max_new > safe_budget:
                            self.config.generation_parameters.max_new_tokens = safe_budget
                    outputs = super()._generate(*args, **kwargs)
                finally:
                    self.config.generation_parameters.max_new_tokens = orig_cfg_max_new
                if outputs:
                    for req_out in outputs:
                        for comp in getattr(req_out, "outputs", ()):
                            token_ids = getattr(comp, "token_ids", None)
                            if token_ids is not None:
                                raw_txt = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                            else:
                                raw_txt = str(getattr(comp, "text", "") or "")
                            comp.text = clean_gemma4_completion(raw_txt, drop_unclosed_thought=True)
                return outputs

        gen_params = GenerationParameters(
            max_new_tokens=lcb_max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        try:
            vllm_cfg = VLLMModelConfig(
                model_name={repr(load_target_path)},
                dtype="bfloat16",
                gpu_memory_utilization=0.88,
                max_model_length=lcb_max_model_len,
                max_num_seqs={int(lcb_max_num_seqs)},
                max_num_batched_tokens=8192,
                override_chat_template=True,
                add_special_tokens=False,
                trust_remote_code=True,
                generation_parameters=gen_params,
            )
            vllm_model = Gemma4LCBVLLMModel(vllm_cfg)
        except Exception:
            _destroy_vllm_and_free_vram()
            lcb_max_model_len = 24576
            gen_params = GenerationParameters(
                max_new_tokens=16384,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
            vllm_cfg = VLLMModelConfig(
                model_name={repr(load_target_path)},
                dtype="bfloat16",
                gpu_memory_utilization=0.88,
                max_model_length=lcb_max_model_len,
                max_num_seqs=max(16, {int(lcb_max_num_seqs)} // 2),
                max_num_batched_tokens=4096,
                override_chat_template=True,
                add_special_tokens=False,
                trust_remote_code=True,
                generation_parameters=gen_params,
            )
            vllm_model = Gemma4LCBVLLMModel(vllm_cfg)

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.VLLM,
            load_tasks_multilingual=False,
            max_samples={repr(limit)},
            remove_reasoning_tags=False,
            bootstrap_iters=0,
        )
        pipeline = Pipeline(
            tasks={repr(lcb_task_str)},
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model=vllm_model,
        )
    else:
        from lighteval.models.transformers.transformers_model import TransformersModelConfig
        gen_params = GenerationParameters(
            max_new_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        tf_cfg = TransformersModelConfig(
            model_name={repr(load_target_path)},
            dtype="bfloat16",
            batch_size=1,
            max_length=8192,
            override_chat_template=True,
            trust_remote_code=True,
            generation_parameters=gen_params,
        )
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            load_tasks_multilingual=False,
            max_samples={repr(limit)},
            remove_reasoning_tags=True,
            reasoning_tags=[("<|channel>", "<channel|>")],
            bootstrap_iters=0,
        )
        pipeline = Pipeline(
            tasks={repr(lcb_task_str)},
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=tf_cfg,
        )

    for _t in getattr(pipeline, "tasks_dict", {{}}).values():
        if hasattr(_t, "num_samples"):
            _t.num_samples = [num_samples]
    for _doc_list in getattr(pipeline, "documents_dict", {{}}).values():
        if isinstance(_doc_list, (list, tuple)):
            for _d in _doc_list:
                if hasattr(_d, "num_samples"):
                    _d.num_samples = num_samples

    doc_counts = {{
        str(k): len(v)
        for k, v in getattr(pipeline, "documents_dict", {{}}).items()
        if isinstance(v, (list, tuple))
    }}
    pipeline.evaluate()
    results_payload = pipeline.get_results()
    pathlib.Path({repr(str(out_json_file))}).write_text(
        json.dumps(
            {{
                "results": results_payload.get("results", {{}}),
                "doc_counts": doc_counts,
            }},
            indent=2,
        ),
        encoding="utf-8",
    )
finally:
    _destroy_vllm_and_free_vram()
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0 or not out_json_file.exists():
            raise RuntimeError(
                f"[{tag}] LightEval LiveCodeBench-v6 subprocess failed with exit code {proc.returncode}"
            )
        raw_lcb = json.loads(out_json_file.read_text(encoding="utf-8"))

    parsed_out: Dict[str, Dict[str, Any]] = {}
    for sname in suite_specs:
        metrics = parse_lighteval_results_json(raw_lcb, sname)
        if metrics.get("n_samples", 0) == 0 and limit is not None:
            metrics["n_samples"] = int(limit)
        parsed_out[sname] = metrics
        logger.info(
            "[%s] retention/%s (lighteval_lcb) -> pass@1=%.2f%% (N=%d)",
            tag,
            sname,
            metrics["eval_score_pct"],
            metrics["n_samples"],
        )
    return parsed_out


def _get_lm_eval_gen_kwargs(
    task_name: str,
    model_key_prefix: str = "e4b",
    use_vllm: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return official Gemma 4 Thinking Mode `generation_kwargs` for `lm_eval` generative tasks (`mmlu_pro`)."""
    _ = model_key_prefix
    clean = task_name.strip().lower()
    if clean == "mmlu_pro":
        kw: Dict[str, Any] = {
            "max_gen_toks": 16384,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "do_sample": True,
            "until": [
                "<turn|>",
                "<end_of_turn>",
                "<eos>",
                "</s>",
                "<|im_end|>",
            ],
        }
        if use_vllm:
            kw["skip_special_tokens"] = False
        return kw
    return None


def _should_apply_chat_template_for_task(
    task_name: str,
    apply_chat_template: bool,
    model_key_prefix: str = "e4b",
) -> bool:
    """Return `True` (when `apply_chat_template` is enabled) for chat-instruct tasks (`mmlu_pro`) and `False` for completion/log-likelihood tasks."""
    _ = model_key_prefix
    clean = task_name.strip().lower()
    if clean in (
        "arc_challenge",
        "arc_easy",
        "humaneval",
        "gsm8k",
        "mgsm_ar",
        "arabicmmlu",
        "madinahqa",
        "aratrust",
        "arabic_exams",
        "arabic_leaderboard_alghafa_light",
        "arabic_leaderboard_acva_light",
    ):
        return False
    return bool(apply_chat_template)


def _configure_lm_eval_thinking_for_task(
    lm_obj: Any,
    effective_use_vllm: bool,
    task_chat_tpl: bool,
    task_name: str = "",
    model_key_prefix: str = "e4b",
) -> None:
    """Configure `enable_thinking=True` and `think_end_token='<channel|>'` for `mmlu_pro` across `VLLM` and `HFLM`."""
    use_thinking = bool(task_chat_tpl) and (task_name.strip().lower() == "mmlu_pro")
    if model_key_prefix == "26b_a4b":
        tok = getattr(lm_obj, "tokenizer", None)
        if tok is not None:
            try:
                if task_chat_tpl:
                    tok.eos_token = "<turn|>"
                    tok.eos_token_id = 106
                else:
                    tok.eos_token = "<eos>"
                    tok.eos_token_id = 1
            except Exception:
                pass
    if effective_use_vllm:
        if hasattr(lm_obj, "enable_thinking"):
            lm_obj.enable_thinking = use_thinking
        if hasattr(lm_obj, "think_end_token"):
            lm_obj.think_end_token = "<channel|>" if use_thinking else None
        if hasattr(lm_obj, "chat_template_args"):
            lm_obj.chat_template_args = {}
    else:
        if hasattr(lm_obj, "enable_thinking"):
            lm_obj.enable_thinking = use_thinking
        if hasattr(lm_obj, "think_end_token"):
            lm_obj.think_end_token = "<channel|>" if use_thinking else None
        if hasattr(lm_obj, "chat_template_args"):
            lm_obj.chat_template_args = {"enable_thinking": True} if use_thinking else {}


def run_lm_eval_task(
    model_path: str,
    task_name: str,
    num_fewshot: int,
    batch_size: int = 8,
    device: str = "cuda:0",
    limit: Optional[int] = None,
    metric_hint: str = "acc_norm",
    use_vllm: bool = False,
    apply_chat_template: bool = True,
) -> float:
    """Run a single `lm_eval` task via `lm_eval.simple_evaluate` and return the score."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    tasks_arg, primary_key, eff_limit = resolve_lm_eval_task_specs(
        task_name=task_name,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    if pathlib.Path(model_path).is_dir():
        try:
            _prepare_gemma4_config_for_load(model_path)
        except Exception:
            pass
    is_26b = "26b" in str(model_path).lower()
    lm_max_len = 24576 if is_26b else 32768
    task_chat_tpl = _should_apply_chat_template_for_task(task_name, apply_chat_template)
    use_thinking = bool(task_chat_tpl) and (task_name.strip().lower() == "mmlu_pro")
    model_backend = "vllm" if use_vllm else "hf"
    if use_vllm:
        model_args: Any = {
            "pretrained": model_path,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.88,
            "max_model_len": lm_max_len,
            "enable_thinking": use_thinking,
            "think_end_token": "<channel|>" if use_thinking else None,
        }
    else:
        model_args = {
            "pretrained": model_path,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "tie_word_embeddings": True,
            "enable_thinking": use_thinking,
            "think_end_token": "<channel|>" if use_thinking else None,
            "chat_template_args": {"enable_thinking": True} if use_thinking else {},
        }
    gen_kwargs = _get_lm_eval_gen_kwargs(task_name, use_vllm=use_vllm)
    payload = lm_eval.simple_evaluate(
        model=model_backend,
        model_args=model_args,
        tasks=tasks_arg,
        num_fewshot=num_fewshot,
        batch_size="auto" if use_vllm else batch_size,
        device=device,
        limit=eff_limit,
        gen_kwargs=gen_kwargs,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        confirm_run_unsafe_code=True,
        apply_chat_template=task_chat_tpl,
        fewshot_as_multiturn=bool(task_chat_tpl),
        log_samples=True,
        bootstrap_iters=0,
        task_manager=TaskManager(),
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"lm_eval.simple_evaluate returned empty payload for task '{task_name}'.")
    return extract_lm_eval_detailed_metrics(
        results_payload=payload,
        task_name=primary_key,
        metric_hint=metric_hint,
    )["eval_score_pct"]


def _run_lm_eval_suites_subprocess(
    model_path: str,
    device: str,
    cfg: Dict[str, Any],
    tag: str,
    limit: Optional[int] = None,
    use_vllm: bool = True,
    apply_chat_template: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the `lm_eval` tasks for `model_path` in an isolated child subprocess so the OS reclaims 100% of vLLM's GPU memory pool upon completion."""
    child_env = os.environ.copy()
    if ":" in device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_idx = device.split(":")[1]
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    child_env.setdefault("HF_ALLOW_CODE_EVAL", "1")

    with tempfile.TemporaryDirectory(prefix="lm_eval_run_") as tmp_dir:
        cfg_json_file = pathlib.Path(tmp_dir) / "eval_cfg.json"
        out_json_file = pathlib.Path(tmp_dir) / "lm_eval_out.json"
        cfg_json_file.write_text(json.dumps(cfg), encoding="utf-8")

        runner_code = f"""
import json, logging, pathlib, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
for _n in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
sys.path.insert(0, {repr(str(PROJECT_ROOT))})
from scripts.run_benchmarks import _evaluate_checkpoint_with_lm_eval

cfg = json.loads(pathlib.Path({repr(str(cfg_json_file))}).read_text(encoding="utf-8"))
res = _evaluate_checkpoint_with_lm_eval(
    model_path={repr(str(model_path))},
    device="cuda:0",
    cfg=cfg,
    tag={repr(str(tag))},
    limit={repr(limit)},
    use_vllm={repr(bool(use_vllm))},
    apply_chat_template={repr(bool(apply_chat_template))},
    _in_lm_eval_subprocess=True,
    _skip_non_lm_eval=True,
)
pathlib.Path({repr(str(out_json_file))}).write_text(
    json.dumps(res, indent=2),
    encoding="utf-8",
)
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0 or not out_json_file.exists():
            raise RuntimeError(
                f"[{tag}] lm_eval child subprocess failed with exit code {proc.returncode}"
            )
        return json.loads(out_json_file.read_text(encoding="utf-8"))


def _evaluate_checkpoint_with_lm_eval(
    model_path: str,
    device: str,
    cfg: Dict[str, Any],
    tag: str,
    limit: Optional[int] = None,
    use_vllm: bool = True,
    apply_chat_template: bool = True,
    _in_lm_eval_subprocess: bool = False,
    _skip_non_lm_eval: bool = False,
    _preserved_arabic: Optional[Dict[str, Dict[str, Any]]] = None,
    _preserved_retention: Optional[Dict[str, Dict[str, Any]]] = None,
    checkpoint_callback: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Evaluate a checkpoint across all 4 isolated runner stages (`lighteval` Arabic, `lm_eval` MMLU-Pro, `inspect_evals` GPQA+AIME, `lighteval_lcb` LiveCodeBench-v6)."""
    eval_cfg = cfg.get("evaluation", {})
    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})
    model_key_prefix = tag.split(":")[0] if ":" in tag else "e4b"
    model_specific_cfg = cfg.get("models", {}).get(model_key_prefix, {})
    default_bs = 8 if model_key_prefix == "26b_a4b" else int(eval_cfg.get("batch_size", 32))
    batch_size = int(model_specific_cfg.get("batch_size", default_bs))
    max_seq_len = int(eval_cfg.get("max_seq_length", 8192))
    lm_eval_max_len = 24576 if model_key_prefix == "26b_a4b" else 32768

    out: Dict[str, Dict[str, Dict[str, Any]]] = {
        "arabic": dict(_preserved_arabic or {}),
        "retention": dict(_preserved_retention or {}),
    }

    def _flush_stage() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(out)

    pending_lighteval_ar: List[str] = [
        sname
        for sname, spec in ar_defs.items()
        if sname not in out["arabic"]
        and str(spec.get("runner", "lm_eval")).strip().lower() == "lighteval"
        and sname in LIGHTEVAL_SUITE_TASK_MAP
    ]
    pending_lm_eval_ar = {
        k: v
        for k, v in ar_defs.items()
        if k not in out["arabic"]
        and str(v.get("runner", "lm_eval")).strip().lower() == "lm_eval"
    }
    pending_lm_eval_ret = {
        k: v
        for k, v in ret_defs.items()
        if k not in out["retention"]
        and str(v.get("runner", "lm_eval")).strip().lower() == "lm_eval"
    }
    pending_inspect_ret = {
        k: v
        for k, v in ret_defs.items()
        if k not in out["retention"]
        and str(v.get("runner", "lm_eval")).strip().lower() == "inspect_evals"
    }
    pending_lcb_ret = {
        k: v
        for k, v in ret_defs.items()
        if k not in out["retention"]
        and str(v.get("runner", "lm_eval")).strip().lower() == "lighteval_lcb"
    }

    # Stage 1: OALL v2 Arabic MCQ via LightEval (isolated subprocess)
    if pending_lighteval_ar and not _skip_non_lm_eval:
        le_metrics = run_lighteval_tasks_subprocess(
            model_path=model_path,
            suite_names=pending_lighteval_ar,
            device=device,
            limit=limit,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            use_vllm=use_vllm,
            tag=tag,
        )
        for sname, smetrics in le_metrics.items():
            out["arabic"][sname] = smetrics
        _flush_stage()

    # Stage 2: MMLU-Pro (and any configured lm_eval suites)
    has_pending_lm_eval = bool(pending_lm_eval_ar or pending_lm_eval_ret)
    if has_pending_lm_eval:
        if use_vllm and isinstance(model_path, str) and not _in_lm_eval_subprocess:
            sub_cfg = dict(cfg)
            sub_cfg["arabic_benchmarks"] = pending_lm_eval_ar
            sub_cfg["retention_anchors"] = pending_lm_eval_ret
            lm_sub_out = _run_lm_eval_suites_subprocess(
                model_path=model_path,
                device=device,
                cfg=sub_cfg,
                tag=tag,
                limit=limit,
                use_vllm=use_vllm,
                apply_chat_template=apply_chat_template,
            )
            for sname, smetrics in lm_sub_out.get("arabic", {}).items():
                out["arabic"][sname] = smetrics
            for sname, smetrics in lm_sub_out.get("retention", {}).items():
                out["retention"][sname] = smetrics
            _flush_stage()
        else:
            import lm_eval
            import torch
            from lm_eval.models.huggingface import HFLM
            from lm_eval.tasks import TaskManager
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )

            cache_dir = _resolve_hf_eval_cache_dir()
            if cache_dir:
                os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

            _prepare_gemma4_config_for_load(
                str(
                    json.loads((pathlib.Path(model_path) / "adapter_config.json").read_text(encoding="utf-8")).get(
                        "base_model_name_or_path", model_path
                    )
                )
                if (pathlib.Path(model_path) / "adapter_config.json").exists()
                and not (pathlib.Path(model_path) / "config.json").exists()
                else model_path
            )

            logger.info(
                "[%s] Initializing lm_eval model (%s) on %s: %s",
                tag,
                "vLLM" if use_vllm else f"HFLM (batch_size={batch_size})",
                device,
                model_path,
            )
            model_obj: Any = None
            lm_obj: Any = None
            effective_use_vllm = bool(use_vllm)

            if effective_use_vllm:
                try:
                    from lm_eval.models.vllm_causallms import VLLM

                    model_dir = pathlib.Path(model_path)
                    if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
                        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
                        base_model_path = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
                        lm_obj = VLLM(
                            pretrained=base_model_path,
                            lora_local_path=str(model_path),
                            dtype="bfloat16",
                            trust_remote_code=True,
                            gpu_memory_utilization=0.88,
                            max_model_len=lm_eval_max_len,
                            batch_size="auto",
                            enable_thinking=True,
                            think_end_token="<channel|>",
                        )
                    else:
                        lm_obj = VLLM(
                            pretrained=model_path,
                            dtype="bfloat16",
                            trust_remote_code=True,
                            gpu_memory_utilization=0.88,
                            max_model_len=lm_eval_max_len,
                            batch_size="auto",
                            enable_thinking=True,
                            think_end_token="<channel|>",
                        )
                except Exception as vllm_exc:
                    logger.warning(
                        "[%s] vLLM initialization failed (%s); falling back to HFLM (batch_size=%d).",
                        tag,
                        vllm_exc,
                        batch_size,
                    )
                    _destroy_vllm_and_free_vram()
                    effective_use_vllm = False

            if not effective_use_vllm:
                model_dir = pathlib.Path(model_path)
                is_peft_adapter = (model_dir / "adapter_config.json").exists() and not (
                    model_dir / "config.json"
                ).exists()
                load_target_path = model_path
                if is_peft_adapter:
                    adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
                    load_target_path = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
                    if not load_target_path:
                        raise ValueError(
                            f"Unmerged PEFT adapter directory '{model_path}' is missing 'base_model_name_or_path' in adapter_config.json"
                        )
                    logger.info(
                        "[%s] Detected unmerged PEFT adapter at %s -> loading base model %s",
                        tag,
                        model_path,
                        load_target_path,
                    )

                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(load_target_path, trust_remote_code=True)

                hf_config = _prepare_gemma4_config_for_load(load_target_path)
                try:
                    model_obj = AutoModelForImageTextToText.from_pretrained(
                        load_target_path,
                        config=hf_config,
                        torch_dtype=torch.bfloat16,
                        device_map={"": device},
                        trust_remote_code=True,
                    )
                except Exception:
                    model_obj = AutoModelForCausalLM.from_pretrained(
                        load_target_path,
                        config=hf_config,
                        torch_dtype=torch.bfloat16,
                        device_map={"": device},
                        trust_remote_code=True,
                    )
                _ensure_gemma4_tied_lm_head(model_obj, tag=tag)
                if is_peft_adapter:
                    from peft import PeftModel

                    model_obj = PeftModel.from_pretrained(model_obj, model_path)
                    _ensure_gemma4_tied_lm_head(model_obj, tag=f"{tag}:peft")
                _patch_gemma4_config_for_lm_eval(model_obj)
                model_obj.eval()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                lm_obj = HFLM(
                    pretrained=model_obj,
                    tokenizer=tokenizer,
                    backend="causal",
                    batch_size=batch_size,
                    max_length=lm_eval_max_len,
                    trust_remote_code=True,
                )
                lm_obj.chat_template_args = {}
                _ensure_gemma4_tied_lm_head(lm_obj, tag=f"{tag}:hflm")

            task_manager = TaskManager()
            suite_groups = (
                ("arabic", pending_lm_eval_ar),
                ("retention", pending_lm_eval_ret),
            )
            for cat, suite_defs in suite_groups:
                for suite_name, spec in suite_defs.items():
                    task_name = str(spec["lm_eval_task"])
                    num_fewshot = int(spec.get("num_fewshot", 0))
                    metric_hint = str(spec.get("metric", "acc_norm"))
                    tasks_arg, primary_key, eff_limit = resolve_lm_eval_task_specs(
                        task_name=task_name,
                        num_fewshot=num_fewshot,
                        limit=limit,
                    )
                    task_chat_tpl = _should_apply_chat_template_for_task(
                        task_name,
                        apply_chat_template,
                        model_key_prefix=model_key_prefix,
                    )
                    _configure_lm_eval_thinking_for_task(
                        lm_obj=lm_obj,
                        effective_use_vllm=effective_use_vllm,
                        task_chat_tpl=task_chat_tpl,
                        task_name=task_name,
                        model_key_prefix=model_key_prefix,
                    )
                    gen_kwargs = _get_lm_eval_gen_kwargs(
                        task_name,
                        model_key_prefix=model_key_prefix,
                        use_vllm=effective_use_vllm,
                    )
                    logger.info(
                        "[%s] Running lm_eval on %s/%s (task=%s, shots=%d, limit=%s, metric=%s, chat_template=%s)...",
                        tag,
                        cat,
                        suite_name,
                        task_name,
                        num_fewshot,
                        str(eff_limit),
                        metric_hint,
                        str(task_chat_tpl),
                    )
                    payload = lm_eval.simple_evaluate(
                        model=lm_obj,
                        tasks=tasks_arg,
                        num_fewshot=num_fewshot,
                        batch_size="auto" if effective_use_vllm else batch_size,
                        limit=eff_limit,
                        gen_kwargs=gen_kwargs,
                        random_seed=42,
                        numpy_random_seed=42,
                        torch_random_seed=42,
                        confirm_run_unsafe_code=True,
                        apply_chat_template=task_chat_tpl,
                        fewshot_as_multiturn=bool(task_chat_tpl),
                        log_samples=True,
                        bootstrap_iters=0,
                        task_manager=task_manager,
                    )
                    if not isinstance(payload, dict):
                        raise RuntimeError(
                            f"[{tag}] lm_eval.simple_evaluate returned None for {cat}/{suite_name} ({task_name})."
                        )
                    metrics = extract_lm_eval_detailed_metrics(
                        results_payload=payload,
                        task_name=primary_key,
                        metric_hint=metric_hint,
                        model_key_prefix=model_key_prefix,
                    )
                    out[cat][suite_name] = metrics
                    nll_display = f"{metrics['nll']:.4f}" if metrics.get("nll") is not None else "N/A"
                    logger.info(
                        "[%s] %s/%s -> %s=%.2f%% | HardTop1=%.2f%% | GoldNLL=%s (N=%d)",
                        tag,
                        cat,
                        suite_name,
                        metric_hint,
                        metrics["eval_score_pct"],
                        metrics["top1_acc_pct"],
                        nll_display,
                        metrics["n_samples"],
                    )

            _destroy_vllm_and_free_vram(lm_obj=lm_obj, model_obj=model_obj)
            lm_obj = None
            model_obj = None
            _flush_stage()

    # Stage 3: GPQA-Diamond & AIME-2026 via Inspect Evals (isolated subprocess)
    if pending_inspect_ret and not _skip_non_lm_eval:
        def _on_inspect_suite(sname: str, smetrics: Dict[str, Any]) -> None:
            out["retention"][sname] = smetrics
            _flush_stage()

        inspect_metrics = run_inspect_evals_subprocess(
            model_path=model_path,
            suite_specs=pending_inspect_ret,
            device=device,
            limit=limit,
            use_vllm=use_vllm,
            tag=tag,
            on_suite_complete=_on_inspect_suite,
        )
        for sname, smetrics in inspect_metrics.items():
            out["retention"][sname] = smetrics
        _flush_stage()

    # Stage 4: LiveCodeBench-v6 via LightEval (isolated subprocess)
    if pending_lcb_ret and not _skip_non_lm_eval:
        lcb_metrics = run_lighteval_lcb_subprocess(
            model_path=model_path,
            suite_specs=pending_lcb_ret,
            device=device,
            limit=limit,
            use_vllm=use_vllm,
            tag=tag,
        )
        for sname, smetrics in lcb_metrics.items():
            out["retention"][sname] = smetrics
        _flush_stage()

    return out


def _atomic_write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    """Atomically write `payload` to `path` via temp-file + fsync + `os.replace` for interrupt-safe Vast.ai checkpointing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    serialized = json.dumps(payload, indent=2)
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(serialized)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _load_stage_checkpoint(
    cache_path: pathlib.Path,
    limit: Optional[int],
    base_model_id: str,
    tuned_model_path: str,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Load completed per-stage `(base_ar, base_ret, tuned_ar, tuned_ret)` from `live_raw_{model_key}.json` when run parameters match."""
    if not cache_path.exists():
        return ({}, {}, {}, {})
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if raw.get("limit") != limit:
            logger.info(
                "Ignoring checkpoint %s due to limit mismatch (checkpoint=%s, current=%s).",
                cache_path.name,
                raw.get("limit"),
                limit,
            )
            return ({}, {}, {}, {})
        if str(raw.get("base_model_id") or "") != str(base_model_id):
            return ({}, {}, {}, {})
        if str(raw.get("tuned_model_path") or "") != str(tuned_model_path):
            return ({}, {}, {}, {})

        def _valid_entries(section: Any) -> Dict[str, Dict[str, Any]]:
            if not isinstance(section, dict):
                return {}
            return {
                k: v
                for k, v in section.items()
                if isinstance(v, dict)
                and "eval_score_pct" in v
                and not v.get("_skipped")
            }

        base_ar = _valid_entries(raw.get("base", {}).get("arabic", {}))
        base_ret = _valid_entries(raw.get("base", {}).get("retention", {}))
        tuned_ar = _valid_entries(raw.get("tuned", {}).get("arabic", {}))
        tuned_ret = _valid_entries(raw.get("tuned", {}).get("retention", {}))
        return (base_ar, base_ret, tuned_ar, tuned_ret)
    except Exception as exc:
        logger.warning("Could not read stage checkpoint %s: %s", cache_path, exc)
        return ({}, {}, {}, {})


def _resolve_selected_suites(
    cfg: Dict[str, Any],
    tasks_arg: Optional[str],
) -> Optional[Tuple[List[str], List[str]]]:
    """Parse `--tasks` / `--benchmarks` into `(selected_arabic_suites, selected_retention_suites)`."""
    if not tasks_arg or not str(tasks_arg).strip():
        return None

    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})

    def _norm_key(s: str) -> str:
        return re.sub(r"[\s_\-]+", "", str(s or "").strip().lower())

    alias_map: Dict[str, Tuple[str, str]] = {}
    for sname, spec in ar_defs.items():
        alias_map[_norm_key(sname)] = ("arabic", sname)
        if spec.get("lm_eval_task"):
            alias_map[_norm_key(str(spec["lm_eval_task"]))] = ("arabic", sname)
    for sname, spec in ret_defs.items():
        alias_map[_norm_key(sname)] = ("retention", sname)
        if spec.get("lm_eval_task"):
            alias_map[_norm_key(str(spec["lm_eval_task"]))] = ("retention", sname)

    shorthands = {
        "mmlu": ("retention", "MMLU-Pro"),
        "mmlupro": ("retention", "MMLU-Pro"),
        "gpqa": ("retention", "GPQA-Diamond"),
        "gpqadiamond": ("retention", "GPQA-Diamond"),
        "aime": ("retention", "AIME-2026"),
        "aime26": ("retention", "AIME-2026"),
        "aime2026": ("retention", "AIME-2026"),
        "lcb": ("retention", "LiveCodeBench-v6"),
        "lcbv6": ("retention", "LiveCodeBench-v6"),
        "livecodebench": ("retention", "LiveCodeBench-v6"),
        "livecodebenchv6": ("retention", "LiveCodeBench-v6"),
        "arabicmmlu": ("arabic", "ArabicMMLU"),
        "madinah": ("arabic", "MadinahQA"),
        "madinahqa": ("arabic", "MadinahQA"),
        "alghafa": ("arabic", "AlGhafa"),
        "acva": ("arabic", "ACVA"),
        "exams": ("arabic", "EXAMS-AR"),
        "examsar": ("arabic", "EXAMS-AR"),
        "arabicexams": ("arabic", "EXAMS-AR"),
        "aratrust": ("arabic", "AraTrust"),
    }
    for k, v in shorthands.items():
        alias_map.setdefault(k, v)

    raw_tokens = [tok.strip() for tok in re.split(r"[,;]+", str(tasks_arg)) if tok.strip()]
    sel_ar: List[str] = []
    sel_ret: List[str] = []
    unknown: List[str] = []

    for tok in raw_tokens:
        nk = _norm_key(tok)
        if nk not in alias_map:
            unknown.append(tok)
            continue
        cat, canonical_name = alias_map[nk]
        if cat == "arabic" and canonical_name not in sel_ar:
            sel_ar.append(canonical_name)
        elif cat == "retention" and canonical_name not in sel_ret:
            sel_ret.append(canonical_name)

    if unknown:
        valid_names = list(ar_defs.keys()) + list(ret_defs.keys())
        raise ValueError(
            f"Unknown benchmark name(s) in --tasks: {unknown}. "
            f"Valid benchmarks are: {', '.join(valid_names)}"
        )
    return (sel_ar, sel_ret)


def _extract_all_cached_suites(
    cache_path: pathlib.Path,
    model_key: str = "e4b",
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Return `(base_ar, tuned_ar, base_ret, tuned_ret)` strictly from `live_raw_{model_key}.json` if present."""
    _ = model_key
    out_base_ar: Dict[str, Dict[str, Any]] = {}
    out_tuned_ar: Dict[str, Dict[str, Any]] = {}
    out_base_ret: Dict[str, Dict[str, Any]] = {}
    out_tuned_ret: Dict[str, Dict[str, Any]] = {}

    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            base_ar = raw.get("base", {}).get("arabic", {})
            tuned_ar = raw.get("tuned", {}).get("arabic", {})
            base_ret = raw.get("base", {}).get("retention", {})
            tuned_ret = raw.get("tuned", {}).get("retention", {})
            if isinstance(base_ar, dict):
                out_base_ar.update(base_ar)
            if isinstance(tuned_ar, dict):
                out_tuned_ar.update(tuned_ar)
            if isinstance(base_ret, dict):
                out_base_ret.update(base_ret)
            if isinstance(tuned_ret, dict):
                out_tuned_ret.update(tuned_ret)
        except Exception:
            pass

    return (out_base_ar, out_tuned_ar, out_base_ret, out_tuned_ret)


def run_inprocess_gpu_eval(
    model_key: str,
    base_model_id: str,
    tuned_model_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    target_device: str = "cuda:0",
    limit: Optional[int] = None,
    use_cached_eval: bool = False,
    use_vllm: bool = True,
    apply_chat_template: bool = True,
    selected_tasks: Optional[str] = None,
    load_cache_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run standardized GPU evaluation across all 10 suites (or `--tasks` subset) with interrupt-safe per-stage checkpointing."""
    if cfg is None:
        cfg = load_eval_config(PROJECT_ROOT / "configs" / "eval_config.yaml")

    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})
    cache_path = PROJECT_ROOT / "outputs" / "eval_reports" / f"live_raw_{model_key}.json"

    if load_cache_only and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if "base" in cached and "tuned" in cached:
            logger.info("Loaded completed live GPU results from %s", cache_path)
            return cached

    if use_cached_eval and not selected_tasks:
        c_base_ar, c_base_ret, c_tuned_ar, c_tuned_ret = _load_stage_checkpoint(
            cache_path=cache_path,
            limit=limit,
            base_model_id=base_model_id,
            tuned_model_path=tuned_model_path,
        )
        all_ar_done = all(s in c_base_ar and s in c_tuned_ar for s in ar_defs)
        all_ret_done = all(s in c_base_ret and s in c_tuned_ret for s in ret_defs)
        if ar_defs and ret_defs and all_ar_done and all_ret_done:
            logger.info("All 10 benchmark suites already completed in %s; reusing checkpoint.", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        c_base_ar, c_base_ret, c_tuned_ar, c_tuned_ret = ({}, {}, {}, {})

    resolved_selection = _resolve_selected_suites(cfg, selected_tasks)

    import torch

    if not torch.cuda.is_available():
        return None

    use_dev = target_device
    if ":" in use_dev:
        idx = int(use_dev.split(":")[1])
        if idx >= torch.cuda.device_count():
            use_dev = "cuda:0"

    if resolved_selection is not None:
        sel_ar, sel_ret = resolved_selection
        ex_base_ar, ex_tuned_ar, ex_base_ret, ex_tuned_ret = _extract_all_cached_suites(
            cache_path, model_key=model_key
        )
        pres_base_ar: Dict[str, Dict[str, Any]] = {}
        pres_tuned_ar: Dict[str, Dict[str, Any]] = {}
        pres_base_ret: Dict[str, Dict[str, Any]] = {}
        pres_tuned_ret: Dict[str, Dict[str, Any]] = {}

        for sname in ar_defs:
            if sname not in sel_ar:
                pres_base_ar[sname] = ex_base_ar.get(sname, {"_skipped": True})
                pres_tuned_ar[sname] = ex_tuned_ar.get(sname, {"_skipped": True})
        for sname in ret_defs:
            if sname not in sel_ret:
                pres_base_ret[sname] = ex_base_ret.get(sname, {"_skipped": True})
                pres_tuned_ret[sname] = ex_tuned_ret.get(sname, {"_skipped": True})

        logger.info(
            "[%s] --tasks filter active -> evaluating ONLY arabic=%s, retention=%s",
            model_key,
            sel_ar,
            sel_ret,
        )
    else:
        pres_base_ar = dict(c_base_ar)
        pres_base_ret = dict(c_base_ret)
        pres_tuned_ar = dict(c_tuned_ar)
        pres_tuned_ret = dict(c_tuned_ret)
        if pres_base_ar or pres_base_ret or pres_tuned_ar or pres_tuned_ret:
            logger.info(
                "[%s] Resuming stage checkpoint from %s (base: %d ar / %d ret done; tuned: %d ar / %d ret done)",
                model_key,
                cache_path.name,
                len(pres_base_ar),
                len(pres_base_ret),
                len(pres_tuned_ar),
                len(pres_tuned_ret),
            )

    def _clean_side(side: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        cleaned_side: Dict[str, Dict[str, Any]] = {"arabic": {}, "retention": {}}
        for grp_name in ("arabic", "retention"):
            for k, v in (side.get(grp_name) or {}).items():
                if isinstance(v, dict) and not v.get("_skipped"):
                    cleaned_side[grp_name][k] = v
        return cleaned_side

    live_state: Dict[str, Dict[str, Dict[str, Any]]] = {
        "base": _clean_side({"arabic": pres_base_ar, "retention": pres_base_ret}),
        "tuned": _clean_side({"arabic": pres_tuned_ar, "retention": pres_tuned_ret}),
    }

    def _persist_live_state() -> Dict[str, Any]:
        snap = {
            "model_key": model_key,
            "base_model_id": base_model_id,
            "tuned_model_path": tuned_model_path,
            "device": use_dev,
            "limit": limit,
            "base": _clean_side(live_state["base"]),
            "tuned": _clean_side(live_state["tuned"]),
        }
        _atomic_write_json(cache_path, snap)
        return snap

    def _on_base_checkpoint(updated_base: Dict[str, Dict[str, Any]]) -> None:
        live_state["base"] = updated_base
        _persist_live_state()

    def _on_tuned_checkpoint(updated_tuned: Dict[str, Dict[str, Any]]) -> None:
        live_state["tuned"] = updated_tuned
        _persist_live_state()

    base_metrics = _evaluate_checkpoint_with_lm_eval(
        model_path=base_model_id,
        device=use_dev,
        cfg=cfg,
        tag=f"{model_key}:base",
        limit=limit,
        use_vllm=use_vllm,
        apply_chat_template=apply_chat_template,
        _preserved_arabic=pres_base_ar,
        _preserved_retention=pres_base_ret,
        checkpoint_callback=_on_base_checkpoint,
    )
    live_state["base"] = base_metrics
    _persist_live_state()

    tuned_metrics = _evaluate_checkpoint_with_lm_eval(
        model_path=tuned_model_path,
        device=use_dev,
        cfg=cfg,
        tag=f"{model_key}:tuned",
        limit=limit,
        use_vllm=use_vllm,
        apply_chat_template=apply_chat_template,
        _preserved_arabic=pres_tuned_ar,
        _preserved_retention=pres_tuned_ret,
        checkpoint_callback=_on_tuned_checkpoint,
    )
    live_state["tuned"] = tuned_metrics
    payload = _persist_live_state()
    logger.info("Saved live GPU evaluation metrics to %s", cache_path)
    return payload


def _run_single_suite_fallback(
    checkpoint_path: str,
    suite_name: str,
    spec: Dict[str, Any],
    batch_size: int,
    device: str,
    limit: Optional[int],
    use_vllm: bool,
    apply_chat_template: bool,
    tag: str,
) -> float:
    """Dispatch a single suite to its configured runner (`lighteval`, `inspect_evals`, `lighteval_lcb`, or `lm_eval`)."""
    runner = str(spec.get("runner", "lm_eval")).strip().lower()
    if runner == "lighteval":
        res = run_lighteval_tasks_subprocess(
            model_path=checkpoint_path,
            suite_names=[suite_name],
            device=device,
            limit=limit,
            batch_size=batch_size,
            use_vllm=use_vllm,
            tag=tag,
        )
        return float(res[suite_name]["eval_score_pct"])
    if runner == "inspect_evals":
        res = run_inspect_evals_subprocess(
            model_path=checkpoint_path,
            suite_specs={suite_name: spec},
            device=device,
            limit=limit,
            use_vllm=use_vllm,
            tag=tag,
        )
        return float(res[suite_name]["eval_score_pct"])
    if runner == "lighteval_lcb":
        res = run_lighteval_lcb_subprocess(
            model_path=checkpoint_path,
            suite_specs={suite_name: spec},
            device=device,
            limit=limit,
            use_vllm=use_vllm,
            tag=tag,
        )
        return float(res[suite_name]["eval_score_pct"])
    return run_lm_eval_task(
        model_path=checkpoint_path,
        task_name=str(spec["lm_eval_task"]),
        num_fewshot=int(spec.get("num_fewshot", 0)),
        batch_size=batch_size,
        device=device,
        limit=limit,
        metric_hint=str(spec.get("metric", "acc_norm")),
        use_vllm=use_vllm,
        apply_chat_template=apply_chat_template,
    )


def build_benchmark_results(
    cfg: Dict[str, Any],
    model_path: str,
    dry_run: bool = False,
    limit: Optional[int] = None,
    model_key: str = "e4b",
    eval_base: bool = False,
    use_cached_eval: bool = False,
    no_reference_fallback: bool = True,
    use_vllm: bool = False,
    apply_chat_template: bool = True,
    selected_tasks: Optional[str] = None,
    load_cache_only: bool = False,
) -> Dict[str, Any]:
    """Run all 6 OALL v2 Arabic benchmarks (`lighteval`) and 4 official Gemma 4 retention anchors (`lm_eval`, `inspect_evals`, `lighteval_lcb`)."""
    eval_cfg = cfg.get("evaluation", {})
    gate_cfg = cfg.get("retention_gate", {})
    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})
    resolved_selection = _resolve_selected_suites(cfg, selected_tasks)

    models_cfg = cfg.get("models", {})
    model_entry = models_cfg.get(model_key, {}) if isinstance(models_cfg, dict) else {}
    base_model_id = str(model_entry.get("base_model_id", model_path))
    target_device = str(model_entry.get("device", eval_cfg.get("device", "cuda:0")))

    live_gpu_data: Optional[Dict[str, Any]] = None
    if not dry_run:
        live_gpu_data = run_inprocess_gpu_eval(
            model_key=model_key,
            base_model_id=base_model_id,
            tuned_model_path=model_path,
            cfg=cfg,
            target_device=target_device,
            limit=limit,
            use_cached_eval=use_cached_eval,
            use_vllm=use_vllm,
            apply_chat_template=apply_chat_template,
            selected_tasks=selected_tasks,
            load_cache_only=load_cache_only,
        )

    arabic_results: Dict[str, Dict[str, Any]] = {}
    for name, spec in ar_defs.items():
        base_score = float(spec.get("baseline_score", 0.0))
        base_nll: Optional[float] = None
        tuned_nll: Optional[float] = None
        n_samples_ar = int(limit or 0) if limit else 0

        if dry_run or (resolved_selection is not None and name not in resolved_selection[0] and (
            live_gpu_data is None
            or name not in live_gpu_data.get("base", {}).get("arabic", {})
            or name not in live_gpu_data.get("tuned", {}).get("arabic", {})
        )):
            evaluated_score = base_score
        elif (
            live_gpu_data is not None
            and name in live_gpu_data.get("base", {}).get("arabic", {})
            and name in live_gpu_data.get("tuned", {}).get("arabic", {})
        ):
            b_item = live_gpu_data["base"]["arabic"][name]
            t_item = live_gpu_data["tuned"]["arabic"][name]
            base_nll = float(b_item["nll"]) if b_item.get("nll") is not None else None
            tuned_nll = float(t_item["nll"]) if t_item.get("nll") is not None else None
            n_samples_ar = int(t_item.get("n_samples", b_item.get("n_samples", n_samples_ar)) or 0)
            base_score = round(float(b_item["eval_score_pct"]), 2)
            evaluated_score = round(float(t_item["eval_score_pct"]), 2)
        else:
            if eval_base or base_score == 0.0 or no_reference_fallback:
                try:
                    base_score = _run_single_suite_fallback(
                        checkpoint_path=str(base_model_id),
                        suite_name=name,
                        spec=spec,
                        batch_size=int(eval_cfg.get("batch_size", 16)),
                        device=target_device,
                        limit=limit,
                        use_vllm=use_vllm,
                        apply_chat_template=apply_chat_template,
                        tag=f"{model_key}:base",
                    )
                except Exception as exc:
                    if no_reference_fallback:
                        raise RuntimeError(
                            f"Base task '{name}' ({spec['lm_eval_task']}) failed and "
                            f"--no-reference-fallback is active: {exc}"
                        ) from exc
                    logger.warning("Base eval fallback for %s (%s): %s", name, spec["lm_eval_task"], exc)
            try:
                evaluated_score = _run_single_suite_fallback(
                    checkpoint_path=model_path,
                    suite_name=name,
                    spec=spec,
                    batch_size=int(eval_cfg.get("batch_size", 16)),
                    device=target_device,
                    limit=limit,
                    use_vllm=use_vllm,
                    apply_chat_template=apply_chat_template,
                    tag=f"{model_key}:tuned",
                )
            except Exception as exc:
                if no_reference_fallback:
                    raise RuntimeError(
                        f"Tuned task '{name}' ({spec['lm_eval_task']}) failed and "
                        f"--no-reference-fallback is active: {exc}"
                    ) from exc
                logger.warning(
                    "Live task '%s' (%s) failed: %s",
                    name,
                    spec["lm_eval_task"],
                    exc,
                )
                evaluated_score = base_score

        entry_dict: Dict[str, Any] = {
            "lm_eval_task": spec["lm_eval_task"],
            "lighteval_task": spec.get("lighteval_task", ""),
            "runner": spec.get("runner", "lm_eval"),
            "metric": spec["metric"],
            "num_fewshot": int(spec.get("num_fewshot", 0)),
            "baseline_score": base_score,
            "evaluated_score": evaluated_score,
            "gain": round(evaluated_score - base_score, 4),
        }
        if n_samples_ar > 0:
            entry_dict["n_samples"] = n_samples_ar
        if base_nll is not None and tuned_nll is not None:
            entry_dict["base_nll"] = round(base_nll, 4)
            entry_dict["tuned_nll"] = round(tuned_nll, 4)
        arabic_results[name] = entry_dict

    anchor_results: Dict[str, Dict[str, Any]] = {}
    for name, spec in ret_defs.items():
        base_score = (
            float(spec.get("baseline_score", spec.get("min_acceptable_score", 0.0)))
            if dry_run
            else float(spec.get("baseline_score", 0.0))
        )
        base_nll = None
        tuned_nll = None
        n_samples_anchor = int(limit or 0) if limit else 0

        if dry_run or (resolved_selection is not None and name not in resolved_selection[1] and (
            live_gpu_data is None
            or name not in live_gpu_data.get("base", {}).get("retention", {})
            or name not in live_gpu_data.get("tuned", {}).get("retention", {})
        )):
            evaluated_score = base_score
        elif (
            live_gpu_data is not None
            and name in live_gpu_data.get("base", {}).get("retention", {})
            and name in live_gpu_data.get("tuned", {}).get("retention", {})
        ):
            b_item = live_gpu_data["base"]["retention"][name]
            t_item = live_gpu_data["tuned"]["retention"][name]
            base_nll = float(b_item["nll"]) if b_item.get("nll") is not None else None
            tuned_nll = float(t_item["nll"]) if t_item.get("nll") is not None else None
            n_samples_anchor = int(t_item.get("n_samples", b_item.get("n_samples", n_samples_anchor)) or 0)
            base_score = round(float(b_item["eval_score_pct"]), 2)
            evaluated_score = round(float(t_item["eval_score_pct"]), 2)
        else:
            if eval_base or no_reference_fallback:
                try:
                    base_score = _run_single_suite_fallback(
                        checkpoint_path=str(base_model_id),
                        suite_name=name,
                        spec=spec,
                        batch_size=int(eval_cfg.get("batch_size", 16)),
                        device=target_device,
                        limit=limit,
                        use_vllm=use_vllm,
                        apply_chat_template=apply_chat_template,
                        tag=f"{model_key}:base",
                    )
                except Exception as exc:
                    if no_reference_fallback:
                        raise RuntimeError(
                            f"Base retention anchor '{name}' ({spec['lm_eval_task']}) failed and "
                            f"--no-reference-fallback is active: {exc}"
                        ) from exc
                    logger.warning("Base anchor eval fallback for %s: %s", name, exc)
            try:
                evaluated_score = _run_single_suite_fallback(
                    checkpoint_path=model_path,
                    suite_name=name,
                    spec=spec,
                    batch_size=int(eval_cfg.get("batch_size", 16)),
                    device=target_device,
                    limit=limit,
                    use_vllm=use_vllm,
                    apply_chat_template=apply_chat_template,
                    tag=f"{model_key}:tuned",
                )
            except Exception as exc:
                if no_reference_fallback:
                    raise RuntimeError(
                        f"Tuned retention anchor '{name}' ({spec['lm_eval_task']}) failed and "
                        f"--no-reference-fallback is active: {exc}"
                    ) from exc
                logger.warning(
                    "Live retention anchor '%s' (%s) fallback to baseline score: %s",
                    name,
                    spec["lm_eval_task"],
                    exc,
                )
                evaluated_score = base_score

        anchor_entry: Dict[str, Any] = {
            "domain": spec["domain"],
            "lm_eval_task": spec["lm_eval_task"],
            "runner": spec.get("runner", "lm_eval"),
            "metric": spec["metric"],
            "num_fewshot": int(spec.get("num_fewshot", 0)),
            "baseline_score": base_score,
            "evaluated_score": evaluated_score,
            "max_regression_pct": float(spec.get("max_regression_pct", 0.20)),
        }
        if n_samples_anchor > 0:
            anchor_entry["n_samples"] = n_samples_anchor
        if base_nll is not None and tuned_nll is not None:
            anchor_entry["base_nll"] = round(base_nll, 4)
            anchor_entry["tuned_nll"] = round(tuned_nll, 4)
        anchor_results[name] = anchor_entry

    gate_summary = evaluate_retention_gate(
        anchor_results=anchor_results,
        min_retention_ratio=float(gate_cfg.get("min_retention_ratio", 0.998)),
        max_allowed_drop_pct=float(gate_cfg.get("max_allowed_drop_pct", 0.20)),
        sweep_lambdas=gate_cfg.get("sweep_lambdas_on_regression", [0.80, 0.85, 0.90]),
    )

    avg_ar_base = sum(v["baseline_score"] for v in arabic_results.values()) / max(
        1, len(arabic_results)
    )
    avg_ar_eval = sum(v["evaluated_score"] for v in arabic_results.values()) / max(
        1, len(arabic_results)
    )

    return {
        "mode": "dry_run" if dry_run else ("live_lm_eval" if live_gpu_data else "live_eval"),
        "model_key": model_key,
        "model_path": model_path,
        "arabic_benchmarks_count": len(arabic_results),
        "retention_anchors_count": len(anchor_results),
        "average_arabic_baseline": round(avg_ar_base, 4),
        "average_arabic_evaluated": round(avg_ar_eval, 4),
        "average_arabic_gain": round(avg_ar_eval - avg_ar_base, 4),
        "arabic_benchmarks": arabic_results,
        "retention_anchors": anchor_results,
        "retention_gate": gate_summary,
    }


def generate_family_markdown_report(
    e4b_results: Dict[str, Any],
    moe_results: Dict[str, Any],
) -> str:
    """Render a side-by-side Markdown report comparing E4B and 26B-A4B using Base | Arabic-Tuned columns."""
    e4b_gain_sign = "+" if e4b_results["average_arabic_gain"] >= 0 else ""
    moe_gain_sign = "+" if moe_results["average_arabic_gain"] >= 0 else ""
    lines: List[str] = [
        "# Gemma-4-Arabic Iso-Active Family Benchmark Comparison Report",
        "",
        f"- **Dense Edge Model (`E4B`):** `{e4b_results['model_path']}` (Avg Arabic Gain: `{e4b_gain_sign}{e4b_results['average_arabic_gain']:.2f}` pts)",
        f"- **Sparse MoE Flagship (`26B-A4B`):** `{moe_results['model_path']}` (Avg Arabic Gain: `{moe_gain_sign}{moe_results['average_arabic_gain']:.2f}` pts)",
        f"- **E4B Zero-Forgetting Gate:** `{'PASSED' if e4b_results['retention_gate']['gate_passed'] else 'FAILED'}` (`{e4b_results['retention_gate']['min_observed_retention_ratio']:.4f}`)",
        f"- **26B-A4B Zero-Forgetting Gate:** `{'PASSED' if moe_results['retention_gate']['gate_passed'] else 'FAILED'}` (`{moe_results['retention_gate']['min_observed_retention_ratio']:.4f}`)",
        "",
        "## 1. OALL v2 Arabic Benchmarks (Full Dataset)",
        "",
        "| Benchmark | Metric | Shots | E4B Base (%) | E4B Arabic-Tuned (%) | E4B Gain (% pts) | 26B-A4B Base (%) | 26B-A4B Arabic-Tuned (%) | 26B-A4B Gain (% pts) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, e4b_item in e4b_results["arabic_benchmarks"].items():
        moe_item = moe_results["arabic_benchmarks"].get(name, e4b_item)
        e4b_base = float(e4b_item["baseline_score"])
        e4b_ev = float(e4b_item["evaluated_score"])
        e4b_gain = e4b_ev - e4b_base
        e4b_s = "+" if e4b_gain >= 0 else ""
        moe_base = float(moe_item["baseline_score"])
        moe_ev = float(moe_item["evaluated_score"])
        moe_gain = moe_ev - moe_base
        moe_s = "+" if moe_gain >= 0 else ""
        lines.append(
            f"| **{name}** | `{e4b_item['metric']}` | {e4b_item['num_fewshot']} | "
            f"{e4b_base:.2f} | **{e4b_ev:.2f}** | `{e4b_s}{e4b_gain:.2f}` | "
            f"{moe_base:.2f} | **{moe_ev:.2f}** | `{moe_s}{moe_gain:.2f}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Official Gemma 4 Retention Anchors (Thinking Mode ON)",
            "",
            "| Anchor | Domain | Shots | E4B Base (%) | E4B Arabic-Tuned (%) | E4B Delta (% pts) | 26B-A4B Base (%) | 26B-A4B Arabic-Tuned (%) | 26B-A4B Delta (% pts) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name, e4b_item in e4b_results["retention_anchors"].items():
        moe_item = moe_results["retention_anchors"].get(name, e4b_item)
        e4b_base = float(e4b_item["baseline_score"])
        e4b_ev = float(e4b_item["evaluated_score"])
        e4b_delta = e4b_ev - e4b_base
        e4b_s = "+" if e4b_delta >= 0 else ""
        moe_base = float(moe_item["baseline_score"])
        moe_ev = float(moe_item["evaluated_score"])
        moe_delta = moe_ev - moe_base
        moe_s = "+" if moe_delta >= 0 else ""
        lines.append(
            f"| **{name}** | `{e4b_item['domain']}` | {e4b_item.get('num_fewshot', 0)} | "
            f"{e4b_base:.2f} | **{e4b_ev:.2f}** | `{e4b_s}{e4b_delta:.2f}` | "
            f"{moe_base:.2f} | **{moe_ev:.2f}** | `{moe_s}{moe_delta:.2f}` |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the benchmark evaluation runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the 6 OALL v2 Arabic benchmarks (lighteval) and 4 official Gemma 4 retention anchors "
            "(MMLU-Pro, GPQA-Diamond, AIME-2026, LiveCodeBench-v6) across Base and Arabic-Tuned checkpoints."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "eval_config.yaml"),
        help="Path to evaluation YAML configuration file.",
    )
    parser.add_argument(
        "--model-key",
        "--model-family",
        dest="model_key",
        type=str,
        choices=["e4b", "26b_a4b", "both"],
        default="e4b",
        help="Model family member to evaluate: e4b, 26b_a4b, or both.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model checkpoint path or Hugging Face ID to evaluate.",
    )
    parser.add_argument(
        "--eval-base",
        action="store_true",
        help="Also run live evaluation on the untuned base model instead of using cached baseline scores.",
    )
    parser.add_argument(
        "--use-cached-eval",
        action="store_true",
        default=False,
        help="Resume from completed stage checkpoints in outputs/eval_reports/live_raw_*.json when limit and model paths match.",
    )
    parser.add_argument(
        "--no-reference-fallback",
        dest="no_reference_fallback",
        action="store_true",
        default=True,
        help="Fail loudly instead of falling back to reference YAML numbers if live evaluation fails (default: True).",
    )
    parser.add_argument(
        "--allow-reference-fallback",
        dest="no_reference_fallback",
        action="store_false",
        help="Allow falling back to reference YAML baseline scores if live evaluation fails.",
    )
    parser.add_argument(
        "--use-vllm",
        dest="use_vllm",
        action="store_true",
        default=True,
        help="Use vLLM backend for lighteval, lm_eval, and inspect_evals (default: True).",
    )
    parser.add_argument(
        "--no-vllm",
        dest="use_vllm",
        action="store_false",
        help="Use Hugging Face HFLM / Accelerate instead of vLLM.",
    )
    parser.add_argument(
        "--apply-chat-template",
        dest="apply_chat_template",
        action="store_true",
        default=True,
        help="Apply the model's Hugging Face chat template consistently across both base and tuned generative runs (default: True).",
    )
    parser.add_argument(
        "--no-apply-chat-template",
        dest="apply_chat_template",
        action="store_false",
        help="Disable applying the chat template on generative lm_eval tasks.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to write the Markdown benchmark comparison report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write the JSON benchmark results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit per benchmark suite (omit for full-dataset evaluation).",
    )
    parser.add_argument(
        "--tasks",
        "--benchmarks",
        dest="tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated list of specific benchmarks to evaluate on Base and Arabic-Tuned "
            "(e.g., 'MMLU-Pro,GPQA-Diamond,AIME-2026,LiveCodeBench-v6' or 'ArabicMMLU,ACVA'). "
            "Unselected benchmarks are loaded from outputs/eval_reports/live_raw_*.json if present."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all 6 OALL v2 Arabic benchmarks, 4 retention anchors, and Markdown report generation.",
    )
    return parser.parse_args(argv)


def _build_child_eval_cmd(args: argparse.Namespace, model_key: str) -> List[str]:
    """Construct CLI command for an isolated single-GPU child subprocess."""
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--model-key",
        model_key,
    ]
    if args.eval_base:
        cmd.append("--eval-base")
    if args.use_cached_eval:
        cmd.append("--use-cached-eval")
    if args.no_reference_fallback:
        cmd.append("--no-reference-fallback")
    else:
        cmd.append("--allow-reference-fallback")
    if args.use_vllm:
        cmd.append("--use-vllm")
    else:
        cmd.append("--no-vllm")
    if args.apply_chat_template:
        cmd.append("--apply-chat-template")
    else:
        cmd.append("--no-apply-chat-template")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if getattr(args, "tasks", None):
        cmd.extend(["--tasks", str(args.tasks)])
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for automated benchmark evaluation and report generation."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    args = parse_args(argv)

    cfg = load_eval_config(pathlib.Path(args.config))
    eval_cfg = cfg.get("evaluation", {})
    models_cfg = cfg.get("models", {})

    if args.model_key == "both":
        e4b_path = str(
            models_cfg.get("e4b", {}).get(
                "merged_checkpoint_dir", "outputs/merged_gemma4_e4b_arabic/merged_16bit"
            )
        )
        moe_path = str(
            models_cfg.get("26b_a4b", {}).get(
                "merged_checkpoint_dir", "outputs/merged_gemma4_26b_a4b_arabic/merged_16bit"
            )
        )

        can_parallel_dual_gpu = False
        if not args.dry_run:
            try:
                import torch

                can_parallel_dual_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2
            except Exception:
                can_parallel_dual_gpu = False

        if can_parallel_dual_gpu:
            logger.info(
                "Detected >= 2 GPUs: launching isolated subprocesses for E4B (CUDA_VISIBLE_DEVICES=0) and 26B-A4B (CUDA_VISIBLE_DEVICES=1)..."
            )
            env_gpu0 = os.environ.copy()
            env_gpu0["CUDA_VISIBLE_DEVICES"] = "0"
            env_gpu1 = os.environ.copy()
            env_gpu1["CUDA_VISIBLE_DEVICES"] = "1"

            proc_e4b = subprocess.Popen(
                _build_child_eval_cmd(args, "e4b"),
                env=env_gpu0,
                cwd=str(PROJECT_ROOT),
            )
            proc_moe = subprocess.Popen(
                _build_child_eval_cmd(args, "26b_a4b"),
                env=env_gpu1,
                cwd=str(PROJECT_ROOT),
            )
            rc_e4b = proc_e4b.wait()
            rc_moe = proc_moe.wait()
            if rc_e4b != 0 or rc_moe != 0:
                raise RuntimeError(
                    f"Parallel Dual-GPU evaluation subprocess failed (rc_e4b={rc_e4b}, rc_26b_a4b={rc_moe})."
                )

            # Load completed results from outputs/eval_reports/live_raw_e4b.json and live_raw_26b_a4b.json
            e4b_res = build_benchmark_results(
                cfg=cfg,
                model_path=e4b_path,
                dry_run=False,
                limit=args.limit,
                model_key="e4b",
                eval_base=args.eval_base,
                use_cached_eval=True,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
                selected_tasks=args.tasks,
                load_cache_only=True,
            )
            moe_res = build_benchmark_results(
                cfg=cfg,
                model_path=moe_path,
                dry_run=False,
                limit=args.limit,
                model_key="26b_a4b",
                eval_base=args.eval_base,
                use_cached_eval=True,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
                selected_tasks=args.tasks,
                load_cache_only=True,
            )
        else:
            e4b_res = build_benchmark_results(
                cfg=cfg,
                model_path=e4b_path,
                dry_run=args.dry_run,
                limit=args.limit,
                model_key="e4b",
                eval_base=args.eval_base,
                use_cached_eval=args.use_cached_eval,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
                selected_tasks=args.tasks,
            )
            moe_res = build_benchmark_results(
                cfg=cfg,
                model_path=moe_path,
                dry_run=args.dry_run,
                limit=args.limit,
                model_key="26b_a4b",
                eval_base=args.eval_base,
                use_cached_eval=args.use_cached_eval,
                no_reference_fallback=args.no_reference_fallback,
                use_vllm=args.use_vllm,
                apply_chat_template=args.apply_chat_template,
                selected_tasks=args.tasks,
            )

        family_md = generate_family_markdown_report(e4b_res, moe_res)
        family_payload = {
            "mode": e4b_res.get("mode", "live_lm_eval"),
            "models": {
                "e4b": e4b_res,
                "26b_a4b": moe_res,
            },
            **e4b_res,
        }
        if args.output_md or not args.dry_run:
            md_path = pathlib.Path(
                args.output_md
                or eval_cfg.get(
                    "output_report_path",
                    "outputs/eval_reports/benchmark_comparison.md",
                )
            )
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(family_md, encoding="utf-8")
            fam_md_path = PROJECT_ROOT / "outputs" / "eval_reports" / "family_benchmark_comparison.md"
            fam_md_path.parent.mkdir(parents=True, exist_ok=True)
            fam_md_path.write_text(family_md, encoding="utf-8")
            legacy_md_path = PROJECT_ROOT / "outputs" / "eval_reports" / "benchmark_report.md"
            legacy_md_path.write_text(family_md, encoding="utf-8")
        if args.output_json or not args.dry_run:
            json_path = pathlib.Path(
                args.output_json
                or eval_cfg.get(
                    "output_json_path",
                    "outputs/eval_reports/benchmark_results.json",
                )
            )
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(family_payload, indent=2), encoding="utf-8")
            fam_json_path = pathlib.Path(
                eval_cfg.get(
                    "family_comparison_json_path",
                    "outputs/eval_reports/family_benchmark_comparison.json",
                )
            )
            fam_json_path.parent.mkdir(parents=True, exist_ok=True)
            fam_json_path.write_text(json.dumps(family_payload, indent=2), encoding="utf-8")
        print(family_md)
        return 0

    default_path = (
        models_cfg.get(args.model_key, {}).get("merged_checkpoint_dir")
        if isinstance(models_cfg, dict) and args.model_key in models_cfg
        else None
    )
    model_path = str(
        args.model_path
        or default_path
        or eval_cfg.get("merged_checkpoint_dir", "outputs/merged_gemma4_e4b_arabic")
    )

    results = build_benchmark_results(
        cfg=cfg,
        model_path=model_path,
        dry_run=args.dry_run,
        limit=args.limit,
        model_key=args.model_key,
        eval_base=args.eval_base,
        use_cached_eval=args.use_cached_eval,
        no_reference_fallback=args.no_reference_fallback,
        use_vllm=args.use_vllm,
        apply_chat_template=args.apply_chat_template,
        selected_tasks=args.tasks,
    )
    md_report = generate_markdown_report(results)

    if args.output_md or not args.dry_run:
        md_path = pathlib.Path(
            args.output_md
            or eval_cfg.get("output_report_path", "outputs/eval_reports/benchmark_comparison.md")
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_report, encoding="utf-8")

    if args.output_json or not args.dry_run:
        json_path = pathlib.Path(
            args.output_json
            or eval_cfg.get("output_json_path", "outputs/eval_reports/benchmark_results.json")
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(md_report)
    print(
        json.dumps(
            {
                "mode": results["mode"],
                "model_key": results["model_key"],
                "arabic_benchmarks_count": results["arabic_benchmarks_count"],
                "retention_anchors_count": results["retention_anchors_count"],
                "average_arabic_gain": results["average_arabic_gain"],
                "retention_gate_passed": results["retention_gate"]["gate_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
