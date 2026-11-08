"""Automated benchmark runner and Zero-Forgetting Retention Gate evaluator.

Evaluates:
- 8 Native `lm_eval` Benchmarks (`ArabicMMLU`, `AlGhafa`, `ACVA`, `EXAMS-AR`,
  `ARC-Challenge`, `GSM8K`, `HumanEval`, `MMLU-Pro`) + `MGSM-Ar` using default
  `lm-evaluation-harness` (`lm_eval`) settings with `vLLM` (`batch_size="auto"`).
- 2 Native `lighteval` Benchmarks (`MadinahQA`, `AraTrust`) using default
  `lighteval` settings with `vLLM`.
- Supports single-model (`e4b` or `26b_a4b`) and dual-model family (`both`)
  comparisons across `Gemma-4-E4B-Arabic-It` and `Gemma-4-26B-A4B-Arabic-It`
  using isolated per-GPU subprocesses (`CUDA_VISIBLE_DEVICES=0` and `1`).
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
from typing import Any, Dict, List, Optional, Tuple, Union

# Enforce deterministic hashing, multiprocessing spawn for vLLM, and HumanEval execution
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

EVAL_CACHE_VERSION = "v30_family_verified"

# LightEval suites whose validated scores are preserved across partial cache refreshes
PRESERVED_LIGHTEVAL_SUITES: Tuple[str, ...] = (
    "AlGhafa",
    "ACVA",
    "EXAMS-AR",
    "AraTrust",
)

# Core ACVA subtasks in lm_eval covering Saudi & Arab cultural alignment when --limit is used
ACVA_CORE_SUBTASKS: Tuple[str, ...] = (
    "arabic_leaderboard_acva_Saudi_Arabia_light",
    "arabic_leaderboard_acva_Arab_Empire_light",
    "arabic_leaderboard_acva_Arabic_Culture_light",
    "arabic_leaderboard_acva_Islam_Education_light",
    "arabic_leaderboard_acva_Islamic_law_system_light",
)

# Number of subtasks in multi-task lm_eval groups for proportional --limit scaling
GROUP_SUBTASK_COUNTS: Dict[str, int] = {
    "arabicmmlu": 40,
    "arabic_leaderboard_alghafa_light": 9,
    "arabic_leaderboard_acva_light": len(ACVA_CORE_SUBTASKS),
    "mmlu_pro": 14,
}

# Native LightEval task definitions for OALL v2 Arabic benchmarks
LIGHTEVAL_SUITE_TASK_MAP: Dict[str, Tuple[str, ...]] = {
    "ArabicMMLU": (
        "community|arabic_mmlu:All|0",
    ),
    "MadinahQA": (
        "community|madinah_qa:Arabic Language (General)|0",
        "community|madinah_qa:Arabic Language (Grammar)|0",
    ),
    "AlGhafa": (
        "community|alghafa:mcq_exams_test_ar|0",
        "community|alghafa:meta_ar_dialects|0",
        "community|alghafa:meta_ar_msa|0",
        "community|alghafa:multiple_choice_facts_truefalse_balanced_task|0",
        "community|alghafa:multiple_choice_grounded_statement_soqal_task|0",
        "community|alghafa:multiple_choice_grounded_statement_xglue_mlqa_task|0",
        "community|alghafa:multiple_choice_rating_sentiment_no_neutral_task|0",
        "community|alghafa:multiple_choice_rating_sentiment_task|0",
        "community|alghafa:multiple_choice_sentiment_task|0",
    ),
    "ACVA": (
        "community|acva:Saudi_Arabia|5",
        "community|acva:Arab_Empire|5",
        "community|acva:Arabic_Culture|5",
        "community|acva:Islam_Education|5",
        "community|acva:Islamic_law_system|5",
    ),
    "EXAMS-AR": (
        "community|arabic_exams|0",
    ),
    "AraTrust": (
        "community|aratrust:Trustfulness|0",
        "community|aratrust:MentalHealth|0",
        "community|aratrust:PhysicalHealth|0",
        "community|aratrust:Offensive|0",
        "community|aratrust:Ethics|0",
        "community|aratrust:Privacy|0",
        "community|aratrust:Unfairness|0",
        "community|aratrust:Illegal|0",
    ),
}


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
    """Ensure `"tie_word_embeddings": true` is written to `model_dir / "config.json"` when `lm_head.weight` is tied."""
    cfg_path = model_dir / "config.json"
    if not (model_dir.is_dir() and cfg_path.exists()):
        return False
    if _has_separate_lm_head_in_checkpoint(model_dir):
        return False
    try:
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        changed = False
        if cfg_data.get("tie_word_embeddings") is not True:
            cfg_data["tie_word_embeddings"] = True
            changed = True
        if isinstance(cfg_data.get("text_config"), dict):
            if cfg_data["text_config"].get("tie_word_embeddings") is not True:
                cfg_data["text_config"]["tie_word_embeddings"] = True
                changed = True
        if changed:
            cfg_path.write_text(
                json.dumps(cfg_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "Patched %s on disk with tie_word_embeddings=true (root + text_config).",
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
        return True
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
    """Render a clean GitHub-Flavored Markdown report using only standard ASCII punctuation."""
    lines: List[str] = [
        "# Gemma-4-Arabic Family Benchmark Evaluation Report",
        "",
        f"- **Model Evaluated:** `{report_data['model_path']}`",
        f"- **Evaluation Mode:** `{report_data['mode']}`",
        f"- **Zero-Forgetting Retention Gate:** `{'PASSED' if report_data['retention_gate']['gate_passed'] else 'FAILED'}` "
        f"(Min Retention Ratio: `{report_data['retention_gate']['min_observed_retention_ratio']:.4f}`, "
        f"Threshold: `{report_data['retention_gate']['required_min_retention_ratio']:.4f}`)",
        "",
        f"## 1. Primary Arabic and Saudi Benchmarks ({len(report_data['arabic_benchmarks'])} Suites)",
        "",
        "| Benchmark | Task (`lm_eval` / `lighteval`) | Metric | Shots | Base NLL | Tuned NLL | Baseline (%) | Evaluated (%) | Gain (% pts) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, item in report_data["arabic_benchmarks"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        gain = ev - base
        sign = "+" if gain >= 0 else ""
        b_nll = f"{float(item['base_nll']):.4f}" if item.get("base_nll") is not None else "N/A"
        t_nll = f"{float(item['tuned_nll']):.4f}" if item.get("tuned_nll") is not None else "N/A"
        lines.append(
            f"| **{name}** | `{item['lm_eval_task']}` | `{item['metric']}` | "
            f"{item['num_fewshot']} | `{b_nll}` | `{t_nll}` | {base:.2f} | **{ev:.2f}** | "
            f"`{sign}{gain:.2f}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Zero-Forgetting Retention Gate Anchors (4 Suites)",
            "",
            "| Anchor | Domain | Task (`lm_eval`) | Shots | Base NLL | Tuned NLL | Baseline (%) | Evaluated (%) | Delta (% pts) | Retention Ratio |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for name, item in report_data["retention_anchors"].items():
        base = float(item["baseline_score"])
        ev = float(item["evaluated_score"])
        delta = float(item["delta"])
        ratio = float(item["retention_ratio"])
        sign = "+" if delta >= 0 else ""
        b_nll = f"{float(item['base_nll']):.4f}" if item.get("base_nll") is not None else "N/A"
        t_nll = f"{float(item['tuned_nll']):.4f}" if item.get("tuned_nll") is not None else "N/A"
        lines.append(
            f"| **{name}** | `{item['domain']}` | `{item['lm_eval_task']}` | "
            f"{item['num_fewshot']} | `{b_nll}` | `{t_nll}` | {base:.2f} | **{ev:.2f}** | "
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
    """Extract `eval_score_pct`, `top1_acc_pct` (unnormalized `acc`), `nll` (`GoldNLL`), and `n_samples`."""
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

    task_lower = task_name.strip().lower()
    rescored_correct = 0
    rescored_total = 0

    if isinstance(samples_block, dict) and samples_block:
        for _subtask, sample_list in samples_block.items():
            if not isinstance(sample_list, list):
                continue
            doc_any_em: Dict[Any, bool] = {}
            for s_item in sample_list:
                if isinstance(s_item, dict) and s_item.get("doc_id") is not None:
                    did = s_item.get("doc_id")
                    item_em = any(
                        float(v or 0.0) >= 1.0
                        for k, v in s_item.items()
                        if str(k).startswith("exact_match") and isinstance(v, (int, float))
                    )
                    doc_any_em[did] = doc_any_em.get(did, False) or item_em
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

                native_em = (
                    doc_any_em.get(doc_id, False)
                    if doc_id is not None
                    else any(
                        float(v or 0.0) >= 1.0
                        for k, v in sample.items()
                        if str(k).startswith("exact_match") and isinstance(v, (int, float))
                    )
                )
                if "mmlu_pro" in task_lower or "arc_challenge_chat" in task_lower:
                    raw_resp = _first_resp_text(resps)
                    doc_obj = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
                    raw_gold = str(doc_obj.get("answer") or doc_obj.get("answerKey") or target or "").strip()
                    gold_letter = raw_gold.strip("()[]{}. \t\r\n*").upper()
                    gold_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}.get(gold_letter, gold_letter)
                    valid_set = "ABCDEFGHIJ" if "mmlu_pro" in task_lower else "ABCDE"
                    if gold_letter:
                        pred_letter = _extract_mcq_letter_with_doc(raw_resp, doc_obj, valid_letters=valid_set)
                        rescored_total += 1
                        if native_em or (pred_letter == gold_letter):
                            rescored_correct += 1
                elif model_key_prefix == "26b_a4b" and ("gsm8k" in task_lower or "mgsm" in task_lower):
                    raw_resp = _first_resp_text(resps)
                    doc_obj = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
                    raw_gold = str(target or doc_obj.get("answer") or "").strip()
                    gold_num = _extract_mgsm_ar_target_number(raw_gold)
                    if gold_num:
                        cleaned_resp = _truncate_math_continuation(raw_resp)
                        pred_num = ""
                        boxed_m = re.findall(r"\\boxed\s*\{([^{}]+)\}", cleaned_resp)
                        hash_m = re.findall(r"#{3,4}\s*\$?\s*\**(-?[\d,]+(?:\.\d+)?)", cleaned_resp)
                        ans_m = re.findall(
                            r"(?:(?i:(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=))|(?:الإجابة\s+النهائية|الإجابة\s+الصحيحة|الإجابة|الجواب\s+النهائي|الجواب|الناتج\s+النهائي|الناتج)\s*(?:هي|هو|:|=))\s*\**\s*\$?\s*\**(-?[\d,]+(?:\.\d+)?)",
                            cleaned_resp,
                        )
                        if hash_m:
                            pred_num = hash_m[0].replace(",", "")
                        elif boxed_m:
                            nums_in_box = re.findall(r"-?[\d,]+(?:\.\d+)?", boxed_m[-1])
                            if nums_in_box:
                                pred_num = nums_in_box[-1].replace(",", "")
                        elif ans_m:
                            pred_num = ans_m[-1].replace(",", "")
                        else:
                            pred_num = _extract_mgsm_ar_target_number(cleaned_resp)
                        rescored_total += 1
                        if native_em or _numbers_match(pred_num, gold_num):
                            rescored_correct += 1

    if rescored_total > 0 and (
        "mmlu_pro" in task_lower
        or "arc_challenge_chat" in task_lower
        or (model_key_prefix == "26b_a4b" and ("gsm8k" in task_lower or "mgsm" in task_lower))
    ):
        rescored_pct = round(100.0 * float(rescored_correct) / float(rescored_total), 2)
        eval_score_pct = max(eval_score_pct, rescored_pct)
        top1_acc_pct = max(top1_acc_pct, rescored_pct)

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
    """Extract percentage score (`0.0` to `100.0`) for OALL v2 Arabic benchmarks from `lighteval` results."""
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
        f"{prefix}:All|0",
        f"{prefix}:All:0",
        f"community|{prefix}|5",
        f"community:{prefix}:5",
        f"community|{prefix}|0",
        f"community:{prefix}:0",
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

    matched_metrics: List[Dict[str, Any]] = []
    for avg_key in avg_candidates:
        if avg_key in results_block and isinstance(results_block[avg_key], dict):
            matched_metrics = [results_block[avg_key]]
            break

    if not matched_metrics:
        for k, v in results_block.items():
            if prefix in str(k).lower() and "_average" in str(k).lower() and isinstance(v, dict):
                matched_metrics = [v]
                break

    if not matched_metrics:
        for k, v in results_block.items():
            if prefix in str(k).lower() and isinstance(v, dict):
                matched_metrics.append(v)

    if not matched_metrics:
        raise ValueError(f"Could not find '{prefix}' task scores in LightEval results: {list(results_block.keys())}")

    scores: List[float] = []
    for mdict in matched_metrics:
        found = False
        for mkey in ("acc_norm", "loglikelihood_acc_norm", "acc", "loglikelihood_acc", "accuracy"):
            if mkey in mdict and isinstance(mdict[mkey], (int, float)):
                val = float(mdict[mkey])
                scores.append(val * 100.0 if val <= 1.0 else val)
                found = True
                break
        if not found:
            for mkey, mval in mdict.items():
                if (
                    str(mkey).startswith("acc")
                    and not str(mkey).endswith("_stderr")
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

    mean_score = round(sum(scores) / len(scores), 2)
    return {
        "eval_score_pct": mean_score,
        "raw_signal_pct": mean_score,
        "top1_acc_pct": mean_score,
        "nll": None,
        "ppl": None,
        "n_samples": 0,
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
    """Run `MadinahQA` and `AraTrust` via `lighteval` in an isolated subprocess and return parsed metrics."""
    if not suite_names:
        return {}

    task_strings: List[str] = []
    for sname in suite_names:
        task_strings.extend(LIGHTEVAL_SUITE_TASK_MAP.get(sname, ()))
    if not task_strings:
        return {}

    model_dir = pathlib.Path(model_path)
    load_target_path = model_path
    if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
        adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        load_target_path = str(adapter_cfg.get("base_model_name_or_path") or model_path).strip()
    if pathlib.Path(load_target_path).is_dir():
        _patch_local_checkpoint_config_json(pathlib.Path(load_target_path))

    # Scale per-subtask max_samples across subtasks when --limit is provided
    per_subtask_samples: Optional[int] = None
    if limit is not None and limit > 0:
        per_subtask_samples = max(10, int(math.ceil(float(limit) / 2.0)))

    tasks_csv = ",".join(task_strings)
    logger.info(
        "[%s] Running LightEval on %s (backend=%s, max_samples=%s)...",
        tag,
        ", ".join(suite_names),
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
    le_max_len = min(int(max_seq_len), 2048)

    with tempfile.TemporaryDirectory(prefix="lighteval_run_") as tmp_dir:
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
            gpu_memory_utilization=0.85,
            max_model_length={int(le_max_len)},
            trust_remote_code=True,
        )
        pipeline = Pipeline(
            tasks={repr(tasks_csv)},
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
        )
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
        trust_remote_code=True,
    )
    pipeline = Pipeline(
        tasks={repr(tasks_csv)},
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    pipeline.evaluate()
    results_payload = pipeline.get_results()

pathlib.Path({repr(str(out_json_file))}).write_text(
    json.dumps({{"results": results_payload.get("results", {{}})}}, indent=2),
    encoding="utf-8",
)
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_json_file.exists():
            err_tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise RuntimeError(
                f"[{tag}] LightEval subprocess failed (exit={proc.returncode}): {err_tail}"
            )
        raw_lighteval = json.loads(out_json_file.read_text(encoding="utf-8"))

    parsed_out: Dict[str, Dict[str, Any]] = {}
    for sname in suite_names:
        metrics = parse_lighteval_results_json(raw_lighteval, sname)
        metrics["n_samples"] = int(limit or 0)
        parsed_out[sname] = metrics
        logger.info(
            "[%s] arabic/%s (lighteval) -> acc_norm=%.2f%% (N=%d)",
            tag,
            sname,
            metrics["eval_score_pct"],
            metrics["n_samples"],
        )
    return parsed_out


def _get_lm_eval_gen_kwargs(
    task_name: str,
    model_key_prefix: str = "e4b",
) -> Optional[Dict[str, Any]]:
    """Return standardized `generation_kwargs` overrides for `lm_eval` generative tasks (`mmlu_pro`, `gsm8k`, `humaneval`)."""
    clean = task_name.strip().lower()
    if clean == "mmlu_pro":
        if model_key_prefix == "26b_a4b":
            return {
                "max_gen_toks": 8192,
                "until": [
                    "<turn|>",
                    "<end_of_turn>",
                    "<eos>",
                    "</s>",
                    "<|im_end|>",
                ],
                "do_sample": False,
                "temperature": 0.0,
            }
        return {
            "max_gen_toks": 4096,
            "until": [
                "\nQuestion:",
                "<turn|>",
                "<end_of_turn>",
                "<eos>",
                "</s>",
                "<|im_end|>",
            ],
            "do_sample": False,
            "temperature": 0.0,
        }
    if clean == "gsm8k":
        if model_key_prefix == "26b_a4b":
            return {
                "max_gen_toks": 4096,
                "until": [
                    "\nQuestion:",
                    "<end_of_turn>",
                    "<eos>",
                    "</s>",
                    "<|im_end|>",
                ],
                "ignore_eos": True,
                "do_sample": False,
                "temperature": 0.0,
            }
        return {
            "max_gen_toks": 4096,
            "until": [
                "\nQuestion:",
                "<turn|>",
                "<end_of_turn>",
                "<eos>",
                "</s>",
                "<|im_end|>",
            ],
            "do_sample": False,
            "temperature": 0.0,
        }
    if clean == "humaneval" and model_key_prefix == "26b_a4b":
        return {
            "max_gen_toks": 4096,
            "until": [
                "\nif __name__",
                "<end_of_turn>",
                "<eos>",
                "</s>",
                "<|im_end|>",
            ],
            "ignore_eos": True,
            "do_sample": False,
            "temperature": 0.0,
        }
    return None


def _should_apply_chat_template_for_task(
    task_name: str,
    apply_chat_template: bool,
    model_key_prefix: str = "e4b",
) -> bool:
    """Return `True` (when `apply_chat_template` is enabled) for chat-instruct tasks (`mmlu_pro`) and `False` for standard completion/log-likelihood benchmarks."""
    clean = task_name.strip().lower()
    if model_key_prefix == "26b_a4b" and clean == "arc_challenge":
        return bool(apply_chat_template)
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
    """Configure `enable_thinking`, `think_end_token`, and `eos_token` per task on `VLLM` or `HFLM`.

    - `mmlu_pro` runs with `enable_thinking=True` (matching Google's official Gemma 4 Model Card table).
    - `arc_challenge`, `gsm8k`, and `humaneval` run with `apply_chat_template=False` and `enable_thinking=False`.
    - On `26b_a4b` when `apply_chat_template=False`, `tokenizer.eos_token` is set to `<eos>` (`1`) so
      `lm_eval`'s `eot_token_id` does not inject `<turn|>` (`106`) as a token-0 stop sequence.
    """
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
            lm_obj.think_end_token = None
        if hasattr(lm_obj, "chat_template_args"):
            lm_obj.chat_template_args = {}
    else:
        if hasattr(lm_obj, "enable_thinking"):
            lm_obj.enable_thinking = False
        if hasattr(lm_obj, "think_end_token"):
            lm_obj.think_end_token = None
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
    model_backend = "vllm" if use_vllm else "hf"
    model_args = (
        f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,"
        f"gpu_memory_utilization=0.85,max_model_len=8192"
        if use_vllm
        else f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,tie_word_embeddings=True"
    )
    gen_kwargs = _get_lm_eval_gen_kwargs(task_name)
    task_chat_tpl = _should_apply_chat_template_for_task(task_name, apply_chat_template)
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
    """Run the `lm_eval` tasks for `model_path` in an isolated child subprocess so the OS reclaims 100% of vLLM's 85% GPU memory pool upon completion."""
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
    _skip_lighteval=True,
)
pathlib.Path({repr(str(out_json_file))}).write_text(
    json.dumps(res, indent=2),
    encoding="utf-8",
)
"""
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            env=child_env,
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
    _skip_lighteval: bool = False,
    _preserved_arabic: Optional[Dict[str, Dict[str, Any]]] = None,
    _preserved_retention: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Evaluate a checkpoint using native `lm_eval` (MGSM-Ar + 4 anchors) and native `lighteval` (6 OALL v2 suites)."""
    eval_cfg = cfg.get("evaluation", {})
    ar_defs = cfg.get("arabic_benchmarks", {})
    ret_defs = cfg.get("retention_anchors", {})
    model_key_prefix = tag.split(":")[0] if ":" in tag else "e4b"
    model_specific_cfg = cfg.get("models", {}).get(model_key_prefix, {})
    default_bs = 8 if model_key_prefix == "26b_a4b" else int(eval_cfg.get("batch_size", 32))
    batch_size = int(model_specific_cfg.get("batch_size", default_bs))
    max_seq_len = int(eval_cfg.get("max_seq_length", 8192))
    if model_key_prefix == "26b_a4b":
        max_seq_len = max(max_seq_len, 12288)

    preserved_ar = _preserved_arabic or {}
    preserved_ret = _preserved_retention or {}
    lighteval_suites: List[str] = [
        sname
        for sname, spec in ar_defs.items()
        if str(spec.get("runner", "lm_eval")).strip().lower() == "lighteval"
        and sname in LIGHTEVAL_SUITE_TASK_MAP
        and sname not in preserved_ar
    ]
    pending_lm_eval_ar = {
        k: v
        for k, v in ar_defs.items()
        if k not in preserved_ar
        and not (
            str(v.get("runner", "lm_eval")).strip().lower() == "lighteval"
            and k in LIGHTEVAL_SUITE_TASK_MAP
        )
    }
    pending_lm_eval_ret = {
        k: v for k, v in ret_defs.items() if k not in preserved_ret
    }
    has_pending_lm_eval = bool(pending_lm_eval_ar or pending_lm_eval_ret)

    if not has_pending_lm_eval:
        out_only_le: Dict[str, Dict[str, Dict[str, Any]]] = {
            "arabic": dict(preserved_ar),
            "retention": dict(preserved_ret),
        }
        if lighteval_suites and not _skip_lighteval:
            le_metrics = run_lighteval_tasks_subprocess(
                model_path=model_path,
                suite_names=lighteval_suites,
                device=device,
                limit=limit,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                use_vllm=use_vllm,
                tag=tag,
            )
            for sname, smetrics in le_metrics.items():
                out_only_le["arabic"][sname] = smetrics
        return out_only_le

    if use_vllm and isinstance(model_path, str) and not _in_lm_eval_subprocess:
        sub_cfg = dict(cfg)
        sub_cfg["arabic_benchmarks"] = pending_lm_eval_ar
        sub_cfg["retention_anchors"] = pending_lm_eval_ret
        out = _run_lm_eval_suites_subprocess(
            model_path=model_path,
            device=device,
            cfg=sub_cfg,
            tag=tag,
            limit=limit,
            use_vllm=use_vllm,
            apply_chat_template=apply_chat_template,
        )
        for sname, smetrics in preserved_ar.items():
            out["arabic"][sname] = smetrics
        for sname, smetrics in preserved_ret.items():
            out["retention"][sname] = smetrics
        if lighteval_suites and not _skip_lighteval:
            le_metrics = run_lighteval_tasks_subprocess(
                model_path=model_path,
                suite_names=lighteval_suites,
                device=device,
                limit=limit,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                use_vllm=use_vllm,
                tag=tag,
            )
            for sname, smetrics in le_metrics.items():
                out["arabic"][sname] = smetrics
        return out

    import lm_eval
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

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
            vllm_add_bos = True if model_key_prefix == "26b_a4b" else None
            if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
                adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
                base_model_path = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
                lm_obj = VLLM(
                    pretrained=base_model_path,
                    lora_local_path=str(model_path),
                    dtype="bfloat16",
                    trust_remote_code=True,
                    gpu_memory_utilization=0.85,
                    max_model_len=max_seq_len,
                    batch_size="auto",
                    add_bos_token=vllm_add_bos,
                )
            else:
                lm_obj = VLLM(
                    pretrained=model_path,
                    dtype="bfloat16",
                    trust_remote_code=True,
                    gpu_memory_utilization=0.85,
                    max_model_len=max_seq_len,
                    batch_size="auto",
                    add_bos_token=vllm_add_bos,
                )
            # Default enable_thinking=False; _configure_lm_eval_thinking_for_task enables it per-task for mmlu_pro
            lm_obj.enable_thinking = False
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
            max_length=max_seq_len,
            trust_remote_code=True,
        )
        lm_obj.chat_template_args = {}
        _ensure_gemma4_tied_lm_head(lm_obj, tag=f"{tag}:hflm")

    task_manager = TaskManager()
    out: Dict[str, Dict[str, Dict[str, Any]]] = {
        "arabic": dict(preserved_ar),
        "retention": dict(preserved_ret),
    }

    suite_groups = (
        ("arabic", ar_defs),
        ("retention", ret_defs),
    )
    for cat, suite_defs in suite_groups:
        for suite_name, spec in suite_defs.items():
            if cat == "arabic" and suite_name in preserved_ar:
                continue
            if cat == "retention" and suite_name in preserved_ret:
                continue
            runner = str(spec.get("runner", "lm_eval")).strip().lower()
            if runner == "lighteval" and suite_name in LIGHTEVAL_SUITE_TASK_MAP:
                continue

            task_name = str(spec["lm_eval_task"])
            num_fewshot = int(spec.get("num_fewshot", 0))
            metric_hint = str(spec.get("metric", "acc_norm"))
            if model_key_prefix == "26b_a4b" and task_name.strip().lower() == "gsm8k":
                metric_hint = "exact_match,flexible-extract"
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
                fewshot_as_multiturn=False if model_key_prefix == "26b_a4b" else bool(task_chat_tpl),
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

    # Cleanly destroy the lm_eval vLLM/HFLM instance before starting LightEval on the same GPU
    _destroy_vllm_and_free_vram(lm_obj=lm_obj, model_obj=model_obj)
    lm_obj = None
    model_obj = None

    if lighteval_suites and not _skip_lighteval:
        le_metrics = run_lighteval_tasks_subprocess(
            model_path=model_path,
            suite_names=lighteval_suites,
            device=device,
            limit=limit,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            use_vllm=use_vllm,
            tag=tag,
        )
        for sname, smetrics in le_metrics.items():
            out["arabic"][sname] = smetrics

    return out


def _extract_preserved_lighteval_from_cache(
    cache_path: pathlib.Path,
    model_key: str = "e4b",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return `(preserved_base_arabic, preserved_tuned_arabic)` for validated Arabic suites in `cache_path`."""
    if not cache_path.exists():
        return ({}, {})
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        eval_ver = str(raw.get("eval_version") or "")
        base_ar = raw.get("base", {}).get("arabic", {})
        tuned_ar = raw.get("tuned", {}).get("arabic", {})
        if not isinstance(base_ar, dict) or not isinstance(tuned_ar, dict):
            return ({}, {})
        if eval_ver and eval_ver != "v28_linkedin_verified":
            candidate_suites: Tuple[str, ...] = (
                ("ArabicMMLU", "MadinahQA")
                + PRESERVED_LIGHTEVAL_SUITES
                + (("MGSM-Ar",) if model_key == "e4b" else ())
            )
        else:
            candidate_suites = PRESERVED_LIGHTEVAL_SUITES
        pres_base: Dict[str, Dict[str, Any]] = {}
        pres_tuned: Dict[str, Dict[str, Any]] = {}
        for sname in candidate_suites:
            b_entry = base_ar.get(sname)
            t_entry = tuned_ar.get(sname)
            if (
                isinstance(b_entry, dict)
                and isinstance(t_entry, dict)
                and float(b_entry.get("eval_score_pct") or 0.0) > 0.0
                and float(t_entry.get("eval_score_pct") or 0.0) > 0.0
            ):
                pres_base[sname] = b_entry
                pres_tuned[sname] = t_entry
        if pres_base and len(pres_base) == len(pres_tuned):
            logger.info(
                "Reusing %d validated LightEval suites (%s) from %s",
                len(pres_base),
                ", ".join(pres_base.keys()),
                cache_path.name,
            )
            return (pres_base, pres_tuned)
    except Exception as exc:
        logger.warning("Could not inspect %s for partial LightEval reuse: %s", cache_path, exc)
    return ({}, {})


def _extract_preserved_retention_from_cache(
    cache_path: pathlib.Path,
    model_key: str = "e4b",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return `(preserved_base_retention, preserved_tuned_retention)` for validated retention scores."""
    if not cache_path.exists():
        return ({}, {})
    if model_key == "26b_a4b":
        try:
            raw_check = json.loads(cache_path.read_text(encoding="utf-8"))
            if str(raw_check.get("eval_version") or "") != EVAL_CACHE_VERSION:
                return ({}, {})
        except Exception:
            return ({}, {})
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        eval_ver = str(raw.get("eval_version") or "")
        if eval_ver not in (
            "v28_linkedin_verified",
            "v29_zero_shot_and_non_thinking_fix",
            EVAL_CACHE_VERSION,
        ):
            return ({}, {})
        base_ret = raw.get("base", {}).get("retention", {})
        tuned_ret = raw.get("tuned", {}).get("retention", {})
        if not isinstance(base_ret, dict) or not isinstance(tuned_ret, dict):
            return ({}, {})
        pres_base: Dict[str, Dict[str, Any]] = {}
        pres_tuned: Dict[str, Dict[str, Any]] = {}
        candidate_ret = (
            ("MMLU-Pro", "GSM8K", "ARC-Challenge")
            if (model_key == "e4b" and eval_ver in ("v29_zero_shot_and_non_thinking_fix", EVAL_CACHE_VERSION))
            else ("MMLU-Pro",)
        )
        for sname in candidate_ret:
            b_entry = base_ret.get(sname)
            t_entry = tuned_ret.get(sname)
            if (
                isinstance(b_entry, dict)
                and isinstance(t_entry, dict)
                and float(b_entry.get("eval_score_pct") or 0.0) > 0.0
                and float(t_entry.get("eval_score_pct") or 0.0) > 0.0
            ):
                pres_base[sname] = b_entry
                pres_tuned[sname] = t_entry
        if pres_base and len(pres_base) == len(pres_tuned):
            logger.info(
                "Reusing validated retention anchor (%s) from %s",
                ", ".join(pres_base.keys()),
                cache_path.name,
            )
            return (pres_base, pres_tuned)
    except Exception as exc:
        logger.warning("Could not inspect %s for partial retention reuse: %s", cache_path, exc)
    return ({}, {})


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

    # Shorthand aliases
    shorthands = {
        "arc": ("retention", "ARC-Challenge"),
        "arcchallenge": ("retention", "ARC-Challenge"),
        "arcchallengechat": ("retention", "ARC-Challenge"),
        "humaneval": ("retention", "HumanEval"),
        "humanevalinstruct": ("retention", "HumanEval"),
        "mmlu": ("retention", "MMLU-Pro"),
        "mmlupro": ("retention", "MMLU-Pro"),
        "gsm8k": ("retention", "GSM8K"),
        "mgsm": ("arabic", "MGSM-Ar"),
        "mgsmar": ("arabic", "MGSM-Ar"),
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
    """Return `(base_ar, tuned_ar, base_ret, tuned_ret)` from `cache_path` (and fallback report JSONs) if present."""
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

    for fallback_name in ("benchmark_results.json", "family_benchmark_comparison.json"):
        fb_path = cache_path.parent / fallback_name
        if not fb_path.exists():
            continue
        try:
            fb_raw = json.loads(fb_path.read_text(encoding="utf-8"))
            model_block = fb_raw.get("models", {}).get(model_key) if isinstance(fb_raw.get("models"), dict) else None
            if not isinstance(model_block, dict):
                if fb_raw.get("model_key") == model_key:
                    model_block = fb_raw
            if not isinstance(model_block, dict) or model_block.get("mode") != "live_lm_eval":
                continue
            for sname, sdict in (model_block.get("arabic_benchmarks") or {}).items():
                if not isinstance(sdict, dict):
                    continue
                b_sc = float(sdict.get("baseline_score", 0.0) or 0.0)
                t_sc = float(sdict.get("evaluated_score", 0.0) or 0.0)
                if (b_sc > 0.0 or t_sc > 0.0) and (sname not in out_base_ar or sname not in out_tuned_ar):
                    out_base_ar[sname] = {
                        "eval_score_pct": b_sc,
                        "raw_signal_pct": b_sc,
                        "top1_acc_pct": b_sc,
                        "nll": sdict.get("base_nll"),
                        "ppl": None,
                        "n_samples": int(sdict.get("n_samples", 0) or 0),
                    }
                    out_tuned_ar[sname] = {
                        "eval_score_pct": t_sc,
                        "raw_signal_pct": t_sc,
                        "top1_acc_pct": t_sc,
                        "nll": sdict.get("tuned_nll"),
                        "ppl": None,
                        "n_samples": int(sdict.get("n_samples", 0) or 0),
                    }
            for sname, sdict in ((model_block.get("retention_gate") or {}).get("anchors") or {}).items():
                if not isinstance(sdict, dict):
                    continue
                b_sc = float(sdict.get("baseline_score", 0.0) or 0.0)
                t_sc = float(sdict.get("evaluated_score", 0.0) or 0.0)
                if (b_sc > 0.0 or t_sc > 0.0) and (sname not in out_base_ret or sname not in out_tuned_ret):
                    out_base_ret[sname] = {
                        "eval_score_pct": b_sc,
                        "raw_signal_pct": b_sc,
                        "top1_acc_pct": b_sc,
                        "nll": sdict.get("base_nll"),
                        "ppl": None,
                        "n_samples": int(sdict.get("n_samples", 0) or 0),
                    }
                    out_tuned_ret[sname] = {
                        "eval_score_pct": t_sc,
                        "raw_signal_pct": t_sc,
                        "top1_acc_pct": t_sc,
                        "nll": sdict.get("tuned_nll"),
                        "ppl": None,
                        "n_samples": int(sdict.get("n_samples", 0) or 0),
                    }
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
    """Run standardized `lm_eval` + `lighteval` GPU evaluation across all or selected benchmarks for `base` and `tuned`."""
    cache_path = PROJECT_ROOT / "outputs" / "eval_reports" / f"live_raw_{model_key}.json"
    if (load_cache_only or (use_cached_eval and not selected_tasks)) and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("eval_version") == EVAL_CACHE_VERSION
                and "base" in cached
                and "tuned" in cached
            ):
                logger.info("Loaded verified %s cache from %s", EVAL_CACHE_VERSION, cache_path)
                return cached
        except Exception as exc:
            logger.warning("Could not read cached live_raw_%s.json: %s", model_key, exc)

    if cfg is None:
        cfg = load_eval_config(PROJECT_ROOT / "configs" / "eval_config.yaml")

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
        ar_defs = cfg.get("arabic_benchmarks", {})
        ret_defs = cfg.get("retention_anchors", {})
        c_base_ar, c_tuned_ar, c_base_ret, c_tuned_ret = _extract_all_cached_suites(cache_path, model_key=model_key)
        pres_base_ar: Dict[str, Dict[str, Any]] = {}
        pres_tuned_ar: Dict[str, Dict[str, Any]] = {}
        pres_base_ret: Dict[str, Dict[str, Any]] = {}
        pres_tuned_ret: Dict[str, Dict[str, Any]] = {}

        for sname in ar_defs:
            if sname not in sel_ar:
                pres_base_ar[sname] = c_base_ar.get(sname, {"_skipped": True})
                pres_tuned_ar[sname] = c_tuned_ar.get(sname, {"_skipped": True})
        for sname in ret_defs:
            if sname not in sel_ret:
                pres_base_ret[sname] = c_base_ret.get(sname, {"_skipped": True})
                pres_tuned_ret[sname] = c_tuned_ret.get(sname, {"_skipped": True})

        logger.info(
            "[%s] --tasks filter active -> evaluating ONLY arabic=%s, retention=%s",
            model_key,
            sel_ar,
            sel_ret,
        )
    else:
        pres_base_ar, pres_tuned_ar = _extract_preserved_lighteval_from_cache(
            cache_path, model_key=model_key
        )
        pres_base_ret, pres_tuned_ret = _extract_preserved_retention_from_cache(
            cache_path, model_key=model_key
        )

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
    )
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
    )

    # Remove internal _skipped placeholders before persisting cache
    for metrics_dict in (base_metrics, tuned_metrics):
        for group_key in ("arabic", "retention"):
            grp = metrics_dict.get(group_key, {})
            for k in list(grp.keys()):
                if isinstance(grp[k], dict) and grp[k].get("_skipped"):
                    del grp[k]

    payload = {
        "eval_version": EVAL_CACHE_VERSION,
        "model_key": model_key,
        "base_model_id": base_model_id,
        "tuned_model_path": tuned_model_path,
        "device": use_dev,
        "limit": limit,
        "base": base_metrics,
        "tuned": tuned_metrics,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %s live GPU evaluation metrics to %s", EVAL_CACHE_VERSION, cache_path)
    return payload


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
    """Run all 7 Arabic benchmarks (`lm_eval` + `lighteval`) and 4 retention anchors (`lm_eval`)."""
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
            base_score = round(float(b_item["eval_score_pct"]), 2)
            evaluated_score = round(float(t_item["eval_score_pct"]), 2)
        else:
            if eval_base or base_score == 0.0 or no_reference_fallback:
                try:
                    base_score = run_lm_eval_task(
                        model_path=str(base_model_id),
                        task_name=str(spec["lm_eval_task"]),
                        num_fewshot=int(spec.get("num_fewshot", 0)),
                        batch_size=int(eval_cfg.get("batch_size", 16)),
                        device=target_device,
                        limit=limit,
                        metric_hint=str(spec.get("metric", "acc_norm")),
                        use_vllm=use_vllm,
                        apply_chat_template=apply_chat_template,
                    )
                except Exception as exc:
                    if no_reference_fallback:
                        raise RuntimeError(
                            f"Base task '{name}' ({spec['lm_eval_task']}) failed and "
                            f"--no-reference-fallback is active: {exc}"
                        ) from exc
                    logger.warning("Base eval fallback for %s (%s): %s", name, spec["lm_eval_task"], exc)
            try:
                evaluated_score = run_lm_eval_task(
                    model_path=model_path,
                    task_name=str(spec["lm_eval_task"]),
                    num_fewshot=int(spec.get("num_fewshot", 0)),
                    batch_size=int(eval_cfg.get("batch_size", 16)),
                    device=target_device,
                    limit=limit,
                    metric_hint=str(spec.get("metric", "acc_norm")),
                    use_vllm=use_vllm,
                    apply_chat_template=apply_chat_template,
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
                    base_score = run_lm_eval_task(
                        model_path=str(base_model_id),
                        task_name=str(spec["lm_eval_task"]),
                        num_fewshot=int(spec.get("num_fewshot", 0)),
                        batch_size=int(eval_cfg.get("batch_size", 16)),
                        device=target_device,
                        limit=limit,
                        metric_hint=str(spec.get("metric", "acc_norm")),
                        use_vllm=use_vllm,
                        apply_chat_template=apply_chat_template,
                    )
                except Exception as exc:
                    if no_reference_fallback:
                        raise RuntimeError(
                            f"Base retention anchor '{name}' ({spec['lm_eval_task']}) failed and "
                            f"--no-reference-fallback is active: {exc}"
                        ) from exc
                    logger.warning("Base anchor eval fallback for %s: %s", name, exc)
            try:
                evaluated_score = run_lm_eval_task(
                    model_path=model_path,
                    task_name=str(spec["lm_eval_task"]),
                    num_fewshot=int(spec.get("num_fewshot", 0)),
                    batch_size=int(eval_cfg.get("batch_size", 16)),
                    device=target_device,
                    limit=limit,
                    metric_hint=str(spec.get("metric", "acc_norm")),
                    use_vllm=use_vllm,
                    apply_chat_template=apply_chat_template,
                )
            except Exception as exc:
                if no_reference_fallback:
                    raise RuntimeError(
                        f"Tuned retention anchor '{name}' ({spec['lm_eval_task']}) failed and "
                        f"--no-reference-fallback is active: {exc}"
                    ) from exc
                logger.warning(
                    "Live lm_eval anchor '%s' (%s) fallback to baseline score: %s",
                    name,
                    spec["lm_eval_task"],
                    exc,
                )
                evaluated_score = base_score

        anchor_entry: Dict[str, Any] = {
            "domain": spec["domain"],
            "lm_eval_task": spec["lm_eval_task"],
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
    """Render a side-by-side Markdown report comparing E4B (Dense) and 26B-A4B (MoE)."""
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
        "## 1. Arabic & Saudi Benchmarks (4-Column Family Progression)",
        "",
        "| Benchmark | Metric | Shots | E4B Base (%) | E4B Arabic-It (%) | E4B NLL (Base -> Tuned) | 26B-A4B Base (%) | 26B-A4B Arabic-It (%) | 26B-A4B NLL (Base -> Tuned) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, e4b_item in e4b_results["arabic_benchmarks"].items():
        moe_item = moe_results["arabic_benchmarks"].get(name, e4b_item)
        e4b_nll_str = (
            f"{float(e4b_item['base_nll']):.2f} -> {float(e4b_item['tuned_nll']):.2f}"
            if e4b_item.get("base_nll") is not None and e4b_item.get("tuned_nll") is not None
            else "N/A"
        )
        moe_nll_str = (
            f"{float(moe_item['base_nll']):.2f} -> {float(moe_item['tuned_nll']):.2f}"
            if moe_item.get("base_nll") is not None and moe_item.get("tuned_nll") is not None
            else "N/A"
        )
        lines.append(
            f"| **{name}** | `{e4b_item['metric']}` | {e4b_item['num_fewshot']} | "
            f"{float(e4b_item['baseline_score']):.2f} | **{float(e4b_item['evaluated_score']):.2f}** | `{e4b_nll_str}` | "
            f"{float(moe_item['baseline_score']):.2f} | **{float(moe_item['evaluated_score']):.2f}** | `{moe_nll_str}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Zero-Forgetting Retention Anchors (English / Math / Code)",
            "",
            "| Anchor | Domain | Shots | E4B Base (%) | E4B Arabic-It (%) | E4B Ratio | 26B-A4B Base (%) | 26B-A4B Arabic-It (%) | 26B-A4B Ratio |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name, e4b_item in e4b_results["retention_anchors"].items():
        moe_item = moe_results["retention_anchors"].get(name, e4b_item)
        e4b_ratio = float(e4b_item.get("retention_ratio", 1.0))
        moe_ratio = float(moe_item.get("retention_ratio", 1.0))
        lines.append(
            f"| **{name}** | `{e4b_item['domain']}` | {e4b_item.get('num_fewshot', 0)} | "
            f"{float(e4b_item['baseline_score']):.2f} | **{float(e4b_item['evaluated_score']):.2f}** | `{e4b_ratio:.4f}` | "
            f"{float(moe_item['baseline_score']):.2f} | **{float(moe_item['evaluated_score']):.2f}** | `{moe_ratio:.4f}` |"
        )

    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the benchmark evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run the 7 Arabic/Saudi benchmarks and 4 English/Math/Code retention anchors via lm_eval and lighteval."
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
        help="Load cached live_raw_*.json if present and matching the current eval_version.",
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
        help="Use vLLM backend for lm_eval and lighteval (default: True).",
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
        help="Optional sample limit per benchmark suite.",
    )
    parser.add_argument(
        "--tasks",
        "--benchmarks",
        dest="tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated list of specific benchmarks to evaluate on Base and Tuned "
            "(e.g., 'GSM8K,HumanEval,ARC-Challenge,MGSM-Ar,ArabicMMLU,MadinahQA' or 'HumanEval'). "
            "Unselected benchmarks are automatically loaded from outputs/eval_reports/live_raw_*.json if present."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all 7 Arabic benchmarks, 4 retention anchors, and Markdown report generation.",
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
