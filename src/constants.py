"""Central constants, verified dataset identifiers, and Gemma 4 native token invariants."""

import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def load_dotenv_if_present(dotenv_path: Optional[pathlib.Path] = None) -> None:
    """Load key=value pairs (such as HF_TOKEN) from `.env` into `os.environ` if not already set."""
    if dotenv_path is None:
        dotenv_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ and val:
                os.environ[key] = val
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model Identifiers (Dense 4.5B E4B + Sparse 25.2B-A4B MoE Family)
# ---------------------------------------------------------------------------
# 1. Gemma-4-E4B-it (4.5B Total / 4.0B Active Dense Model)
PRIMARY_MODEL_ID: str = "unsloth/gemma-4-E4B-it"
BASE_MODEL_ID: str = PRIMARY_MODEL_ID
FALLBACK_MODEL_ID: str = "google/gemma-4-E4B-it"
GOOGLE_REFERENCE_MODEL_ID: str = "google/gemma-4-E4B-it"

# 2. Gemma-4-26B-A4B-it (25.2B Total / 3.8B Active Sparse MoE: 128 Routed Experts + 1 Shared)
PRIMARY_MODEL_ID_26B: str = "unsloth/gemma-4-26B-A4B-it"
FALLBACK_MODEL_ID_26B: str = "google/gemma-4-26B-A4B-it"
MOE_ROUTER_MODULE_PATTERNS: Tuple[str, ...] = ("router", ".gate.")
MOE_ROUTER_KEYWORDS: Tuple[str, ...] = MOE_ROUTER_MODULE_PATTERNS

# Published Hugging Face Dataset Repositories (for fast pull on GPU nodes)
PUBLISHED_SFT_DATASET_REPO: str = "noman-asif/Arabic-Reasoning-Instruct-120k"
PUBLISHED_DPO_DATASET_REPO: str = "noman-asif/Arabic-Preference-DPO-20k"

# ---------------------------------------------------------------------------
# Native Gemma 4 Chat & Reasoning Special Tokens (Verified Against Tokenizer)
#   <bos>       = 2
#   <eos>       = 1
#   <|turn>     = 105
#   <turn|>     = 106  (generation eos_token in instruct checkpoints)
#   <|think|>   = 98   (injected inside <|turn>system\n<|think|>\n when enable_thinking=True)
#   <|channel>  = 100  (opens <|channel>thought\n inside <|turn>model\n)
#   <channel|>  = 101  (closes reasoning channel before final answer)
# NOTE: Legacy Gemma 2/3 `<start_of_turn>` and `<end_of_turn>` do NOT exist in
# Gemma 4's vocabulary (they map to <unk>=3) and must NEVER appear in data!
# ---------------------------------------------------------------------------
BOS_TOKEN: str = "<bos>"
EOS_TOKEN: str = "<eos>"
TURN_START_TOKEN: str = "<|turn>"
TURN_END_TOKEN: str = "<turn|>"
THINK_TOKEN: str = "<|think|>"
CHANNEL_START_TOKEN: str = "<|channel>"
CHANNEL_END_TOKEN: str = "<channel|>"
THOUGHT_CHANNEL_OPEN: str = "<|channel>thought\n"
THOUGHT_CHANNEL_CLOSE: str = "\n<channel|>"
EMPTY_THOUGHT_CHANNEL: str = "<|channel>thought\n<channel|>"
CHANNEL_THOUGHT_PREFIX: str = THOUGHT_CHANNEL_OPEN
EMPTY_THINKING_BLOCK: str = EMPTY_THOUGHT_CHANNEL

# Forbidden legacy Gemma 2/3 / malformed markers verified to have 0 occurrences
FORBIDDEN_LEGACY_MARKERS: Tuple[str, ...] = (
    "<start_of_turn>",
    "<end_of_turn>",
    "<|turn|>",
    "<|think|>\n<|think|>",
)

DEFAULT_ARABIC_SYSTEM_PROMPT: str = (
    "أنت مساعد ذكي متقن للغة العربية الفصحى واللهجات السعودية، دقيق في الاستدلال والأنظمة."
)

ARABIC_SYSTEM_PROMPT_POOL: Tuple[str, ...] = (
    "أنت مساعد ذكي متقن للغة العربية الفصحى واللهجات السعودية، دقيق في الاستدلال والأنظمة.",
    "أجب بدقة ووضوح باللغة العربية السليمة مع تنظيم الأفكار خطوة بخطوة عند الحاجة.",
    "أنت خبير لغوي وعلمي باللغة العربية؛ قدّم إجابات موثوقة ومباشرة.",
    "أنت مستشار ذكي يتحدث العربية الفصحى واللهجة السعودية بطلاقة ويقدم إجابات دقيقة.",
    "قدّم إجابة عربية واضحة ومكتملة ومبنية على الاستنتاج السليم.",
    "أنت مساعد ذكي متخصص في المعرفة العامة والعلوم والأنظمة باللغة العربية.",
)

ENGLISH_SYSTEM_PROMPT_POOL: Tuple[str, ...] = (
    "You are a helpful, precise assistant skilled in reasoning, mathematics, science, and software engineering.",
    "Provide clear, accurate, well-structured solutions.",
    "You are an expert problem solver. Reason carefully and state the final answer clearly.",
    "Answer accurately and concisely.",
)

# Fraction of SFT samples that include an explicit system message (70% omit synthetic
# system prompts so 0-shot and chat benchmarks without system turns match training).
SYSTEM_PROMPT_INJECTION_PROB: float = 0.30

# ---------------------------------------------------------------------------
# Architecture & Memory Optimization Constants
# ---------------------------------------------------------------------------
PLE_MODULE_PATTERN: str = "embed_tokens_per_layer"
MULTIMODAL_TOWER_PATTERNS: Tuple[str, ...] = ("audio_tower", "vision_tower")
FROZEN_EMBEDDING_PATTERNS: Tuple[str, ...] = (
    "embed_tokens",
    "embed_tokens_per_layer",
    "lm_head",
)

# Bare leaf projection names for Dense LoRA (NEVER prefix with "mlp." because Unsloth
# regex expansion would produce `mlp.*?\.mlp\.gate_proj` and match 0 layers).
# For 26B-A4B MoE, pass `target_modules=None` to Unsloth `FastModel.get_peft_model`
# with `finetune_attention_modules=True, finetune_mlp_modules=True` so Unsloth attaches
# both attention/shared-MLP LoRAs AND 3D `ParamWrapper` LoRAs on all 128 routed experts
# (`target_parameters=["experts.gate_up_proj", "experts.down_proj"]`) while freezing `router`.
LORA_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LORA_RANK: int = 32
LORA_ALPHA: int = 32
LORA_RANK_E4B: int = 32
LORA_ALPHA_E4B: int = 32
LORA_RANK_26B: int = 16
LORA_ALPHA_26B: int = 16
LORA_DROPOUT: float = 0.0
LORA_BIAS: str = "none"
USE_RSLORA: bool = False
DEFAULT_MAX_SEQ_LENGTH: int = 4096
LORA_EXCLUDE_MODULES: Tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    "embed_tokens_per_layer",
    "audio_tower",
    "vision_tower",
    "multi_modal_projector",
    "router",
)

PLE_VOCAB_SIZE: int = 262_144
PLE_PER_LAYER_DIM: int = 256
NUM_TRANSFORMER_LAYERS: int = 42
PLE_TOTAL_DIM: int = PLE_PER_LAYER_DIM * NUM_TRANSFORMER_LAYERS  # 10,752

# ---------------------------------------------------------------------------
# Stage 1 & Stage 2 Mixing, Sequence Length, and Merge Constants
# ---------------------------------------------------------------------------
SFT_ARABIC_RATIO: float = 0.75
SFT_REPLAY_RATIO: float = 0.25
SFT_MAX_TOKEN_LENGTH_FILTER: int = 3800

DPO_BETA: float = 0.1
DPO_LEARNING_RATE: float = 5.0e-6
DPO_MAX_SEQ_LENGTH: int = 3072
DPO_MAX_PROMPT_LENGTH: int = 1536
DPO_MAX_TOKEN_LENGTH_FILTER: int = 2800
DPO_MAX_PROMPT_TOKEN_FILTER: int = 1400
DPO_TRUST_REGION_ALPHA: float = 0.6  # Legacy constant kept for backwards compatibility

DEFAULT_MERGE_LAMBDA: float = 1.0
DEFAULT_TIES_LAMBDA: float = 1.0
DEFAULT_TIES_DENSITY: float = 1.0
TIES_LAMBDA_SWEEP_CANDIDATES: Tuple[float, ...] = (0.85, 0.90, 1.00)


@dataclass(frozen=True)
class DatasetSpec:
    """Verified Hugging Face dataset specification."""

    key: str
    repo_id: str
    configs: Tuple[Optional[str], ...]
    default_split: str
    stage: str
    role: str
    target_samples: int
    enable_thinking: bool = False
    fallback_repo_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Curated V2 Dataset Registry (100% Public `gated=False`, High-Signal, Verified)
# Stage 1 SFT: 90,000 Arabic (75.0%) + 30,000 Replay (25.0%) = 120,000 Total
# ---------------------------------------------------------------------------
SFT_ARABIC_DATASETS: Dict[str, DatasetSpec] = {
    "smolkalam_arabic_science": DatasetSpec(
        key="smolkalam_arabic_science",
        repo_id="AdaMLLab/smolkalam-arabic-conversational-sft",
        configs=("Mixture_of_Thoughts_science_no_think",),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=15_000,
        enable_thinking=False,
    ),
    "smolkalam_arabic_reasoning_cot": DatasetSpec(
        key="smolkalam_arabic_reasoning_cot",
        repo_id="AdaMLLab/smolkalam-arabic-conversational-sft",
        configs=("OpenThoughts3_50K",),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=12_000,
        enable_thinking=True,
    ),
    "smolkalam_arabic_openhermes": DatasetSpec(
        key="smolkalam_arabic_openhermes",
        repo_id="AdaMLLab/smolkalam-arabic-conversational-sft",
        configs=("OpenHermes_2.5_no_think",),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=15_000,
        enable_thinking=False,
    ),
    "arabic_math_reasoning_synth": DatasetSpec(
        key="arabic_math_reasoning_synth",
        repo_id="oddadmix/arabic-math-reasoning-synth",
        configs=(None,),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=12_000,
        enable_thinking=True,
    ),
    "alpaca_gpt4_arabic": DatasetSpec(
        key="alpaca_gpt4_arabic",
        repo_id="FreedomIntelligence/alpaca-gpt4-arabic",
        configs=(None,),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=12_000,
        enable_thinking=False,
    ),
    "medical_reasoning_arabic": DatasetSpec(
        key="medical_reasoning_arabic",
        repo_id="Gheras/Medical_Reasoning_Arabic_Dataset",
        configs=(None,),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=6_000,
        enable_thinking=True,
    ),
    "aya_dataset_arabic": DatasetSpec(
        key="aya_dataset_arabic",
        repo_id="CohereLabs/aya_dataset",
        configs=(None,),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=6_000,
        enable_thinking=False,
    ),
    "voho_saudi_dialogues": DatasetSpec(
        key="voho_saudi_dialogues",
        repo_id="VohoAI/voho-saudi-dialogues",
        configs=(None,),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=6_000,
        enable_thinking=False,
    ),
    "saudi_arabic_laws": DatasetSpec(
        key="saudi_arabic_laws",
        repo_id="SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus",
        configs=("moj", "boe"),
        default_split="train",
        stage="sft",
        role="arabic",
        target_samples=6_000,
        enable_thinking=False,
        fallback_repo_id=None,
    ),
}

SFT_REPLAY_DATASETS: Dict[str, DatasetSpec] = {
    "opencode_reasoning_cot": DatasetSpec(
        key="opencode_reasoning_cot",
        repo_id="nvidia/OpenCodeReasoning",
        configs=("split_0",),
        default_split="split_0",
        stage="sft",
        role="replay",
        target_samples=7_500,
        enable_thinking=True,
    ),
    "codeforces_python_cot": DatasetSpec(
        key="codeforces_python_cot",
        repo_id="open-r1/codeforces-cots",
        configs=("solutions_py_decontaminated",),
        default_split="train",
        stage="sft",
        role="replay",
        target_samples=2_500,
        enable_thinking=True,
    ),
    "codefeedback_instruction": DatasetSpec(
        key="codefeedback_instruction",
        repo_id="m-a-p/CodeFeedback-Filtered-Instruction",
        configs=(None,),
        default_split="train",
        stage="sft",
        role="replay",
        target_samples=2_000,
        enable_thinking=False,
    ),
    "openmath_instruct2_cot": DatasetSpec(
        key="openmath_instruct2_cot",
        repo_id="nvidia/OpenMathInstruct-2",
        configs=(None,),
        default_split="train_1M",
        stage="sft",
        role="replay",
        target_samples=7_500,
        enable_thinking=True,
    ),
    "openscience_reasoning2_cot": DatasetSpec(
        key="openscience_reasoning2_cot",
        repo_id="nvidia/OpenScienceReasoning-2",
        configs=(None,),
        default_split="train",
        stage="sft",
        role="replay",
        target_samples=6_000,
        enable_thinking=True,
    ),
    "ultrachat_english_replay": DatasetSpec(
        key="ultrachat_english_replay",
        repo_id="HuggingFaceH4/ultrachat_200k",
        configs=(None,),
        default_split="train_sft",
        stage="sft",
        role="replay",
        target_samples=4_500,
        enable_thinking=False,
    ),
}

# ---------------------------------------------------------------------------
# Stage 2 DPO Dataset Registry: 17,000 Arabic/Saudi (85%) + 3,000 English (15%) = 20,000
# ---------------------------------------------------------------------------
DPO_DATASETS: Dict[str, DatasetSpec] = {
    "arabic_dialect_dpo_saudi": DatasetSpec(
        key="arabic_dialect_dpo_saudi",
        repo_id="HeshamHaroon/arabic-dialect-dpo",
        configs=("saudi",),
        default_split="train",
        stage="dpo",
        role="preference_saudi",
        target_samples=4_000,
        enable_thinking=False,
    ),
    "arabic_orpo_dpo_mix_40k": DatasetSpec(
        key="arabic_orpo_dpo_mix_40k",
        repo_id="arcee-globe/arabic-orpo-dpo-mix-40k-filtered",
        configs=(None,),
        default_split="train",
        stage="dpo",
        role="preference_msa",
        target_samples=13_000,
        enable_thinking=False,
    ),
    "ultrafeedback_english_dpo": DatasetSpec(
        key="ultrafeedback_english_dpo",
        repo_id="argilla/ultrafeedback-binarized-preferences-cleaned",
        configs=(None,),
        default_split="train",
        stage="dpo",
        role="preference_english_retention",
        target_samples=3_000,
        enable_thinking=False,
    ),
}

ALL_CURATED_DATASET_REPOS: Tuple[str, ...] = tuple(
    dict.fromkeys(
        [spec.repo_id for spec in SFT_ARABIC_DATASETS.values()]
        + [spec.repo_id for spec in SFT_REPLAY_DATASETS.values()]
        + [spec.repo_id for spec in DPO_DATASETS.values()]
    )
)
ALL_DATASET_REPOS: Tuple[str, ...] = ALL_CURATED_DATASET_REPOS
ALL_9_DATASET_REPOS: Tuple[str, ...] = ALL_CURATED_DATASET_REPOS

ARABIC_DIALECT_DPO_REPO: str = "HeshamHaroon/arabic-dialect-dpo"
ARABIC_DIALECT_REQUIRED_CONFIG: str = "saudi"
