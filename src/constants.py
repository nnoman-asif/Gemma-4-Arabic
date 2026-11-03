"""Centralized constants, dataset registries, special tokens, and Arabic/Saudi system prompts."""

from dataclasses import dataclass
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Base Model and Dual-Model Family Constants (E4B Dense + 26B-A4B MoE)
# ---------------------------------------------------------------------------
BASE_MODEL_ID: str = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
PRIMARY_MODEL_ID: str = BASE_MODEL_ID
GOOGLE_REFERENCE_MODEL_ID: str = "google/gemma-4-E4B-it"

PRIMARY_MODEL_ID_26B: str = "unsloth/gemma-4-26B-A4B-it"
FALLBACK_MODEL_ID_26B: str = "google/gemma-4-26B-A4B-it"

TOTAL_PARAMETERS: int = 7_996_156_490
EFFECTIVE_PARAMETERS: int = 4_500_000_000
MOE_TOTAL_PARAMETERS: int = 26_000_000_000
MOE_ACTIVE_PARAMETERS: int = 3_800_000_000
MOE_NUM_EXPERTS: int = 128
MOE_TOP_K_EXPERTS: int = 8

NUM_LAYERS: int = 42
HIDDEN_SIZE: int = 2560
INTERMEDIATE_SIZE: int = 10240
VOCAB_SIZE: int = 262_144
ARABIC_VOCAB_TOKENS: int = 8_460

DEFAULT_MAX_SEQ_LENGTH: int = 2048
A100_MAX_SEQ_LENGTH: int = 4096
DPO_MAX_SEQ_LENGTH: int = 1536
DPO_MAX_PROMPT_LENGTH: int = 768
TARGET_VRAM_BUDGET_GB: float = 8.0
A100_VRAM_BUDGET_GB: float = 80.0

MGSM_AR_TASK: str = "mgsm_native_cot_ar"

# Per-Layer Embeddings (PLE), Multimodal, and MoE Router Patterns
PLE_MODULE_PATTERN: str = "embed_tokens_per_layer"
PLE_EMBEDDING_SHAPE: Tuple[int, int] = (262_144, 10_752)
PLE_PER_LAYER_DIM: int = 256
PLE_EXPECTED_VRAM_GB: float = 5.637

MULTIMODAL_TOWER_PATTERNS: Tuple[str, ...] = (
    "audio_tower",
    "vision_tower",
    "embed_audio",
    "embed_vision",
)
MOE_ROUTER_KEYWORDS: Tuple[str, ...] = (
    "router",
    ".gate.",
)
AUDIO_TOWER_VRAM_GB: float = 0.618
VISION_TOWER_VRAM_GB: float = 0.339

# rsLoRA Linear Target and Exclusion Projections
LORA_RANK: int = 32
LORA_ALPHA: int = 64
LORA_DROPOUT: float = 0.0
LORA_BIAS: str = "none"
USE_RSLORA: bool = True

LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

LORA_EXCLUDE_MODULES: Tuple[str, ...] = (
    "vision_tower",
    "multi_modal_projector",
    "audio_tower",
    "embed_vision",
    "embed_audio",
    "embed_tokens_per_layer",
)


# ---------------------------------------------------------------------------
# Gemma 4 Special Tokens and Chat Template Constants
# ---------------------------------------------------------------------------
BOS_TOKEN: str = "<bos>"
EOS_TOKEN: str = "<eos>"
PAD_TOKEN: str = "<pad>"
START_OF_TURN_TOKEN: str = "<start_of_turn>"
END_OF_TURN_TOKEN: str = "<end_of_turn>"
THINK_TOKEN: str = "<|think|>"

# Mandatory 4-token empty thinking header emitted when enable_thinking=False
EMPTY_THINKING_BLOCK: str = "<|think|>\n<|think|>\n"
ASSISTANT_TURN_PREFIX: str = "<start_of_turn>model\n"
USER_TURN_PREFIX: str = "<start_of_turn>user\n"


# ---------------------------------------------------------------------------
# All 9 Dataset Repository IDs, Configs, and Splits
# ---------------------------------------------------------------------------
VOHO_SAUDI_DIALOGUES_REPO: str = "VohoAI/voho-saudi-dialogues"
SAUDI_ARABIC_LAWS_REPO: str = "SaudiArabicLaws/saudi-arabic-laws-and-regulations-corpus"
ACEGPT_V2_TUNING_REPO: str = "FreedomIntelligence/AceGPT-v2-tuning"
ARABIC_MATH_SFT_REPO: str = "Omartificial-Intelligence-Space/Arabic-Math-SFT"
AYA_COLLECTION_REPO: str = "CohereForAI/aya_collection_language_split"

ULTRACHAT_200K_REPO: str = "HuggingFaceH4/ultrachat_200k"
OPENMATH_INSTRUCT_2_REPO: str = "nvidia/OpenMathInstruct-2"

ARABIC_DIALECT_DPO_REPO: str = "HeshamHaroon/arabic-dialect-dpo"
ARABIC_ORPO_DPO_MIX_REPO: str = "arcee-globe/arabic-orpo-dpo-mix-40k-filtered"


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable specification for a training or alignment dataset source."""

    key: str
    repo_id: str
    config_names: Tuple[str, ...]
    split: str
    stage: str
    enable_thinking: bool
    weight: float
    drop_columns: Tuple[str, ...] = ()


SFT_ARABIC_DATASETS: Dict[str, DatasetSpec] = {
    "voho_saudi_dialogues": DatasetSpec(
        key="voho_saudi_dialogues",
        repo_id=VOHO_SAUDI_DIALOGUES_REPO,
        config_names=("default",),
        split="train",
        stage="sft_arabic",
        enable_thinking=False,
        weight=0.25,
    ),
    "saudi_arabic_laws": DatasetSpec(
        key="saudi_arabic_laws",
        repo_id=SAUDI_ARABIC_LAWS_REPO,
        config_names=("moj", "boe"),
        split="train",
        stage="sft_arabic",
        enable_thinking=False,
        weight=0.15,
    ),
    "acegpt_v2_tuning": DatasetSpec(
        key="acegpt_v2_tuning",
        repo_id=ACEGPT_V2_TUNING_REPO,
        config_names=("default",),
        split="train",
        stage="sft_arabic",
        enable_thinking=False,
        weight=0.25,
    ),
    "arabic_math_sft": DatasetSpec(
        key="arabic_math_sft",
        repo_id=ARABIC_MATH_SFT_REPO,
        config_names=("default",),
        split="train",
        stage="sft_arabic",
        enable_thinking=True,
        weight=0.10,
        drop_columns=("image",),
    ),
    "aya_arabic_instruct": DatasetSpec(
        key="aya_arabic_instruct",
        repo_id=AYA_COLLECTION_REPO,
        config_names=("arabic",),
        split="train",
        stage="sft_arabic",
        enable_thinking=False,
        weight=0.25,
    ),
}

SFT_REPLAY_DATASETS: Dict[str, DatasetSpec] = {
    "ultrachat_200k": DatasetSpec(
        key="ultrachat_200k",
        repo_id=ULTRACHAT_200K_REPO,
        config_names=("default",),
        split="train_sft",
        stage="sft_replay",
        enable_thinking=False,
        weight=0.60,
    ),
    "openmath_instruct_2": DatasetSpec(
        key="openmath_instruct_2",
        repo_id=OPENMATH_INSTRUCT_2_REPO,
        config_names=("default",),
        split="train_1M",
        stage="sft_replay",
        enable_thinking=True,
        weight=0.40,
    ),
}

DPO_DATASETS: Dict[str, DatasetSpec] = {
    "arabic_dialect_dpo_saudi": DatasetSpec(
        key="arabic_dialect_dpo_saudi",
        repo_id=ARABIC_DIALECT_DPO_REPO,
        config_names=("saudi",),
        split="train",
        stage="dpo_preference",
        enable_thinking=False,
        weight=0.35,
    ),
    "arabic_orpo_dpo_mix_40k": DatasetSpec(
        key="arabic_orpo_dpo_mix_40k",
        repo_id=ARABIC_ORPO_DPO_MIX_REPO,
        config_names=("default",),
        split="train",
        stage="dpo_preference",
        enable_thinking=False,
        weight=0.65,
    ),
}

ALL_9_DATASET_REPOS: Tuple[str, ...] = (
    VOHO_SAUDI_DIALOGUES_REPO,
    SAUDI_ARABIC_LAWS_REPO,
    ACEGPT_V2_TUNING_REPO,
    ARABIC_MATH_SFT_REPO,
    AYA_COLLECTION_REPO,
    ULTRACHAT_200K_REPO,
    OPENMATH_INSTRUCT_2_REPO,
    ARABIC_DIALECT_DPO_REPO,
    ARABIC_ORPO_DPO_MIX_REPO,
)

SFT_ARABIC_MIX_RATIO: float = 0.75
SFT_REPLAY_MIX_RATIO: float = 0.25


# ---------------------------------------------------------------------------
# Arabic & Saudi Domain System Prompts
# ---------------------------------------------------------------------------
DEFAULT_MSA_SYSTEM_PROMPT: str = (
    "أنت مساعد ذكي متقن للغة العربية الفصحى وعلومها، تقدم إجابات دقيقة ومنظمة وموثوقة."
)

SAUDI_DIALECT_SYSTEM_PROMPT: str = (
    "أنت مساعد ذكي يتحدث اللهجة السعودية بطلاقة وطبيعية (نجدية، حجازية، وخليجية) "
    "مع مراعاة الثقافة المحلية والسياق اليومي والمهني في المملكة العربية السعودية."
)

SAUDI_LEGAL_SYSTEM_PROMPT: str = (
    "أنت مستشار قانوني متخصص في الأنظمة واللوائح السعودية الصادرة عن وزارة العدل "
    "وهيئة الخبراء بمجلس الوزراء. استند بدقة إلى نصوص المواد النظامية عند الإجابة."
)

ARABIC_MATH_REASONING_SYSTEM_PROMPT: str = (
    "أنت خبير في الرياضيات والعلوم والمنطق باللغة العربية. قم بتحليل المسألة خطوة بخطوة "
    "والتحقق من العمليات الحسابية قبل تقديم الإجابة النهائية."
)

SOVEREIGN_TOOL_CALLING_SYSTEM_PROMPT: str = (
    "أنت وكيل ذكي قادر على استدعاء الأدوات والوظائف البرمجية بتنسيق JSON دقيق، "
    "والاستجابة باللغة العربية الفصحى أو اللهجة السعودية حسب طلب المستخدم."
)


# ---------------------------------------------------------------------------
# Trust-Region DPO and TIES-Merging Constants
# ---------------------------------------------------------------------------
DPO_BETA: float = 0.1
DPO_TRUST_REGION_ALPHA: float = 0.6
DPO_LEARNING_RATE: float = 5.0e-7

DEFAULT_TIES_LAMBDA: float = 0.85
DEFAULT_TIES_DENSITY: float = 0.70
MIN_RETENTION_RATIO: float = 0.998
