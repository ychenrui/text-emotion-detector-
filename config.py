"""Shared configuration for the emotion classifier pipeline.

Pipeline stages:
    1) MLM domain adaptation on Empathetic Dialogues
    2) Multi-label classification fine-tuning on GoEmotions
    3) Temperature scaling for calibrated probabilities
    4) Inference returning top-3 emotions

All hyperparameters live here so each stage script stays small.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MLM_OUTPUT_DIR = CHECKPOINT_DIR / "stage1_mlm"
CLASSIFIER_OUTPUT_DIR = CHECKPOINT_DIR / "stage2_classifier"
CALIBRATION_FILE = CHECKPOINT_DIR / "temperature.json"

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_SEQ_LENGTH = 128  # most conversational utterances are short

# --------------------------------------------------------------------------- #
# GoEmotions labels: 27 emotions + neutral, alphabetised + neutral last
# This ordering MUST match the dataset's `simplified` config.
# --------------------------------------------------------------------------- #
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
NUM_LABELS = len(GOEMOTIONS_LABELS)  # 28

# --------------------------------------------------------------------------- #
# Stage 1: MLM continued pretraining on Empathetic Dialogues
# Sized for a single RTX 4090 (24 GB VRAM) running bf16 mixed precision.
# --------------------------------------------------------------------------- #
MLM_CONFIG = {
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,   # effective batch = 16
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "mlm_probability": 0.15,
    "logging_steps": 100,
    "save_steps": 1000,
    "eval_steps": 1000,
    "save_total_limit": 2,
}

# --------------------------------------------------------------------------- #
# Stage 2: Multi-label classification on GoEmotions
# --------------------------------------------------------------------------- #
CLS_CONFIG = {
    "num_train_epochs": 5,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "gradient_accumulation_steps": 1,   # effective batch = 32
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 2,
    "metric_for_best_model": "f1_macro",
    "load_best_model_at_end": True,
}

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
