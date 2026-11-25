"""
src/config.py
=============
Central configuration for DeepShield.
All paths and hyperparameters in one place.
"""

import os
import torch

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_A_PATH = os.path.join(PROJECT_ROOT, "models", "resnext50_lstm_v2_best.pt")
MODEL_B_PATH = os.path.join(PROJECT_ROOT, "models", "model_b_rppg_classifier.joblib")
MODEL_C_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_models.joblib")

# ── Artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS_A = os.path.join(PROJECT_ROOT, "artifacts", "model_a")
ARTIFACTS_B = os.path.join(PROJECT_ROOT, "artifacts", "model_b")
ARTIFACTS_C = os.path.join(PROJECT_ROOT, "artifacts", "model_c")

# ── Model A (ResNeXt50 + LSTM) confirmed from checkpoint ─────────────────────
NUM_FRAMES = 16          # confirmed: checkpoint['num_frames'] == 16
IMAGE_SIZE = 224
CNN_FEATURE_DIM = 2048
LSTM_HIDDEN = 256
CLASSIFIER_HIDDEN = 128
MODEL_A_THRESHOLD = 0.51  # validated on val set, from artifacts

# ── Preprocessing (ImageNet stats) ───────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Model B (rPPG CHROM features) ────────────────────────────────────────────
RPPG_FEATURE_COLS = [
    'pulse_rate_bpm_proxy',
    'band_power',
    'peak_power_ratio',
    'spectral_entropy',
    'pulse_std',
    'pulse_rms',
    'pulse_skew',
    'pulse_kurtosis',
]
MODEL_B_THRESHOLD = 0.70   # from artifacts
RPPG_MAX_FPS = 30.0
RPPG_MIN_SECONDS = 8.0

# ── Model C (hybrid fusion) ───────────────────────────────────────────────────
# Thresholds from model_comparison_metrics.json
MODEL_C_HYBRID_THRESHOLD = 0.50
MODEL_C_VISUAL_ONLY_THRESHOLD = 0.52

# ── File validation ───────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 500
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
