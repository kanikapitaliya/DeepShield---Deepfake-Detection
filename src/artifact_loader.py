"""
src/artifact_loader.py
======================
Loads verified evaluation artifacts from artifacts/model_a/, model_b/, model_c/.
No fabrication. If a file is missing, returns None gracefully.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_A = os.path.join(PROJECT_ROOT, "artifacts", "model_a")
ARTIFACTS_B = os.path.join(PROJECT_ROOT, "artifacts", "model_b")
ARTIFACTS_C = os.path.join(PROJECT_ROOT, "artifacts", "model_c")


def _load_json(path: str) -> Optional[Dict]:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load JSON {path}: {e}")
    return None


def _load_text(path: str) -> Optional[str]:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to load text {path}: {e}")
    return None


def _load_image(path: str) -> Optional[Image.Image]:
    if os.path.isfile(path):
        try:
            return Image.open(path).copy()
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
    return None


# ── Model A artifacts ─────────────────────────────────────────────────────────

def load_model_a_metrics() -> Optional[Dict]:
    return _load_json(os.path.join(ARTIFACTS_A, "test_metrics.json"))


def load_model_a_report() -> Optional[str]:
    return _load_text(os.path.join(ARTIFACTS_A, "classification_report.txt"))


def load_model_a_confusion_matrix() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_A, "confusion_matrix.png"))


def load_model_a_roc_curve() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_A, "roc_curve.png"))


def load_model_a_training_history() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_A, "training_history.png"))


# ── Model B artifacts ─────────────────────────────────────────────────────────

def load_model_b_metrics() -> Optional[Dict]:
    return _load_json(os.path.join(ARTIFACTS_B, "model_b_metrics.json"))


def load_model_b_report() -> Optional[str]:
    return _load_text(os.path.join(ARTIFACTS_B, "model_b_classification_report.txt"))


def load_model_b_confusion_matrix() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_B, "model_b_confusion_matrix.png"))


def load_model_b_roc_curve() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_B, "model_b_roc_curve.png"))


def load_model_b_feature_importance() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_B, "model_b_feature_importance.png"))


# ── Model C artifacts ─────────────────────────────────────────────────────────

def load_model_c_metrics() -> Optional[Dict]:
    return _load_json(os.path.join(ARTIFACTS_C, "model_comparison_metrics.json"))


def load_model_c_comparison_chart() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_C, "model_comparison.png"))


def load_model_c_hybrid_confusion_matrix() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_C, "hybrid_fusion_confusion_matrix.png"))


def load_model_c_visual_confusion_matrix() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_C, "visual_only_confusion_matrix.png"))


def load_model_c_rppg_confusion_matrix() -> Optional[Image.Image]:
    return _load_image(os.path.join(ARTIFACTS_C, "rppg_only_confusion_matrix.png"))
