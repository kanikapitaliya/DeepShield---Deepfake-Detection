"""
src/inference_pipeline.py
=========================
Unified inference pipeline for DeepShield.

Runs all three models and returns a consolidated result:
  - Model A:  ResNeXt50+LSTM visual prediction + 256-dim embedding
  - Model B:  CHROM-rPPG physiological signal features + ExtraTrees classifier
  - Model C:  Hybrid fusion (visual + rPPG) or visual-only fallback

Result routes:
  1. hybrid_fusion  — both Model A and Model B succeeded
  2. visual_only    — Model B failed (quality rejection); Model A fallback used
  3. error          — Model A itself failed (unusual)
"""

import torch
import numpy as np
import logging
import time
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Full result from the DeepShield 3-model pipeline."""

    # ── Overall verdict ──────────────────────────────────────────────────────
    label: str              # "FAKE" or "REAL"
    confidence: float       # P(FAKE) ∈ [0, 1]
    is_fake: bool
    route: str              # "hybrid_fusion" | "visual_only" | "error"

    # ── Model A ──────────────────────────────────────────────────────────────
    visual_prob: float      # P(FAKE) from Model A direct threshold
    visual_label: str       # "FAKE" | "REAL"
    visual_embedding: Optional[np.ndarray] = None   # 256-dim

    # ── Model B ──────────────────────────────────────────────────────────────
    rppg_available: bool = False
    rppg_prob: Optional[float] = None   # P(FAKE) from ExtraTrees (if available)
    rppg_label: Optional[str] = None
    rppg_status: str = "not_run"        # rejection reason or 'ok'
    rppg_features: Optional[Dict[str, float]] = None

    # ── Video metadata ────────────────────────────────────────────────────────
    total_frames: int = 0
    faces_detected: int = 0
    video_meta: Dict[str, Any] = field(default_factory=dict)
    face_crops: List[np.ndarray] = field(default_factory=list)

    # ── Timing / warnings ────────────────────────────────────────────────────
    inference_time_sec: float = 0.0
    warning: Optional[str] = None


class InferencePipeline:
    """
    Loads all three models once and runs the full pipeline per video.
    Thread-safe for concurrent requests (each call is stateless).
    """

    def __init__(self, device: torch.device = None):
        from src.config import get_device, MODEL_A_PATH, MODEL_B_PATH, MODEL_C_PATH
        from src.model import load_model

        self.device = device or get_device()
        logger.info(f"InferencePipeline device: {self.device}")

        # Model A
        logger.info("Loading Model A (ResNeXt50+LSTM)...")
        self.model_a = load_model(MODEL_A_PATH, self.device)
        self.model_a.eval()

        # Model B
        logger.info("Loading Model B (rPPG ExtraTrees)...")
        self.model_b_bundle = self._load_model_b(MODEL_B_PATH)

        # Model C
        logger.info("Loading Model C (hybrid fusion)...")
        self.model_c_bundle = self._load_model_c(MODEL_C_PATH)

        logger.info("All models loaded successfully.")

    # ── Loaders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model_b(path: str):
        import joblib
        if not os.path.isfile(path):
            logger.warning(f"Model B not found: {path}")
            return None
        bundle = joblib.load(path)
        # keys: model, feature_cols, threshold
        logger.info(
            f"Model B loaded — classifier: {type(bundle['model']).__name__}, "
            f"threshold: {bundle['threshold']}"
        )
        return bundle

    @staticmethod
    def _load_model_c(path: str):
        import joblib
        if not os.path.isfile(path):
            logger.warning(f"Model C not found: {path}")
            return None
        bundle = joblib.load(path)
        # Compatibility shim for SimpleImputer pickled in sklearn 1.6.1 and loaded in 1.9.x
        for pipe in bundle.get('experiments', {}).values():
            if hasattr(pipe, 'named_steps'):
                for step in pipe.named_steps.values():
                    if not hasattr(step, '_fill_dtype') and hasattr(step, '_fit_dtype'):
                        step._fill_dtype = step._fit_dtype
        # keys: experiments, feature_columns, thresholds
        logger.info(f"Model C loaded — experiments: {list(bundle['experiments'].keys())}")
        return bundle

    # ── Main inference entry point ────────────────────────────────────────────

    def run(self, video_path: str) -> InferenceResult:
        """
        Full 3-model pipeline for a single video.

        Args:
            video_path: Path to the video file.

        Returns:
            InferenceResult with all outputs.
        """
        from src.config import (
            NUM_FRAMES, IMAGE_SIZE,
            MODEL_A_THRESHOLD, MODEL_B_THRESHOLD,
            MODEL_C_HYBRID_THRESHOLD, MODEL_C_VISUAL_ONLY_THRESHOLD,
        )
        from src.video_processor import validate_video, extract_frames, VideoValidationError
        from src.face_detector import preprocess_frames, get_face_crops_for_display
        from src.rppg import extract_rppg_features

        t0 = time.perf_counter()

        # ── 1. Validate ───────────────────────────────────────────────────────
        video_meta = validate_video(video_path)

        # ── 2. Extract frames ────────────────────────────────────────────────
        frames_bgr, _ = extract_frames(video_path, num_frames=NUM_FRAMES)

        # ── 3. Face preprocessing ─────────────────────────────────────────────
        tensor, faces_detected = preprocess_frames(frames_bgr, image_size=IMAGE_SIZE)
        face_crops = get_face_crops_for_display(frames_bgr, image_size=IMAGE_SIZE, max_display=8)

        # ── 4. Model A inference ──────────────────────────────────────────────
        tensor = tensor.to(self.device)
        with torch.no_grad():
            embedding = self.model_a.forward_embedding(tensor)   # [1, 256]
            logit     = self.model_a.classifier(embedding)        # [1, 1]
            visual_prob = torch.sigmoid(logit).item()

        visual_embedding = embedding.squeeze(0).cpu().numpy()    # [256]
        visual_label = "FAKE" if visual_prob >= MODEL_A_THRESHOLD else "REAL"

        # ── 5. Model B: rPPG ──────────────────────────────────────────────────
        rppg_available = False
        rppg_prob = None
        rppg_label = None
        rppg_status = "not_run"
        rppg_features = None

        if self.model_b_bundle is not None:
            feat_dict, status = extract_rppg_features(video_path)
            rppg_status = status
            if status == 'ok':
                clf    = self.model_b_bundle['model']
                f_cols = self.model_b_bundle['feature_cols']
                thresh = self.model_b_bundle['threshold']

                feat_arr = np.array([[feat_dict[c] for c in f_cols]], dtype=np.float64)
                rppg_prob = float(clf.predict_proba(feat_arr)[0, 1])
                rppg_label = "FAKE" if rppg_prob >= thresh else "REAL"
                rppg_available = True
                rppg_features = feat_dict

        # ── 6. Model C: hybrid fusion ─────────────────────────────────────────
        if self.model_c_bundle is not None:
            if rppg_available:
                # Full hybrid fusion
                clf    = self.model_c_bundle['experiments']['hybrid_fusion']
                f_cols = self.model_c_bundle['feature_columns']['hybrid_fusion']
                thresh = self.model_c_bundle['thresholds']['hybrid_fusion']

                # Build 264-dim feature vector: 256 visual + 8 rPPG
                rppg_arr = np.array([rppg_features[c] for c in self.model_c_bundle['feature_columns']['rppg_only']], dtype=np.float64)
                feat_arr = np.concatenate([visual_embedding, rppg_arr]).reshape(1, -1)
                prob = float(clf.predict_proba(feat_arr)[0, 1])
                label = "FAKE" if prob >= thresh else "REAL"
                route = "hybrid_fusion"
                confidence = prob

            else:
                # Visual-only fallback through Model C
                clf    = self.model_c_bundle['experiments']['visual_only']
                thresh = self.model_c_bundle['thresholds']['visual_only']
                feat_arr = visual_embedding.reshape(1, -1)
                prob = float(clf.predict_proba(feat_arr)[0, 1])
                label = "FAKE" if prob >= thresh else "REAL"
                route = "visual_only"
                confidence = prob

        else:
            # No Model C — fall back to raw Model A output
            confidence = visual_prob
            label = visual_label
            route = "visual_only"

        is_fake = label == "FAKE"

        # ── 7. Warnings ───────────────────────────────────────────────────────
        warning = None
        if faces_detected == 0:
            warning = (
                "No faces were detected in this video. The model analyzed full frames "
                "as a fallback. Results may be less reliable."
            )
        elif faces_detected < NUM_FRAMES * 0.5:
            warning = (
                f"Faces detected in only {faces_detected}/{NUM_FRAMES} frames. "
                "Results may be less reliable."
            )
        if not rppg_available and rppg_status not in ("not_run",):
            rppg_note = f" Physiological analysis unavailable ({rppg_status.replace('_', ' ')})."
            warning = (warning or "") + rppg_note

        return InferenceResult(
            label=label,
            confidence=confidence,
            is_fake=is_fake,
            route=route,
            visual_prob=visual_prob,
            visual_label=visual_label,
            visual_embedding=visual_embedding,
            rppg_available=rppg_available,
            rppg_prob=rppg_prob,
            rppg_label=rppg_label,
            rppg_status=rppg_status,
            rppg_features=rppg_features,
            total_frames=len(frames_bgr),
            faces_detected=faces_detected,
            video_meta=video_meta,
            face_crops=face_crops,
            inference_time_sec=time.perf_counter() - t0,
            warning=warning,
        )
