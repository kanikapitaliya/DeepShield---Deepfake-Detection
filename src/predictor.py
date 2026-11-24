"""
src/predictor.py
================
End-to-end inference pipeline: video path → prediction result.

Ties together: validate → extract frames → detect/preprocess faces → model → result.
The model is loaded once at startup and reused for all predictions.
"""

import torch
import numpy as np
import logging
import time
from typing import Optional
from dataclasses import dataclass

from src.video_processor import validate_video, extract_frames, VideoValidationError
from src.face_detector import preprocess_frames, get_face_crops_for_display

logger = logging.getLogger(__name__)

# Decision threshold (confirmed from checkpoint metadata)
DECISION_THRESHOLD = 0.5


@dataclass
class PredictionResult:
    """Container for all inference outputs."""
    label: str                    # "FAKE" or "REAL"
    confidence: float             # probability [0.0, 1.0] — P(FAKE)
    is_fake: bool
    faces_detected: int           # frames where a face was found
    total_frames: int             # frames passed to model
    face_crops: list              # RGB uint8 numpy arrays for display
    video_meta: dict              # fps, duration, resolution, etc.
    inference_time_sec: float
    warning: Optional[str] = None  # e.g. "No faces detected — results may be unreliable"


def run_inference(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    num_frames: int = 20,
    image_size: int = 224,
) -> PredictionResult:
    """
    Full inference pipeline for a single video.

    Args:
        video_path:  Path to the uploaded video file.
        model:       Loaded DeepFakeDetector in eval mode.
        device:      Inference device.
        num_frames:  Number of frames to sample (default 20, matches training).
        image_size:  Frame resize target (default 224, matches training).

    Returns:
        PredictionResult with label, confidence, and display crops.

    Raises:
        VideoValidationError: for invalid/unreadable files.
        RuntimeError: for unexpected inference failures.
    """
    t0 = time.perf_counter()

    # Step 1: Validate
    video_meta = validate_video(video_path)
    logger.info(f"Video: {video_meta['width']}×{video_meta['height']}, "
                f"{video_meta['total_frames']} frames, {video_meta['fps']:.1f} fps")

    # Step 2: Extract frames
    frames_bgr, frame_indices = extract_frames(video_path, num_frames=num_frames)

    # Step 3: Face detection and preprocessing
    tensor, faces_detected = preprocess_frames(frames_bgr, image_size=image_size)

    # Step 4: Get display crops (before tensor conversion)
    face_crops = get_face_crops_for_display(frames_bgr, image_size=image_size, max_display=8)

    # Step 5: Inference
    tensor = tensor.to(device)
    with torch.no_grad():
        logit = model(tensor)          # [1, 1]
        prob_fake = torch.sigmoid(logit).item()

    # Step 6: Decision
    is_fake = prob_fake >= DECISION_THRESHOLD
    label = "FAKE" if is_fake else "REAL"

    # Build warning if no faces were found
    warning = None
    if faces_detected == 0:
        warning = (
            "No faces were detected in this video. The model analyzed the full "
            "frames instead. Results may be less reliable for non-face content."
        )
    elif faces_detected < num_frames * 0.5:
        warning = (
            f"Faces detected in only {faces_detected}/{num_frames} frames. "
            "Results may be less reliable."
        )

    t1 = time.perf_counter()

    return PredictionResult(
        label=label,
        confidence=prob_fake,
        is_fake=is_fake,
        faces_detected=faces_detected,
        total_frames=len(frames_bgr),
        face_crops=face_crops,
        video_meta=video_meta,
        inference_time_sec=t1 - t0,
        warning=warning,
    )
