"""
src/face_detector.py
====================
Face detection and preprocessing for Model A (ResNeXt50 + LSTM).

Uses OpenCV Haar Cascade (haarcascade_frontalface_default.xml), consistent
with the training pipeline in notebooks 01/02/03.

For inference on frames passed to Model A:
  - Detect face with Haar Cascade
  - Crop central 80% of face region (avoid hair/background)
  - Resize to 224x224
  - Apply ImageNet normalization

If no face is detected, uses the full frame resized to 224x224.
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Tuple, Optional

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

# Module-level singleton — Haar cascade
_haar_detector = None


def _get_haar_detector():
    global _haar_detector
    if _haar_detector is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_detector = cv2.CascadeClassifier(cascade_path)
        if _haar_detector.empty():
            logger.warning(
                "Failed to load Haar cascade. Face detection disabled; "
                "full frames will be used as fallback."
            )
            _haar_detector = None
    return _haar_detector


def detect_and_crop_face(
    frame_bgr: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Detect face in a BGR frame using Haar cascade and return a preprocessed crop.

    Strategy from notebook 03:
      - Detect largest face
      - Use central 80%×80% region of bounding box (avoids hair/background)
      - Resize to image_size × image_size
      - Normalize with ImageNet stats

    Args:
        frame_bgr: OpenCV BGR uint8 frame
        image_size: target square side

    Returns:
        (crop_normalized_chw, face_found)
        crop_normalized_chw: float32 numpy [3, H, W] ready for torch — always returned
        face_found:          True if a face was actually detected
    """
    h, w = frame_bgr.shape[:2]
    face_found = False
    crop_rgb = None

    detector = _get_haar_detector()
    if detector is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            # Largest face by area
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            # Central 80%×55% crop (avoids hair top and chin bottom)
            x1 = int(x + 0.10 * fw)
            x2 = int(x + 0.90 * fw)
            y1 = int(y + 0.10 * fh)
            y2 = int(y + 0.90 * fh)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size > 0 and roi.shape[0] >= 10 and roi.shape[1] >= 10:
                crop_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                face_found = True

    if crop_rgb is None:
        # Fallback: full frame
        crop_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize to target
    crop_resized = cv2.resize(crop_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # Normalize: [0,255] uint8 → float32 [0,1] → ImageNet normalized CHW
    crop_float = crop_resized.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    normalized = (crop_float - mean) / std       # HWC float32
    chw = np.transpose(normalized, (2, 0, 1))    # CHW

    return chw, face_found


def preprocess_frames(
    frames_bgr: List[np.ndarray],
    image_size: int = IMAGE_SIZE,
) -> Tuple[torch.Tensor, int]:
    """
    Process BGR frames into a model-ready tensor [1, T, 3, H, W].

    Returns:
        tensor:         [1, T, 3, H, W] float32
        faces_detected: number of frames where a face was found
    """
    processed = []
    faces_detected = 0

    for frame in frames_bgr:
        chw, face_found = detect_and_crop_face(frame, image_size)
        if face_found:
            faces_detected += 1
        processed.append(chw)

    stacked = np.stack(processed, axis=0)                        # [T, 3, H, W]
    tensor  = torch.from_numpy(stacked).unsqueeze(0).float()    # [1, T, 3, H, W]

    logger.info(
        f"Preprocessed {len(frames_bgr)} frames; "
        f"faces detected in {faces_detected}/{len(frames_bgr)}"
    )
    return tensor, faces_detected


def get_face_crops_for_display(
    frames_bgr: List[np.ndarray],
    image_size: int = 224,
    max_display: int = 8,
) -> List[np.ndarray]:
    """
    Return RGB uint8 face crops for display in the UI gallery.
    Samples up to max_display frames evenly.
    """
    step = max(1, len(frames_bgr) // max_display)
    sampled = frames_bgr[::step][:max_display]
    crops = []
    for frame in sampled:
        chw, _ = detect_and_crop_face(frame, image_size)
        # CHW float normalized → HWC uint8 unnormalized
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        hwc  = np.transpose(chw, (1, 2, 0)) * std + mean
        uint8 = (hwc * 255).clip(0, 255).astype(np.uint8)
        crops.append(uint8)
    return crops
