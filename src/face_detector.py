"""
src/face_detector.py
====================
Face detection, cropping, and preprocessing for the DeepFakeDetector model.

NOTE ON PREPROCESSING FIDELITY:
  The exact face preprocessing used during training is not documented in the
  checkpoint. The pipeline below follows the standard approach for FaceForensics++
  training with ResNeXt50:
    1. Detect faces using MTCNN (facenet-pytorch)
    2. Crop and pad to a square region with a small margin
    3. Resize to IMAGE_SIZE × IMAGE_SIZE (224)
    4. Normalize with ImageNet statistics

  This module is intentionally isolated so it can be replaced or fine-tuned
  once the original training preprocessing is verified.

Fallback: if MTCNN fails or finds no face, the full frame is resized and used.
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization — standard for pretrained ResNet/ResNeXt
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = 224
FACE_MARGIN = 0.3  # fractional margin added around detected face box


def _load_mtcnn():
    """Lazy-load MTCNN to avoid import-time GPU allocation."""
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(
            keep_all=False,
            device="cpu",
            select_largest=True,
            post_process=False,
        )
        return mtcnn
    except ImportError:
        logger.warning("facenet-pytorch not installed; face detection disabled.")
        return None


_mtcnn = None  # module-level singleton


def get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = _load_mtcnn()
    return _mtcnn


def detect_and_crop_face(
    frame_bgr: np.ndarray,
    image_size: int = IMAGE_SIZE,
    margin: float = FACE_MARGIN,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Detect the largest face in a BGR frame and return a cropped, resized RGB patch.

    Args:
        frame_bgr: OpenCV BGR image (H, W, 3) uint8
        image_size: target square size after crop
        margin: fractional padding around the detected bounding box

    Returns:
        (crop_rgb, face_found)
        crop_rgb: float32 numpy [H, W, 3] in [0,1] — always returned even if no face
        face_found: True if MTCNN detected a face, False if full-frame fallback used
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    face_found = False

    mtcnn = get_mtcnn()
    box = None

    if mtcnn is not None:
        try:
            pil_img = Image.fromarray(frame_rgb)
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]  # [x1, y1, x2, y2]
                face_found = True
        except Exception as e:
            logger.debug(f"MTCNN detection error: {e}")

    if box is not None:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx = int(bw * margin)
        my = int(bh * margin)
        x1 = max(0, int(x1) - mx)
        y1 = max(0, int(y1) - my)
        x2 = min(w, int(x2) + mx)
        y2 = min(h, int(y2) + my)
        crop = frame_rgb[y1:y2, x1:x2]
    else:
        # Full-frame fallback
        crop = frame_rgb

    # Resize to target size
    crop_resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # Convert to float32 in [0, 1]
    crop_float = crop_resized.astype(np.float32) / 255.0

    return crop_float, face_found


def preprocess_frames(
    frames_bgr: List[np.ndarray],
    image_size: int = IMAGE_SIZE,
) -> Tuple[torch.Tensor, int]:
    """
    Process a list of BGR frames into a model-ready tensor.

    Args:
        frames_bgr: list of BGR frames from video_processor.extract_frames
        image_size: target size (default 224)

    Returns:
        tensor: [1, T, 3, H, W] float32 — ready for model.forward()
        faces_detected: count of frames where a face was detected
    """
    processed = []
    faces_detected = 0

    for frame in frames_bgr:
        crop_float, face_found = detect_and_crop_face(frame, image_size)
        if face_found:
            faces_detected += 1

        # Apply ImageNet normalization: (pixel - mean) / std
        normalized = (crop_float - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW
        chw = np.transpose(normalized, (2, 0, 1))  # [3, H, W]
        processed.append(chw)

    # Stack to [T, 3, H, W] then add batch dim → [1, T, 3, H, W]
    stacked = np.stack(processed, axis=0)                       # [T, 3, H, W]
    tensor = torch.from_numpy(stacked).unsqueeze(0).float()    # [1, T, 3, H, W]

    logger.info(
        f"Preprocessed {len(frames_bgr)} frames; faces detected in {faces_detected}/{len(frames_bgr)}"
    )
    return tensor, faces_detected


def get_face_crops_for_display(
    frames_bgr: List[np.ndarray],
    image_size: int = 224,
    max_display: int = 8,
) -> List[np.ndarray]:
    """
    Return RGB crops (uint8) for display in the Frame Analysis tab.
    Samples up to max_display frames evenly.
    """
    step = max(1, len(frames_bgr) // max_display)
    sampled = frames_bgr[::step][:max_display]
    crops = []
    for frame in sampled:
        crop_float, _ = detect_and_crop_face(frame, image_size)
        crop_uint8 = (crop_float * 255).clip(0, 255).astype(np.uint8)
        crops.append(crop_uint8)
    return crops
