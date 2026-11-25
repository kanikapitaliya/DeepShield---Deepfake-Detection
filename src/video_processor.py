"""
src/video_processor.py
======================
Video validation, frame extraction, and uniform temporal sampling.

Training confirmed from checkpoint:
  - num_frames = 16
  - image_size = 224

Frame sampling strategy: uniform sampling across the full video duration.
If the video has fewer frames than num_frames, the available frames are
repeated (tile) to reach the required count.
"""

import cv2
import numpy as np
import logging
import os
from typing import List, Tuple

from src.config import NUM_FRAMES, IMAGE_SIZE, MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class VideoValidationError(Exception):
    """Raised when a video file fails validation."""
    pass


def validate_video(video_path: str) -> dict:
    """
    Validate a video file for size, format, and readability.

    Returns a dict with video metadata on success.
    Raises VideoValidationError on failure.
    """
    if not os.path.isfile(video_path):
        raise VideoValidationError(f"File not found: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise VideoValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoValidationError("Could not open video file. It may be corrupted.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0:
        raise VideoValidationError("Video appears to have no frames.")

    duration = total_frames / fps if fps > 0 else 0

    return {
        "path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "size_mb": size_mb,
    }


def extract_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Uniformly sample `num_frames` frames from a video.

    Returns:
        frames:        list of BGR uint8 numpy arrays [H, W, 3]
        frame_indices: list of original frame indices selected
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoValidationError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise VideoValidationError("Video has no readable frames.")

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    else:
        base = list(range(total_frames))
        indices = (base * (num_frames // total_frames + 1))[:num_frames]
        indices.sort()
        logger.warning(
            f"Video has only {total_frames} frames; tiling to reach {num_frames}."
        )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1].copy())
                logger.warning(f"Frame {idx} could not be read; duplicating previous.")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = cap.read()
                if ret2:
                    frames.append(frame2)
                else:
                    cap.release()
                    raise VideoValidationError("Could not read any frames from video.")

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}")
    return frames, indices


def frames_to_thumbnails(
    frames: List[np.ndarray],
    thumb_size: int = 160,
    max_thumbnails: int = 8,
) -> List[np.ndarray]:
    """
    Convert BGR frames to small RGB thumbnails for display.
    Returns at most `max_thumbnails` evenly sampled thumbnails.
    """
    if not frames:
        return []

    step = max(1, len(frames) // max_thumbnails)
    selected = frames[::step][:max_thumbnails]

    thumbnails = []
    for f in selected:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = thumb_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        thumbnails.append(resized)

    return thumbnails
