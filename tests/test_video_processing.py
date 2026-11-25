"""
tests/test_video_processing.py
===============================
Tests for video validation and frame extraction (16-frame sampling).
Run with: py -m pytest tests/ -v
"""

import pytest
import numpy as np
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import NUM_FRAMES
from src.video_processor import (
    validate_video,
    extract_frames,
    frames_to_thumbnails,
    VideoValidationError,
)


def make_test_video(path: str, num_frames: int = 30, fps: int = 30, size: tuple = (320, 240)):
    """Create a small synthetic video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(num_frames):
        frame = np.ones((size[1], size[0], 3), dtype=np.uint8) * (i * 8 % 255)
        out.write(frame)
    out.release()


class TestVideoValidation:

    def test_validate_valid_video(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p)
        meta = validate_video(p)
        assert meta["total_frames"] > 0
        assert meta["fps"] > 0
        assert meta["width"] > 0

    def test_validate_missing_file_raises(self):
        with pytest.raises(VideoValidationError, match="not found"):
            validate_video("/nonexistent/path/video.mp4")

    def test_validate_unsupported_extension_raises(self, tmp_path):
        p_obj = tmp_path / "test.txt"
        p_obj.write_text("not a video")
        with pytest.raises(VideoValidationError, match="Unsupported"):
            validate_video(str(p_obj))


class TestFrameExtraction:

    def test_extract_correct_count(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p, num_frames=60)
        frames, indices = extract_frames(p, num_frames=NUM_FRAMES)
        assert len(frames) == NUM_FRAMES
        assert len(indices) == NUM_FRAMES

    def test_extract_short_video(self, tmp_path):
        """Video with fewer frames than requested should still return NUM_FRAMES (tiled)."""
        p = str(tmp_path / "short.mp4")
        make_test_video(p, num_frames=5)
        frames, indices = extract_frames(p, num_frames=NUM_FRAMES)
        assert len(frames) == NUM_FRAMES

    def test_frames_are_numpy_arrays(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p)
        frames, _ = extract_frames(p)
        assert len(frames) == NUM_FRAMES
        assert all(isinstance(f, np.ndarray) for f in frames)
        assert all(f.ndim == 3 for f in frames)

    def test_thumbnails_output(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p)
        frames, _ = extract_frames(p)
        thumbs = frames_to_thumbnails(frames, max_thumbnails=4)
        assert len(thumbs) <= 4
        assert all(isinstance(t, np.ndarray) for t in thumbs)
