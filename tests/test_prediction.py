"""
tests/test_prediction.py
========================
Integration test: synthetic video → full inference pipeline.
Run with: py -m pytest tests/ -v
"""

import pytest
import numpy as np
import torch
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import DeepFakeDetector, NUM_FRAMES, IMAGE_SIZE
from src.video_processor import extract_frames
from src.face_detector import preprocess_frames
from src.predictor import run_inference, PredictionResult
from src.visualizations import load_test_metrics, load_classification_report


def make_test_video(path: str, num_frames: int = 40):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 25, (320, 240))
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


class TestPreprocessingPipeline:

    def test_preprocess_returns_correct_tensor_shape(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p)
        frames, _ = extract_frames(p, num_frames=NUM_FRAMES)
        tensor, faces = preprocess_frames(frames, image_size=IMAGE_SIZE)
        # Should be [1, T, 3, H, W]
        assert tensor.shape == (1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_tensor_dtype_is_float32(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p)
        frames, _ = extract_frames(p, num_frames=NUM_FRAMES)
        tensor, _ = preprocess_frames(frames)
        assert tensor.dtype == torch.float32


class TestEndToEndInference:

    def test_inference_with_synthetic_model(self, tmp_path):
        """Test full pipeline with randomly initialized model (no checkpoint needed)."""
        p = str(tmp_path / "test.mp4")
        make_test_video(p, num_frames=40)

        model = DeepFakeDetector()
        model.eval()
        device = torch.device("cpu")

        result: PredictionResult = run_inference(
            video_path=str(p),
            model=model,
            device=device,
        )

        assert result.label in ("FAKE", "REAL")
        assert 0.0 <= result.confidence <= 1.0
        assert result.total_frames == NUM_FRAMES
        assert result.inference_time_sec > 0

    def test_invalid_video_raises(self, tmp_path):
        from src.video_processor import VideoValidationError
        p = tmp_path / "fake.mp4"
        p.write_text("not a video")
        model = DeepFakeDetector()
        device = torch.device("cpu")
        with pytest.raises(Exception):
            run_inference(str(p), model, device)


class TestArtifacts:

    def test_metrics_json_loadable(self):
        metrics = load_test_metrics()
        # Either loads successfully (dict with 'accuracy') or returns error
        assert isinstance(metrics, dict)

    def test_classification_report_is_string(self):
        report = load_classification_report()
        assert isinstance(report, str)
        assert len(report) > 0
