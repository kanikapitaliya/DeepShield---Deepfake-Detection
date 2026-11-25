"""
tests/test_prediction.py
========================
Integration tests: preprocessing, rPPG rejection, unified 3-model pipeline, and artifact loading.
Run with: py -m pytest tests/ -v
"""

import pytest
import numpy as np
import torch
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import NUM_FRAMES, IMAGE_SIZE, MODEL_A_PATH, MODEL_B_PATH, MODEL_C_PATH
from src.video_processor import extract_frames, VideoValidationError
from src.face_detector import preprocess_frames
from src.rppg import extract_rppg_features
from src.inference_pipeline import InferencePipeline, InferenceResult
from src.artifact_loader import (
    load_model_a_metrics,
    load_model_a_report,
    load_model_b_metrics,
    load_model_b_report,
    load_model_c_metrics,
)


def make_test_video(path: str, num_frames: int = 40, fps: int = 25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


class TestPreprocessingPipeline:

    def test_preprocess_returns_correct_tensor_shape(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p)
        frames, _ = extract_frames(p, num_frames=NUM_FRAMES)
        tensor, faces = preprocess_frames(frames, image_size=IMAGE_SIZE)
        assert tensor.shape == (1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(faces, int)

    def test_tensor_dtype_is_float32(self, tmp_path):
        p = str(tmp_path / "test.mp4")
        make_test_video(p)
        frames, _ = extract_frames(p, num_frames=NUM_FRAMES)
        tensor, _ = preprocess_frames(frames)
        assert tensor.dtype == torch.float32


class TestRPPGQualityRejection:

    def test_short_video_rejected(self, tmp_path):
        p = str(tmp_path / "short.mp4")
        make_test_video(p, num_frames=30, fps=30)  # 1 second < 8s minimum
        feats, status = extract_rppg_features(p)
        assert feats is None
        assert status == "video_too_short"


class TestEndToEndInferencePipeline:

    @pytest.mark.skipif(
        not (os.path.isfile(MODEL_A_PATH) and os.path.isfile(MODEL_B_PATH) and os.path.isfile(MODEL_C_PATH)),
        reason="Trained model files not present"
    )
    def test_pipeline_runs_on_synthetic_video(self, tmp_path):
        """Short synthetic video without faces should gracefully fall back to visual_only."""
        p = str(tmp_path / "test.mp4")
        make_test_video(p, num_frames=40, fps=25)

        pipeline = InferencePipeline(device=torch.device("cpu"))
        result: InferenceResult = pipeline.run(p)

        assert result.label in ("FAKE", "REAL")
        assert 0.0 <= result.confidence <= 1.0
        assert result.route == "visual_only"
        assert result.rppg_available is False
        assert result.total_frames == NUM_FRAMES
        assert result.inference_time_sec > 0

    @pytest.mark.skipif(
        not os.path.isfile(MODEL_A_PATH),
        reason="Model A checkpoint not present"
    )
    def test_invalid_video_raises(self, tmp_path):
        p = tmp_path / "fake.mp4"
        p.write_text("not a video")
        pipeline = InferencePipeline(device=torch.device("cpu"))
        with pytest.raises(VideoValidationError):
            pipeline.run(str(p))


class TestArtifacts:

    def test_metrics_json_loadable(self):
        a = load_model_a_metrics()
        b = load_model_b_metrics()
        c = load_model_c_metrics()
        assert a is None or isinstance(a, dict)
        assert b is None or isinstance(b, dict)
        assert c is None or isinstance(c, dict)

    def test_classification_reports_loadable(self):
        ra = load_model_a_report()
        rb = load_model_b_report()
        assert ra is None or (isinstance(ra, str) and len(ra) > 0)
        assert rb is None or (isinstance(rb, str) and len(rb) > 0)
