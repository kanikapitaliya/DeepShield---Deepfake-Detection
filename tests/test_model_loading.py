"""
tests/test_model_loading.py
===========================
Tests for Model A architecture definition, embedding output, and checkpoint loading.
Run with: py -m pytest tests/ -v
"""

import pytest
import torch
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import MODEL_A_PATH, NUM_FRAMES, IMAGE_SIZE
from src.model import DeepFakeDetector, load_model


class TestModelArchitecture:
    """Tests that the model can be instantiated with the correct architecture."""

    def test_config_constants(self):
        assert NUM_FRAMES == 16
        assert IMAGE_SIZE == 224

    def test_model_instantiation(self):
        model = DeepFakeDetector()
        assert model is not None

    def test_encoder_has_9_layers(self):
        model = DeepFakeDetector()
        assert len(model.encoder) == 9

    def test_lstm_config(self):
        model = DeepFakeDetector()
        assert model.lstm.input_size == 2048
        assert model.lstm.hidden_size == 256
        assert model.lstm.num_layers == 1

    def test_classifier_has_5_layers(self):
        model = DeepFakeDetector()
        assert len(model.classifier) == 5

    def test_forward_shape(self):
        """Model should accept [1, 16, 3, 224, 224] and return [1, 1]."""
        model = DeepFakeDetector()
        model.eval()
        dummy = torch.zeros(1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"

    def test_forward_embedding_shape(self):
        """forward_embedding should return [1, 256] LSTM feature vector."""
        model = DeepFakeDetector()
        model.eval()
        dummy = torch.zeros(1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            emb = model.forward_embedding(dummy)
        assert emb.shape == (1, 256), f"Expected (1,256), got {emb.shape}"

    def test_output_is_finite(self):
        model = DeepFakeDetector()
        model.eval()
        dummy = torch.randn(1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = model(dummy)
        assert torch.isfinite(out).all()

    def test_sigmoid_output_range(self):
        """Sigmoid of any finite logit should be in [0, 1]."""
        model = DeepFakeDetector()
        model.eval()
        dummy = torch.randn(1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            logit = model(dummy)
            prob = torch.sigmoid(logit)
        assert 0.0 <= prob.item() <= 1.0


class TestCheckpointLoading:
    """Tests for checkpoint loading — skipped if checkpoint not present."""

    @pytest.mark.skipif(
        not os.path.isfile(MODEL_A_PATH),
        reason="Checkpoint not present (not committed to repo)"
    )
    def test_checkpoint_loads(self):
        device = torch.device("cpu")
        model = load_model(MODEL_A_PATH, device)
        assert model is not None

    @pytest.mark.skipif(
        not os.path.isfile(MODEL_A_PATH),
        reason="Checkpoint not present"
    )
    def test_checkpoint_model_in_eval_mode(self):
        device = torch.device("cpu")
        model = load_model(MODEL_A_PATH, device)
        assert not model.training

    @pytest.mark.skipif(
        not os.path.isfile(MODEL_A_PATH),
        reason="Checkpoint not present"
    )
    def test_checkpoint_forward_pass(self):
        device = torch.device("cpu")
        model = load_model(MODEL_A_PATH, device)
        dummy = torch.zeros(1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 1)

    def test_missing_checkpoint_raises(self):
        device = torch.device("cpu")
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.pt", device)
