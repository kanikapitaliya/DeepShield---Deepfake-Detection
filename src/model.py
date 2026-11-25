"""
src/model.py
============
ResNeXt50 + LSTM visual deepfake detector (Model A).

Architecture confirmed from checkpoint resnext50_lstm_v2_best.pt:
  - encoder:     ResNeXt50_32x4d (layers 0-8, no FC, with global avg pool)
  - lstm:        nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, batch_first=True)
  - classifier:  Dropout -> Linear(256,128) -> ReLU -> Dropout -> Linear(128,1)
  - num_frames:  16
  - image_size:  224
  - epoch:       2
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import logging

from src.config import (
    NUM_FRAMES, IMAGE_SIZE, CNN_FEATURE_DIM,
    LSTM_HIDDEN, CLASSIFIER_HIDDEN, MODEL_A_PATH,
)

logger = logging.getLogger(__name__)


class DeepFakeDetector(nn.Module):
    """
    ResNeXt50 + LSTM deepfake detector — exact architecture as trained.

    Forward input:  [batch, num_frames, 3, 224, 224]
    Forward output: [batch, 1]  raw logit (apply sigmoid for probability)

    Also provides forward_embedding() that returns the 256-dim LSTM embedding,
    used by Model C (hybrid fusion).
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()

        # Encoder: resnext50_32x4d without the FC layer
        backbone = models.resnext50_32x4d(weights=None)
        self.encoder = nn.Sequential(
            backbone.conv1,    # encoder.0
            backbone.bn1,      # encoder.1
            backbone.relu,     # encoder.2
            backbone.maxpool,  # encoder.3
            backbone.layer1,   # encoder.4
            backbone.layer2,   # encoder.5
            backbone.layer3,   # encoder.6
            backbone.layer4,   # encoder.7
            backbone.avgpool,  # encoder.8  AdaptiveAvgPool2d(1,1)
        )

        # LSTM: input=2048, hidden=256
        self.lstm = nn.LSTM(
            input_size=CNN_FEATURE_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(LSTM_HIDDEN, CLASSIFIER_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 256-dim visual embedding (last LSTM hidden state).
        Args:
            x: [B, T, 3, H, W]
        Returns:
            [B, 256]  — visual embedding for hybrid fusion
        """
        b, t, c, h, w = x.shape
        z = self.encoder(x.reshape(b * t, c, h, w)).reshape(b, t, CNN_FEATURE_DIM)
        _, (h_n, _) = self.lstm(z)
        return h_n.squeeze(0)   # [B, 256]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, H, W]
        Returns:
            [B, 1]  raw logit
        """
        embedding = self.forward_embedding(x)   # [B, 256]
        return self.classifier(embedding)        # [B, 1]


def load_model(checkpoint_path: str = MODEL_A_PATH,
               device: torch.device = None) -> DeepFakeDetector:
    """
    Load the trained DeepFakeDetector from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:          torch.device to load onto (default: auto-detect).

    Returns:
        model in eval mode on the given device.

    Raises:
        FileNotFoundError: checkpoint does not exist.
        RuntimeError:      state dict does not match architecture (strict=True).
    """
    if device is None:
        from src.config import get_device
        device = get_device()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading Model A checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info(
            f"Checkpoint — epoch: {checkpoint.get('epoch')}, "
            f"num_frames: {checkpoint.get('num_frames')}, "
            f"image_size: {checkpoint.get('image_size')}, "
            f"architecture: {checkpoint.get('architecture')}"
        )
    else:
        state_dict = checkpoint

    model = DeepFakeDetector()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    logger.info("Model A loaded successfully.")
    return model
