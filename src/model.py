"""
src/model.py
============
Exact ResNeXt50 + LSTM architecture as confirmed from the trained checkpoint.

Architecture confirmed from checkpoint inspection (Phase 1):
  - encoder: resnext50_32x4d feature extractor (8 layers, no FC, with global avg pool)
  - lstm:    nn.LSTM(input_size=2048, hidden_size=2048, num_layers=1, batch_first=True)
  - classifier: Dropout -> Linear(2048, 256) -> ReLU -> Dropout -> Linear(256, 1)

Checkpoint metadata:
  - num_frames = 20
  - image_size = 224
  - saved from cuda:0 (loaded with map_location='cpu')
"""

import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

# Confirmed from checkpoint inspection
NUM_FRAMES = 20
IMAGE_SIZE = 224
CNN_FEATURE_DIM = 2048
LSTM_HIDDEN = 256
CLASSIFIER_HIDDEN = 128


class DeepFakeDetector(nn.Module):
    """
    ResNeXt50 + LSTM deepfake detector.

    Forward input:  [batch, num_frames, 3, 224, 224]
    Forward output: [batch, 1]  raw logit (apply sigmoid for probability)
    Probability > 0.5 → FAKE
    """

    def __init__(self, num_frames: int = NUM_FRAMES, dropout: float = 0.5):
        super().__init__()
        self.num_frames = num_frames

        # --- Encoder: resnext50_32x4d without the final FC layer ---
        # This matches encoder.0 through encoder.8 in the checkpoint.
        backbone = models.resnext50_32x4d(weights=None)
        # Build Sequential identical to what was trained:
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        self.encoder = nn.Sequential(
            backbone.conv1,       # encoder.0
            backbone.bn1,         # encoder.1
            backbone.relu,        # encoder.2
            backbone.maxpool,     # encoder.3
            backbone.layer1,      # encoder.4  (3 Bottleneck blocks)
            backbone.layer2,      # encoder.5  (4 Bottleneck blocks)
            backbone.layer3,      # encoder.6  (6 Bottleneck blocks)
            backbone.layer4,      # encoder.7  (3 Bottleneck blocks)
            backbone.avgpool,     # encoder.8  (AdaptiveAvgPool2d(1,1))
        )

        # --- LSTM: single layer, input=2048, hidden_size=256 ---
        self.lstm = nn.LSTM(
            input_size=CNN_FEATURE_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        # --- Classifier head: matches classifier.0–4 in checkpoint ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),                        # classifier.0
            nn.Linear(LSTM_HIDDEN, CLASSIFIER_HIDDEN),  # classifier.1 (256 -> 128)
            nn.ReLU(inplace=True),                      # classifier.2
            nn.Dropout(dropout),                        # classifier.3
            nn.Linear(CLASSIFIER_HIDDEN, 1),            # classifier.4 (128 -> 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, H, W]  — batch of frame sequences
        Returns:
            [B, 1] raw logit (before sigmoid)
        """
        b, t, c, h, w = x.shape

        # Process all frames through the CNN encoder
        x = x.view(b * t, c, h, w)          # [B*T, 3, H, W]
        x = self.encoder(x)                  # [B*T, 2048, 1, 1]
        x = x.view(b, t, CNN_FEATURE_DIM)   # [B, T, 2048]

        # LSTM — use last hidden state
        _, (h_n, _) = self.lstm(x)          # h_n: [1, B, 256]
        x = h_n.squeeze(0)                  # [B, 256]

        # Classifier
        x = self.classifier(x)              # [B, 1]
        return x


def load_model(checkpoint_path: str, device: torch.device) -> DeepFakeDetector:
    """
    Load the trained DeepFakeDetector from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: torch.device to load onto.

    Returns:
        model in eval mode on the given device.

    Raises:
        FileNotFoundError: if checkpoint does not exist.
        RuntimeError: if the state dict does not match the architecture.
    """
    import os
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Checkpoint is a dict with 'model_state_dict' and metadata
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta_frames = checkpoint.get("num_frames", NUM_FRAMES)
        meta_size = checkpoint.get("image_size", IMAGE_SIZE)
        logger.info(
            f"Checkpoint metadata — epoch: {checkpoint.get('epoch')}, "
            f"num_frames: {meta_frames}, image_size: {meta_size}"
        )
    else:
        # Fallback: checkpoint is the state dict itself
        state_dict = checkpoint

    model = DeepFakeDetector(num_frames=NUM_FRAMES)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully and set to eval mode.")
    return model


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        logger.info("CUDA available — using GPU.")
        return torch.device("cuda")
    logger.info("CUDA not available — using CPU.")
    return torch.device("cpu")
