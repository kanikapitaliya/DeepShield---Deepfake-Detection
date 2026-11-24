"""
src/explainability.py
=====================
Grad-CAM implementation for the DeepFakeDetector.

Grad-CAM is applied to the LAST convolutional layer of the ResNeXt50 encoder:
  encoder.7[-1].conv3  (last conv in layer4, block 2)

This generates a spatial heatmap showing which regions of each frame the CNN
attended to when producing its feature embedding.

IMPORTANT LIMITATIONS:
  - Grad-CAM operates on the CNN encoder only. The LSTM adds temporal context
    that is NOT captured by per-frame spatial maps.
  - The heatmap reflects the CNN's response at the last convolutional layer,
    not necessarily the discriminative cues for deepfake detection.
  - These maps are exploratory visual aids, not proof of authentic tampering regions.
  - The prediction is video-level (from LSTM), not frame-level.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for the DeepFakeDetector.

    Hooks into encoder.7[2].conv3 (last conv layer of layer4).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        # Target: last block of encoder.7 (layer4), last conv
        # encoder.7[2] = Bottleneck block 2, .conv3 = third conv
        try:
            target_layer = self.model.encoder[7][2].conv3
        except (IndexError, AttributeError) as e:
            logger.warning(f"Could not locate Grad-CAM target layer: {e}")
            return

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h_fwd, h_bwd]
        logger.debug("Grad-CAM hooks registered on encoder.7[2].conv3")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(
        self,
        tensor: torch.Tensor,  # [1, T, 3, H, W]
        device: torch.device,
        frame_indices: Optional[List[int]] = None,
        max_maps: int = 8,
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for a subset of sampled frames.

        Returns:
            List of RGB uint8 numpy arrays [H, W, 3] — heatmaps overlaid on frames.
            Returns empty list if Grad-CAM is not available.
        """
        if not self._hooks:
            logger.warning("Grad-CAM hooks not registered; skipping.")
            return []

        self.model.eval()
        tensor = tensor.to(device)
        b, t, c, h, w = tensor.shape

        # Select frames to generate maps for
        step = max(1, t // max_maps)
        frame_idxs = list(range(0, t, step))[:max_maps]

        heatmaps = []
        for fi in frame_idxs:
            # Extract single frame, process through encoder only
            frame = tensor[:, fi:fi+1, :, :, :].clone()  # [1, 1, 3, H, W]
            frame.requires_grad_(False)

            # We need gradients w.r.t. single-frame encoder output
            single_frame = frame.view(1, c, h, w)  # [1, 3, H, W]
            single_frame.requires_grad_(True)

            self._activations = None
            self._gradients = None

            # Forward pass through encoder
            feat = self.model.encoder(single_frame)  # [1, 2048, 1, 1]

            if self._activations is None:
                break

            # Sum the feature map as a scalar proxy for "activation magnitude"
            score = feat.sum()
            self.model.zero_grad()
            score.backward()

            if self._gradients is None:
                break

            # Grad-CAM computation
            grads = self._gradients           # [1, C, H', W']
            acts = self._activations          # [1, C, H', W']

            weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            cam = (weights * acts).sum(dim=1).squeeze()     # [H', W']
            cam = torch.relu(cam)

            cam_np = cam.cpu().numpy()
            if cam_np.max() > 0:
                cam_np = cam_np / cam_np.max()

            # Resize to original frame size
            cam_resized = cv2.resize(cam_np, (w, h))

            # Create heatmap overlay
            heatmap_color = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # Get original frame as RGB uint8
            orig = tensor[0, fi].cpu().permute(1, 2, 0).numpy()
            # Denormalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            orig = (orig * std + mean).clip(0, 1)
            orig_uint8 = (orig * 255).astype(np.uint8)

            # Blend
            overlay = cv2.addWeighted(orig_uint8, 0.55, heatmap_rgb, 0.45, 0)
            heatmaps.append(overlay)

        self.model.zero_grad()
        return heatmaps


def try_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    max_maps: int = 8,
) -> List[np.ndarray]:
    """
    Safe wrapper: attempts Grad-CAM and returns empty list on any failure.
    Never raises to the caller.
    """
    try:
        gc = GradCAM(model)
        maps = gc.generate(tensor, device, max_maps=max_maps)
        gc.remove_hooks()
        return maps
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        return []
