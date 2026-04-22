"""Model definition: U-Net that outputs one heatmap channel per shape class.

At inference the peak location across all channels gives both the (x, y)
coordinates (via soft-argmax refinement) and the predicted class (via the
channel index of the peak).

We rely on segmentation_models_pytorch for the encoder-decoder because it
gives us a wide choice of ImageNet-pretrained encoders with consistent APIs.
"""
from __future__ import annotations

from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    arch = cfg.get("arch", "Unet")
    encoder_name = cfg.get("encoder_name", "efficientnet-b0")
    encoder_weights = cfg.get("encoder_weights", "imagenet")
    in_channels = cfg.get("in_channels", 3)
    num_classes = cfg.get("num_classes", 3)

    model_cls = getattr(smp, arch)
    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # we apply sigmoid in the head for numerical control
    )
    return HeatmapHead(model)


class HeatmapHead(nn.Module):
    """Wraps an SMP model and applies a sigmoid to produce heatmaps in [0, 1]."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)          # (B, C, H, W)
        # Bound the logits to prevent the CenterNet focal loss from saturating
        # to 0/1 in fp16.
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        return torch.sigmoid(logits)
