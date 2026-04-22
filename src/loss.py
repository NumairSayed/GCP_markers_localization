"""Heatmap losses.

Implements the Gaussian focal loss from CornerNet/CenterNet, which is the de
facto standard for Gaussian-heatmap keypoint detection. It is much better
behaved than plain MSE when positives are a handful of pixels in a field of
~600k, as is the case here (median marker radius ~2 px, image >= 1024x576).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFocalLoss(nn.Module):
    """Penalty-reduced focal loss, from CornerNet (Law & Deng, 2018).

    Positives are pixels whose Gaussian target is exactly 1.0.
    Other pixels are negatives whose loss is down-weighted by
    ``(1 - gt) ** beta``, which suppresses the huge background easily.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,     # (B, C, H, W) already sigmoided
        target: torch.Tensor,   # (B, C, H, W)
        mask: torch.Tensor | None = None,  # (B,) 1.0 if sample has a keypoint
    ) -> torch.Tensor:
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        pos = target.eq(1).float()
        neg = 1.0 - pos

        pos_loss = -((1.0 - pred) ** self.alpha) * torch.log(pred) * pos
        neg_loss = (
            -((1.0 - target) ** self.beta)
            * (pred ** self.alpha)
            * torch.log(1.0 - pred)
            * neg
        )

        if mask is not None:
            mask = mask.view(-1, 1, 1, 1).to(pred.dtype)
            pos_loss = pos_loss * mask
            neg_loss = neg_loss * mask

        num_pos = pos.sum().clamp(min=1.0)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class HeatmapMSELoss(nn.Module):
    """Simple weighted MSE alternative."""

    def __init__(self, pos_weight: float = 100.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = 1.0 + (self.pos_weight - 1.0) * target
        loss = F.mse_loss(pred, target, reduction="none") * weight
        if mask is not None:
            mask = mask.view(-1, 1, 1, 1).to(pred.dtype)
            loss = loss * mask
        return loss.mean()


def build_loss(cfg) -> nn.Module:
    t = cfg.get("type", "gaussian_focal")
    if t == "gaussian_focal":
        return GaussianFocalLoss(
            alpha=cfg.get("alpha", 2.0), beta=cfg.get("beta", 4.0),
        )
    if t == "mse":
        return HeatmapMSELoss(pos_weight=cfg.get("pos_weight", 100.0))
    raise ValueError(f"Unknown loss type: {t}")
