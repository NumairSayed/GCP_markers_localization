"""Utility helpers: heatmap rendering, peak decoding, reproducibility."""
from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Gaussian heatmap target rendering
# ---------------------------------------------------------------------------
def render_gaussian(
    heatmap: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
) -> np.ndarray:
    """Paste a 2D Gaussian of unit peak into ``heatmap`` at (cx, cy).

    Operates in-place with max-merge so overlapping peaks keep their amplitude.
    """
    H, W = heatmap.shape
    if not (0 <= cx < W and 0 <= cy < H):
        return heatmap

    radius = int(3 * sigma + 0.5)
    x0 = max(int(cx - radius), 0)
    x1 = min(int(cx + radius) + 1, W)
    y0 = max(int(cy - radius), 0)
    y1 = min(int(cy + radius) + 1, H)

    if x1 <= x0 or y1 <= y0:
        return heatmap

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    g = g.astype(heatmap.dtype)
    np.maximum(heatmap[y0:y1, x0:x1], g, out=heatmap[y0:y1, x0:x1])
    return heatmap


def build_target_heatmap(
    num_classes: int,
    H: int,
    W: int,
    cx: float,
    cy: float,
    class_idx: int,
    sigma: float,
) -> np.ndarray:
    """Create ``(C, H, W)`` target with a single Gaussian on ``class_idx``."""
    target = np.zeros((num_classes, H, W), dtype=np.float32)
    render_gaussian(target[class_idx], cx, cy, sigma)
    return target


# ---------------------------------------------------------------------------
# Peak decoding
# ---------------------------------------------------------------------------
def decode_heatmap(
    heatmap: torch.Tensor,
    peak_window: int = 5,
) -> Tuple[int, float, float, float]:
    """Decode a ``(C, H, W)`` heatmap into (class_idx, x, y, confidence).

    The coordinates are refined with a soft-argmax inside a small window
    centered on the argmax pixel, which gives sub-pixel accuracy.
    """
    assert heatmap.dim() == 3, "expected (C, H, W)"
    C, H, W = heatmap.shape

    flat = heatmap.reshape(C, -1)
    max_per_class, idx_per_class = flat.max(dim=1)
    class_idx = int(max_per_class.argmax().item())
    flat_idx = int(idx_per_class[class_idx].item())
    peak_y, peak_x = divmod(flat_idx, W)
    confidence = float(max_per_class[class_idx].item())

    # Sub-pixel refinement via soft-argmax in a local window
    r = peak_window // 2
    y0, y1 = max(peak_y - r, 0), min(peak_y + r + 1, H)
    x0, x1 = max(peak_x - r, 0), min(peak_x + r + 1, W)
    patch = heatmap[class_idx, y0:y1, x0:x1]

    # Numerical stabilisation: subtract min, square to sharpen (helps when the
    # network outputs are soft) before centroiding.
    patch = (patch - patch.min()).clamp(min=0)
    if patch.sum() < 1e-6:
        return class_idx, float(peak_x), float(peak_y), confidence
    w = patch / patch.sum()

    ys = torch.arange(y0, y1, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.arange(x0, x1, device=heatmap.device, dtype=heatmap.dtype)
    cy = float((w.sum(dim=1) * ys).sum().item())
    cx = float((w.sum(dim=0) * xs).sum().item())

    return class_idx, cx, cy, confidence


# ---------------------------------------------------------------------------
# Coordinate transforms between model input and original image
# ---------------------------------------------------------------------------
def scale_xy(
    x: float,
    y: float,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Rescale ``(x, y)`` from ``src_size`` to ``dst_size``.

    Sizes are ``(W, H)``.
    """
    sw, sh = src_size
    dw, dh = dst_size
    return x * dw / sw, y * dh / sh
