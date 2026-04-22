"""Smoke test: builds the model, runs a forward+backward pass on a tiny
synthetic batch, and decodes the output. Catches wiring issues fast.

Usage:
    python -m scripts.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.loss import build_loss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import build_target_heatmap, decode_heatmap  # noqa: E402


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Small input for speed
    W, H = 256, 256
    B = 2

    model = build_model(cfg.model).to(device)
    loss_fn = build_loss(cfg.train.loss)

    # Synthetic images + targets
    imgs = torch.randn(B, 3, H, W, device=device)
    targets = np.zeros((B, cfg.model.num_classes, H, W), dtype=np.float32)
    for b in range(B):
        cx = np.random.uniform(32, W - 32)
        cy = np.random.uniform(32, H - 32)
        cls = np.random.randint(cfg.model.num_classes)
        t = build_target_heatmap(
            cfg.model.num_classes, H, W, cx, cy, cls, sigma=cfg.data.heatmap_sigma,
        )
        targets[b] = t
    targets_t = torch.from_numpy(targets).to(device)

    preds = model(imgs)
    assert preds.shape == (B, cfg.model.num_classes, H, W), preds.shape
    print("Forward OK:", preds.shape)

    loss = loss_fn(preds, targets_t)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")

    # Decode
    for b in range(B):
        cls, x, y, conf = decode_heatmap(preds[b].detach(), peak_window=5)
        print(f"  sample {b}: class={cls} xy=({x:.1f},{y:.1f}) conf={conf:.3f}")

    print("Smoke test OK.")


if __name__ == "__main__":
    main()
