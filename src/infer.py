"""Inference entry point: runs the trained model on the test directory and
writes a ``predictions.json`` file in the same format as the training labels.

Usage:
    python -m src.infer \
        --config configs/default.yaml \
        --checkpoint runs/default/best.pt \
        --test-dir data/test_dataset \
        --output predictions.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augment import infer_transforms
from .config import load_config
from .dataset import GCPInferenceDataset
from .model import build_model
from .utils import decode_heatmap, scale_xy

logger = logging.getLogger(__name__)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_images(root: Path) -> Tuple[List[Path], List[str]]:
    """Recursively list images under root, returning (abs_paths, rel_paths)."""
    abs_paths: List[Path] = []
    rel_paths: List[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            abs_paths.append(p)
            rel_paths.append(str(p.relative_to(root)).replace("\\", "/"))
    return abs_paths, rel_paths


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return imgs, metas


@torch.no_grad()
def run(
    config: str,
    checkpoint: str,
    test_dir: str,
    output: str,
    batch_size: int | None = None,
    tta: bool | None = None,
) -> None:
    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if batch_size is None:
        batch_size = cfg.infer.get("batch_size", 4)
    if tta is None:
        tta = cfg.infer.get("tta", True)

    # --------------------- Model ---------------------
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    # --------------------- Data ---------------------
    test_root = Path(test_dir)
    abs_paths, rel_paths = find_images(test_root)
    logger.info("Found %d images under %s", len(abs_paths), test_root)

    image_size = tuple(cfg.data.image_size)  # (W, H)
    ds = GCPInferenceDataset(abs_paths, rel_paths, image_size, infer_transforms(image_size))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.train.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate,
    )

    classes: List[str] = list(cfg.data.classes)
    peak_window = cfg.infer.get("peak_window", 5)

    predictions: Dict[str, Dict] = {}

    for imgs, metas in tqdm(loader, desc="infer"):
        imgs = imgs.to(device, non_blocking=True)

        preds = model(imgs)
        if tta:
            # Horizontal flip TTA
            pred_hf = model(torch.flip(imgs, dims=[-1]))
            pred_hf = torch.flip(pred_hf, dims=[-1])
            # Vertical flip TTA
            pred_vf = model(torch.flip(imgs, dims=[-2]))
            pred_vf = torch.flip(pred_vf, dims=[-2])
            preds = (preds + pred_hf + pred_vf) / 3.0

        for i, meta in enumerate(metas):
            cls_idx, px, py, conf = decode_heatmap(preds[i], peak_window=peak_window)
            # Scale from model-input coordinates back to original image size
            ox, oy = scale_xy(px, py, image_size, meta["orig_size"])
            predictions[meta["rel_path"]] = {
                "mark": {"x": float(ox), "y": float(oy)},
                "verified_shape": classes[cls_idx],
                "confidence": float(conf),
            }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Wrote %d predictions → %s", len(predictions), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--output", type=str, default="predictions.json")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    a = parse_args()
    run(
        config=a.config,
        checkpoint=a.checkpoint,
        test_dir=a.test_dir,
        output=a.output,
        batch_size=a.batch_size,
        tta=not a.no_tta,
    )
