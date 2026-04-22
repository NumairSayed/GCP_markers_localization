"""Training entry point.

Usage:
    python -m src.train --config configs/default.yaml

The validation metric we optimize is *mean pixel error* on correctly classified
samples plus a classification accuracy component; both are tracked and the
best checkpoint is saved.
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW

# `torch.amp.GradScaler`/`autocast` (with device_type argument) was introduced
# in PyTorch 2.3. For older 2.0-2.2 installs we fall back to `torch.cuda.amp`.
try:
    from torch.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
    from torch.amp import autocast as _autocast      # type: ignore[attr-defined]
    _AMP_NEW_API = True
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler
    from torch.cuda.amp import autocast as _autocast
    _AMP_NEW_API = False


def make_scaler(enabled: bool):
    if _AMP_NEW_API:
        return _GradScaler("cuda", enabled=enabled)
    return _GradScaler(enabled=enabled)


def make_autocast(device_type: str, enabled: bool):
    if _AMP_NEW_API:
        return _autocast(device_type=device_type, enabled=enabled)
    # Old API only autocasts on CUDA.
    return _autocast(enabled=enabled and device_type == "cuda")
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .augment import train_transforms, val_transforms
from .config import load_config, save_config
from .dataset import GCPDataset, load_records, train_val_split
from .loss import build_loss
from .model import build_model
from .utils import decode_heatmap, scale_xy, seed_everything

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collate: keep metadata as a list of dicts (default collate chokes on tuples)
# ---------------------------------------------------------------------------
def collate(batch):
    # print(type(batch), type(batch[0]), len(batch))
    # print([b for b in batch])
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return imgs, targets, metas


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, fn)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, image_size, peak_window) -> Dict[str, float]:
    model.eval()
    errs_px: List[float] = []
    correct_cls = 0
    total = 0
    W_in, H_in = image_size

    for imgs, _, metas in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)  # (B, C, H, W)

        for i, meta in enumerate(metas):
            if meta["has_kp"] < 0.5:
                continue
            pred_cls, px, py, _ = decode_heatmap(preds[i], peak_window=peak_window)
            gt_cls = meta["gt_class"]
            gx, gy = meta["gt_xy"]

            # The target was rendered on the augmented image so it is already
            # in model-input pixel space; compare directly.
            err = float(math.hypot(px - gx, py - gy))
            errs_px.append(err)
            total += 1
            if pred_cls == gt_cls:
                correct_cls += 1

    if not errs_px:
        return {"val/mean_px": float("nan"),
                "val/median_px": float("nan"),
                "val/cls_acc": 0.0,
                "val/pct_under_10px": 0.0}

    errs = np.asarray(errs_px)
    return {
        "val/mean_px": float(errs.mean()),
        "val/median_px": float(np.median(errs)),
        "val/cls_acc": correct_cls / max(1, total),
        "val/pct_under_10px": float((errs < 10).mean()),
        "val/pct_under_5px": float((errs < 5).mean()),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ---------------------------- Data ----------------------------
    records = load_records(
        cfg.data.annotations, cfg.data.train_dir, classes=cfg.data.classes,
    )
    train_records, val_records = train_val_split(
        records, val_split=cfg.data.val_split, seed=cfg.seed,
    )

    image_size = tuple(cfg.data.image_size)  # (W, H)
    train_ds = GCPDataset(
        train_records,
        image_size=image_size,
        num_classes=len(cfg.data.classes),
        sigma=cfg.data.heatmap_sigma,
        transforms=train_transforms(image_size),
    )
    val_ds = GCPDataset(
        val_records,
        image_size=image_size,
        num_classes=len(cfg.data.classes),
        sigma=cfg.data.heatmap_sigma,
        transforms=val_transforms(image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=cfg.train.num_workers > 0,
    )

    # ---------------------------- Model ----------------------------
    model = build_model(cfg.model).to(device)
    criterion = build_loss(cfg.train.loss)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    total_steps = len(train_loader) * cfg.train.epochs
    warmup_steps = len(train_loader) * cfg.train.warmup_epochs
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = make_scaler(cfg.train.amp and device.type == "cuda")

    # ---------------------------- Train ----------------------------
    best_score = math.inf  # we minimize mean pixel error
    bad_epochs = 0
    global_step = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for it, (imgs, targets, metas) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = torch.tensor(
                [m["has_kp"] for m in metas], device=device, dtype=torch.float32,
            )

            optimizer.zero_grad(set_to_none=True)
            with make_autocast(device.type, cfg.train.amp):
                preds = model(imgs)
                loss = criterion(preds, targets, mask)

            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running = 0.9 * running + 0.1 * float(loss.item()) if it else float(loss.item())
            global_step += 1

            if global_step % cfg.train.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                pbar.set_postfix(loss=f"{running:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # -------- Validation --------
        if (epoch + 1) % cfg.train.val_every == 0:
            metrics = validate(model, val_loader, device, image_size, cfg.infer.peak_window)
            for k, v in metrics.items():
                writer.add_scalar(k, v, global_step)
            logger.info(
                "epoch %d | loss=%.4f | val mean_px=%.2f median_px=%.2f cls_acc=%.3f "
                "(<5px=%.3f <10px=%.3f) | %.1fs",
                epoch, running,
                metrics["val/mean_px"], metrics["val/median_px"],
                metrics["val/cls_acc"],
                metrics.get("val/pct_under_5px", 0.0),
                metrics.get("val/pct_under_10px", 0.0),
                time.time() - t0,
            )

            score = metrics["val/mean_px"]
            if score < best_score:
                best_score = score
                bad_epochs = 0
                ckpt = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "cfg": dict(cfg),
                    "metrics": metrics,
                }
                torch.save(ckpt, out_dir / "best.pt")
                logger.info("  → saved new best (mean_px=%.3f)", score)
            else:
                bad_epochs += 1

            if (epoch + 1) % cfg.train.save_every == 0:
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "cfg": dict(cfg)},
                    out_dir / f"epoch_{epoch:03d}.pt",
                )

            if bad_epochs >= cfg.train.early_stopping_patience:
                logger.info("Early stopping after %d epochs without improvement.", bad_epochs)
                break

    # Always save last
    torch.save({"model": model.state_dict(), "cfg": dict(cfg)}, out_dir / "last.pt")
    writer.close()
    logger.info("Training complete. Best mean_px: %.3f", best_score)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    run(args.config)
