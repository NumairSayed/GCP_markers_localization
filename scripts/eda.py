"""Exploratory Data Analysis for the GCP dataset.

Reproduces the plots provided in the task brief:
  * Class distribution (Cross / Square / L-Shape / None / missing)
  * Spatial distribution of marker centers + marginal X/Y histograms
  * Estimated marker radius distribution (via simple local-intensity analysis)
  * Brightness / contrast per-image crops around the marker
  * Sample images with their annotations overlaid

Outputs are written to ``eda_out/``.

Usage:
    python -m scripts.eda \
        --annotations data/train_dataset/curated_gcp_marks.json \
        --data-root data/train_dataset \
        --out eda_out
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Importing via absolute path so the script runs as `python scripts/eda.py`
# without requiring the package to be installed.
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import _normalize_shape, _safe_get  # noqa: E402

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
def scan_annotations(ann_path: Path, data_root: Path) -> pd.DataFrame:
    with open(ann_path, "r") as f:
        raw = json.load(f)

    rows = []
    for rel, entry in raw.items():
        x = _safe_get(entry, "mark", "x")
        y = _safe_get(entry, "mark", "y")
        shape_raw = _safe_get(entry, "verified_shape")
        canonical = _normalize_shape(shape_raw) if shape_raw else None

        abs_path = data_root / rel
        exists = abs_path.exists()

        rows.append(
            dict(
                rel_path=rel,
                x=x, y=y,
                shape_raw=shape_raw,
                shape=canonical,
                exists=exists,
                abs_path=str(abs_path),
            )
        )
    df = pd.DataFrame(rows)
    return df


def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    cnt = Counter()
    for _, r in df.iterrows():
        if pd.isna(r["shape_raw"]) or r["shape_raw"] is None:
            cnt["missing_field"] += 1
        elif r["shape"] is None:
            cnt["None/other"] += 1
        else:
            cnt[r["shape"]] += 1

    total = sum(cnt.values())
    summary = pd.DataFrame(
        [(k, v, 100.0 * v / total) for k, v in cnt.items()],
        columns=["class", "count", "percentage"],
    ).sort_values("count", ascending=False).reset_index(drop=True)
    summary.to_csv(out / "class_distribution.csv", index=False)
    logger.info("Class distribution:\n%s", summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="class", y="count", ax=ax)
    for i, r in summary.iterrows():
        ax.text(i, r["count"], f"{r['count']} ({r['percentage']:.1f}%)",
                ha="center", va="bottom")
    ax.set_title("Class distribution")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=120)
    plt.close(fig)


def plot_spatial_distribution(df: pd.DataFrame, out: Path) -> None:
    valid = df.dropna(subset=["x", "y"])
    if valid.empty:
        logger.warning("No valid (x, y) records for spatial plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D histogram of marker centers
    h, xe, ye = np.histogram2d(
        valid["x"].astype(float), valid["y"].astype(float),
        bins=[60, 40],
        range=[[0, max(4000, valid["x"].max())], [0, max(1500, valid["y"].max())]],
    )
    axes[0].imshow(
        h.T, origin="upper",
        extent=[xe[0], xe[-1], ye[-1], ye[0]],
        cmap="hot", aspect="auto",
    )
    axes[0].set_title("Spatial distribution of marker centers")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    # Marginal distributions
    axes[1].hist(valid["x"].astype(float), bins=40, alpha=0.6, label="X", color="tab:blue")
    axes[1].hist(valid["y"].astype(float), bins=40, alpha=0.6, label="Y", color="tab:red")
    axes[1].legend()
    axes[1].set_title("Marginal distributions of X and Y coordinates")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out / "spatial_distribution.png", dpi=120)
    plt.close(fig)


def estimate_marker_radius(img_gray: np.ndarray, cx: float, cy: float,
                           win: int = 40) -> Optional[float]:
    """Very rough radius estimate via local intensity deviation.

    This is heuristic and only used for EDA, not training.
    """
    H, W = img_gray.shape
    x0, x1 = max(int(cx) - win, 0), min(int(cx) + win, W)
    y0, y1 = max(int(cy) - win, 0), min(int(cy) + win, H)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = img_gray[y0:y1, x0:x1].astype(np.float32)
    # Local contrast (std over a 5x5 neighborhood of the center)
    local = patch[(patch.shape[0] // 2 - 5):(patch.shape[0] // 2 + 5),
                  (patch.shape[1] // 2 - 5):(patch.shape[1] // 2 + 5)]
    if local.size == 0:
        return None
    mean, std = patch.mean(), patch.std()
    if std < 1e-6:
        return None
    bright = np.abs(patch - mean) > std
    # Estimate radius = half of the bbox side of the bright region around center
    ys, xs = np.where(bright)
    if len(xs) < 5:
        return None
    dx = xs - (patch.shape[1] / 2)
    dy = ys - (patch.shape[0] / 2)
    r = float(np.sqrt(dx ** 2 + dy ** 2).mean())
    return r


def plot_radius_and_photometric(df: pd.DataFrame, out: Path, max_samples: int = 500) -> None:
    radii: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    rng = random.Random(42)
    candidates = df[df["exists"] & df["x"].notna() & df["y"].notna()]
    sample = candidates.sample(n=min(max_samples, len(candidates)), random_state=42)

    for _, r in tqdm(sample.iterrows(), total=len(sample), desc="photometric"):
        img = cv2.imread(r["abs_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        rad = estimate_marker_radius(img, float(r["x"]), float(r["y"]))
        if rad is not None:
            radii.append(rad)

        H, W = img.shape
        x, y = int(r["x"]), int(r["y"])
        x0, x1 = max(x - 30, 0), min(x + 30, W)
        y0, y1 = max(y - 30, 0), min(y + 30, H)
        if x1 > x0 and y1 > y0:
            patch = img[y0:y1, x0:x1].astype(np.float32)
            means.append(float(patch.mean()))
            stds.append(float(patch.std()))

    if radii:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(radii, bins=40, kde=True, ax=ax)
        ax.set_title("Estimated marker radius distribution")
        ax.set_xlabel("Radius (pixels)")
        plt.tight_layout()
        plt.savefig(out / "radius_distribution.png", dpi=120)
        plt.close(fig)
        logger.info(
            "Marker radius — median=%.1f mean=%.1f std=%.1f (n=%d)",
            float(np.median(radii)), float(np.mean(radii)), float(np.std(radii)), len(radii),
        )

    if means and stds:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(means, bins=30, color="tab:blue")
        axes[0].set_title("Mean Pixel Intensity (Brightness)")
        axes[0].set_ylabel("Count")
        axes[1].hist(stds, bins=30, color="tab:orange")
        axes[1].set_title("Standard Deviation (Contrast)")
        plt.tight_layout()
        plt.savefig(out / "photometric.png", dpi=120)
        plt.close(fig)


def plot_samples(df: pd.DataFrame, out: Path, n_per_class: int = 3) -> None:
    classes = ["Cross", "Square", "L-Shape"]
    picks = []
    for c in classes:
        avail = df[(df["shape"] == c) & df["exists"]]
        picks.extend(avail.sample(n=min(n_per_class, len(avail)), random_state=42).to_dict("records"))

    if not picks:
        return

    rows = 3
    cols = n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_2d(axes)

    for idx, rec in enumerate(picks):
        r, c = divmod(idx, cols)
        img = cv2.imread(rec["abs_path"], cv2.IMREAD_COLOR)
        if img is None:
            axes[r, c].axis("off")
            axes[r, c].set_title(f"Missing\n{Path(rec['rel_path']).name}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y = int(rec["x"]), int(rec["y"])
        # Crop 256x256 around marker for visibility
        H, W = img.shape[:2]
        x0, x1 = max(x - 128, 0), min(x + 128, W)
        y0, y1 = max(y - 128, 0), min(y + 128, H)
        crop = img[y0:y1, x0:x1].copy()
        cv2.drawMarker(
            crop, (x - x0, y - y0), (0, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2,
        )
        axes[r, c].imshow(crop)
        axes[r, c].set_title(f"{rec['shape']} at ({rec['x']:.1f}, {rec['y']:.1f})\n{Path(rec['rel_path']).name}", fontsize=8)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out / "samples.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True, type=str)
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--out", default="eda_out", type=str)
    p.add_argument("--max-samples", default=500, type=int)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = scan_annotations(Path(args.annotations), Path(args.data_root))
    logger.info("Total annotation rows: %d", len(df))
    logger.info("Missing files: %d", int((~df["exists"]).sum()))

    df.to_csv(out / "annotations_scan.csv", index=False)

    plot_class_distribution(df, out)
    plot_spatial_distribution(df, out)
    plot_samples(df, out)
    plot_radius_and_photometric(df, out, max_samples=args.max_samples)

    logger.info("EDA outputs written to %s", out)


if __name__ == "__main__":
    main()
