"""Dataset, annotation loading, and train/val split.

The source JSON is not perfectly sanitized: entries may point to missing files,
may lack the ``mark`` or ``verified_shape`` fields, or use the ``None`` class
(~0.4%). This module filters such entries with safe dictionary access and logs
a summary.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import build_target_heatmap

logger = logging.getLogger(__name__)

# Kept in sync with configs/default.yaml ``data.classes``
DEFAULT_CLASSES: Tuple[str, ...] = ("Cross", "Square", "L-Shape")


# ---------------------------------------------------------------------------
# Annotation record
# ---------------------------------------------------------------------------
@dataclass
class GCPRecord:
    rel_path: str
    abs_path: Path
    x: float
    y: float
    shape: str
    class_idx: int


def _safe_get(d: Optional[dict], *keys, default=None):
    """Nested safe getter, tolerates missing keys and non-dict intermediates."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
        if cur is None:
            return default
    return cur


def _normalize_shape(raw: str) -> Optional[str]:
    """Map labels to canonical class names; returns None for unknown/None."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    s_low = s.lower().replace("_", "-").replace(" ", "-")
    if s_low in {"none", "null", ""}:
        return None
    mapping = {
        "cross": "Cross",
        "x": "Cross",
        "square": "Square",
        "rectangle": "Square",
        "l": "L-Shape",
        "l-shape": "L-Shape",
        "l-shaped": "L-Shape",
        "lshape": "L-Shape",
    }
    return mapping.get(s_low, None)


def load_records(
    annotations_path: str | Path,
    data_root: str | Path,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> List[GCPRecord]:
    """Load and sanitize the GCP annotation JSON."""
    annotations_path = Path(annotations_path)
    data_root = Path(data_root)

    with open(annotations_path, "r") as f:
        raw = json.load(f)

    cls_to_idx = {c: i for i, c in enumerate(classes)}

    kept: List[GCPRecord] = []
    missing_file = 0
    missing_fields = 0
    bad_class = 0
    none_class = 0

    for rel_path, entry in raw.items():
        if not isinstance(entry, dict):
            missing_fields += 1
            continue

        x = _safe_get(entry, "mark", "x")
        y = _safe_get(entry, "mark", "y")
        shape_raw = _safe_get(entry, "verified_shape")

        if x is None or y is None or shape_raw is None:
            missing_fields += 1
            continue

        canon = _normalize_shape(shape_raw)
        if canon is None:
            none_class += 1
            continue
        if canon not in cls_to_idx:
            bad_class += 1
            continue

        # The JSON can hold either ``project/survey/gcp/file.JPG`` relative
        # paths or paths that start with ``train_dataset/``. Try both.
        abs_path = (data_root / rel_path).resolve()
        if not abs_path.exists():
            parts = Path(rel_path).parts
            found = False
            # Try stripping the first N path components (handles e.g.
            # ``train_dataset/project/...`` when data_root already points at
            # ``train_dataset``).
            for strip in (1, 2):
                if len(parts) > strip:
                    alt = (data_root.joinpath(*parts[strip:])).resolve()
                    if alt.exists():
                        abs_path = alt
                        found = True
                        break
            if not found:
                missing_file += 1
                continue

        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            missing_fields += 1
            continue

        kept.append(
            GCPRecord(
                rel_path=rel_path,
                abs_path=abs_path,
                x=fx,
                y=fy,
                shape=canon,
                class_idx=cls_to_idx[canon],
            )
        )

    logger.info(
        "Loaded %d records (%d missing files, %d missing/invalid fields, "
        "%d None-class, %d unknown-class)",
        len(kept), missing_file, missing_fields, none_class, bad_class,
    )
    return kept


def train_val_split(
    records: List[GCPRecord],
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[GCPRecord], List[GCPRecord]]:
    """Stratified split by shape class and by survey folder.

    Splitting by survey avoids leakage between train and val when multiple
    photographs of the same flight (and thus the same physical marker) exist.
    """
    rng = np.random.default_rng(seed)

    # Group records by (class, survey). Survey = first two path components.
    groups: Dict[Tuple[str, str], List[GCPRecord]] = {}
    for r in records:
        parts = Path(r.rel_path).parts
        survey = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups.setdefault((r.shape, survey), []).append(r)

    train, val = [], []
    # Per class, shuffle the surveys and move ~val_split of them to val.
    per_class_surveys: Dict[str, List[str]] = {}
    for (cls, survey) in groups:
        per_class_surveys.setdefault(cls, []).append(survey)

    val_surveys = set()
    for cls, surveys in per_class_surveys.items():
        surveys = sorted(set(surveys))
        rng.shuffle(surveys)
        n_val = max(1, int(round(len(surveys) * val_split)))
        val_surveys.update(surveys[:n_val])

    for r in records:
        parts = Path(r.rel_path).parts
        survey = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        (val if survey in val_surveys else train).append(r)

    logger.info("Split: %d train / %d val (across %d val surveys)",
                len(train), len(val), len(val_surveys))
    return train, val


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class GCPDataset(Dataset):
    """Returns ``image`` tensor, target heatmap, and metadata.

    The image is read at original resolution and the keypoint is carried along
    so the Albumentations pipeline can apply geometric augmentations
    consistently.  Final resize to model input happens inside the transform.
    """

    def __init__(
        self,
        records: List[GCPRecord],
        image_size: Tuple[int, int],           # (W, H)
        num_classes: int,
        sigma: float,
        transforms,                             # albumentations.Compose
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.num_classes = num_classes
        self.sigma = sigma
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # cv2.imread is faster than PIL for big JPEGs
        img = cv2.imread(str(rec.abs_path), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback: skip to next record rather than failing the whole epoch
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H0, W0 = img.shape[:2]
        keypoints = [(float(rec.x), float(rec.y))]

        augmented = self.transforms(
            image=img,
            keypoints=keypoints,
            class_labels=[rec.class_idx],
        )
        img_t = augmented["image"]                  # tensor (3, H, W)
        kps = augmented["keypoints"]
        labels = augmented["class_labels"]

        _, H, W = img_t.shape

        if len(kps) == 0:
            # The keypoint was cropped out. Emit empty target so loss ignores.
            target = np.zeros((self.num_classes, H, W), dtype=np.float32)
            has_kp = 0.0
            cx, cy, cls = -1.0, -1.0, -1
        else:
            cx, cy = kps[0]
            cls = int(labels[0])
            target = build_target_heatmap(
                self.num_classes, H, W, cx, cy, cls, self.sigma,
            )
            has_kp = 1.0

        meta = {
            "rel_path": rec.rel_path,
            "orig_size": (W0, H0),
            "has_kp": has_kp,
            "gt_xy": (cx, cy),
            "gt_class": cls,
        }
        return img_t, torch.from_numpy(target), meta


class GCPInferenceDataset(Dataset):
    """Dataset for unlabelled inference."""

    def __init__(
        self,
        image_paths: List[Path],
        rel_paths: List[str],
        image_size: Tuple[int, int],
        transforms,
    ) -> None:
        assert len(image_paths) == len(rel_paths)
        self.image_paths = image_paths
        self.rel_paths = rel_paths
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            # Return zeros as a placeholder so the batch does not crash.
            W, H = self.image_size
            img = np.zeros((H, W, 3), dtype=np.uint8)
            orig = (W, H)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig = (img.shape[1], img.shape[0])

        out = self.transforms(image=img)
        return out["image"], {"rel_path": self.rel_paths[idx], "orig_size": orig}
