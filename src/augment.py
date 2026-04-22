"""Augmentation pipelines for training and evaluation.

Albumentations is used because it handles keypoint-aware geometric transforms
(resize, flip, rotate, scale) correctly and drops keypoints that fall outside
the frame after a crop — we leverage that to build an "empty target" signal
for the heatmap loss when a crop accidentally removes the marker.

Design notes for GCP data:
- Markers are rotationally ambiguous (a rotated L-Shape is still an L-Shape,
  a rotated square is still a square, a rotated cross is still a cross). So
  horizontal, vertical, and 90° rotations are all label-preserving.
- Brightness varies substantially (mean intensity 80-160 per EDA), so
  photometric augmentation is important for generalization.
- We avoid coarse dropout / cutout that could erase the marker itself.

Every transform used here is part of the stable Albumentations core API that
has been present since at least 1.3.x and remains in 2.x, to avoid breakage
on `pip install` drift.
"""
from __future__ import annotations

import inspect
from typing import Tuple

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _keypoint_params() -> A.KeypointParams:
    # remove_invisible=True lets crops that lose the marker be caught in the
    # dataset as empty targets (see GCPDataset.__getitem__).
    return A.KeypointParams(
        format="xy",
        label_fields=["class_labels"],
        remove_invisible=True,
        angle_in_degrees=True,
    )


def _image_compression(p: float = 0.2) -> A.BasicTransform:
    """Albumentations renamed ImageCompression kwargs between versions.

    - Older (<=1.4.x): ``quality_lower`` / ``quality_upper``.
    - Newer (>=1.4.x): ``quality_range=(lo, hi)``.

    Build the transform based on which signature is actually supported.
    """
    try:
        params = inspect.signature(A.ImageCompression.__init__).parameters
    except (TypeError, ValueError):
        return A.ImageCompression(p=p)

    if "quality_range" in params:
        return A.ImageCompression(quality_range=(60, 100), p=p)
    if "quality_lower" in params:
        return A.ImageCompression(quality_lower=60, quality_upper=100, p=p)
    return A.ImageCompression(p=p)


def train_transforms(image_size: Tuple[int, int]) -> A.Compose:
    W, H = image_size
    return A.Compose(
        [
            # Geometric augs FIRST (can swap H/W)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                rotate=(-20, 20),
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
            ),
            # Resize LAST to guarantee fixed output shape
            A.Resize(height=H, width=W, interpolation=cv2.INTER_AREA),
            # Photometric augs after resize (cheaper on smaller image)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25, contrast_limit=0.25, p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=1.0,
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.2,
            ),
            _image_compression(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=_keypoint_params(),
    )

def val_transforms(image_size: Tuple[int, int]) -> A.Compose:
    W, H = image_size
    return A.Compose(
        [
            A.Resize(height=H, width=W, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        keypoint_params=_keypoint_params(),
    )


def infer_transforms(image_size: Tuple[int, int]) -> A.Compose:
    W, H = image_size
    return A.Compose(
        [
            A.Resize(height=H, width=W, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
