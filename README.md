# GCP Marker Detection

Joint **localization** and **shape classification** of Ground Control Point
(GCP) markers in aerial drone imagery.

Given an aerial image, the model predicts:

1. The sub-pixel `(x, y)` center of the GCP marker.
2. The shape of the marker, one of `Cross`, `Square`, `L-Shape`.

---

## 1. Approach

### Why multi-class heatmap regression?

The task is a hybrid of keypoint localization and few-class classification.
Three obvious alternatives were considered:

| Approach | Problem |
|---|---|
| **Object detection (YOLO / DETR)** | Bounding boxes are overkill for sub-pixel center localization; marker radius is ~2 px (per EDA) so anchor- or DFL-based regression has little precision to lean on. |
| **Direct coordinate regression + classifier head** | Single-point MLP regression on full-resolution imagery is notoriously brittle; translation-variant and offers no spatial prior. |
| **Multi-class heatmap regression (chosen)** | One sigmoid-activated output channel per class. Peak location gives `(x, y)`; peak channel gives the class. Fully convolutional, translation-equivariant, precise, and a single forward pass handles both outputs. |

The approach mirrors keypoint/pose estimation (OpenPose, HRNet) and CenterNet,
which are the battle-tested recipes when you need pixel-accurate localization
plus per-point categorical labels.

### Architecture

- **Backbone:** `EfficientNet-B0` encoder + U-Net decoder (via
  `segmentation_models_pytorch`). B0 is the sweet spot — small enough to train
  fast on a single GPU, deep enough to handle the context-dependent markers
  (a white square on dirt vs on concrete looks very different).
- **Output head:** 3 channels, sigmoid-activated, one per class.
- **Decoding:** argmax across all channels gives the peak class + pixel. A
  **soft-argmax over a 5×5 window** around the peak gives sub-pixel accuracy,
  which matters since the evaluation measures pixel distance.

### Loss

**Gaussian Focal Loss (CornerNet / CenterNet style)**, not plain MSE.

With a 1024×576 output and a ~3 px Gaussian, positives are ≈30 of ~590 000
pixels (≈0.005%). MSE collapses: the background dominates the gradient and the
network learns to predict 0 everywhere. The penalty-reduced focal formulation
down-weights easy negatives by `(1 - gt)^β` and easy positives by
`(1 - pred)^α`, giving the model a meaningful gradient signal throughout
training.

---

## 2. EDA findings (and how they shaped the design)

The dataset is **not sanitized**:

- Class distribution is skewed: L-Shape 49.1%, Square 32.8%, Cross 17.7%,
  None/unknown ~0.4%.
- Some JSON entries point to missing image files.
- Some entries are missing the `mark` or `verified_shape` field.
- The ~0.4% `None`-class samples are dropped (below the signal-to-label-noise
  threshold, and there's no plausible prediction for them anyway).
- **Markers are tiny** — median radius ≈ 2 px, mean ≈ 4.7 px. This drives the
  choice of input resolution and target sigma.
- **Photometric variability is high** — mean-intensity crops span 80-160;
  contrast (std) spans 10-65. This motivates aggressive photometric
  augmentation.
- Marker X/Y positions have a wide marginal distribution but with
  concentration in the image interior. Random-crop augmentation is safe.

Run the EDA yourself to reproduce the plots in the brief:

```bash
python -m scripts.eda \
    --annotations data/train_dataset/curated_gcp_marks.json \
    --data-root data/train_dataset \
    --out eda_out
```

---

## 3. Training strategy

### Handling dataset quirks

- `src/dataset.py::load_records` uses `_safe_get` to walk nested dicts
  without KeyErrors. Records are dropped if any of `mark.x`, `mark.y`, or
  `verified_shape` is missing, or if the image file doesn't exist. The
  counts are logged so you know exactly what was filtered.
- `_normalize_shape` maps label variants (`L-Shaped`, `lshape`, `l_shape`)
  to canonical names.
- `train_val_split` splits **by survey** (`project/survey`) not by image.
  Multiple photos from the same flight capture the same physical marker;
  splitting by image would leak near-duplicates into validation and
  overstate accuracy.

### Augmentations (`src/augment.py`)

Geometric (label-preserving for all three classes — a rotated L-shape is
still an L-shape):

- Horizontal flip, vertical flip, 90° rotation.
- Small affine (±20° rotation, 0.9-1.1 scale, ±5% translation).
- Random sized crop for scale robustness.

Photometric:

- Brightness / contrast / HSV jitter.
- CLAHE to handle the wide brightness range seen in EDA.
- Noise / blur / JPEG compression to bridge the sim-to-real gap between
  high-quality and low-quality drone captures.

No coarse cutout or erasing — we don't want to occlude the marker itself.

### Optimization

- AdamW, lr 3e-4, weight decay 1e-4.
- Cosine schedule with 2 warmup epochs over 60 total.
- Mixed-precision (AMP) training.
- Gradient clipping at 1.0.
- Early stopping on validation mean pixel error after 10 stale epochs.

### Validation metric

Mean/median pixel error on correctly classified samples, plus plain
classification accuracy, plus the `% under {5, 10}` pixel thresholds (which
matches the evaluation brief). Best checkpoint saved by mean pixel error.

### Inference

- Test-time augmentation: horizontal and vertical flip, averaged.
- Predictions are decoded at model resolution then rescaled to the original
  image size using the image's own `(W, H)`, so differently-sized test
  images are handled correctly.
- Output JSON preserves the training format exactly, plus a `confidence`
  field for downstream QA.

---

## 4. Repository layout

```
gcp_detection/
├── configs/
│   └── default.yaml           # all hyperparameters
├── data/                       # you add this: train_dataset/ + test_dataset/
├── src/
│   ├── config.py               # YAML loader with attribute access
│   ├── utils.py                # seed, Gaussian rendering, peak decoding, coord scaling
│   ├── dataset.py              # safe JSON loader, survey-level split, datasets
│   ├── augment.py              # Albumentations pipelines (keypoint-aware)
│   ├── model.py                # UNet wrapped with sigmoid head
│   ├── loss.py                 # GaussianFocalLoss + MSE alternative
│   ├── train.py                # training entrypoint
│   └── infer.py                # inference entrypoint → predictions.json
└── scripts/
    ├── eda.py                  # reproduces the EDA plots
    ├── prepare_data.py         # validates + cleans the source JSON
    └── smoke_test.py           # 30-second sanity check
```

---

## 5. How to reproduce

### 5.1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2. Arrange data

```
data/
├── train_dataset/
│   ├── curated_gcp_marks.json
│   └── <project>/<survey>/<gcp_id>/<image>.JPG
└── test_dataset/
    └── <project>/<survey>/<gcp_id>/<image>.JPG
```

### 5.3. Sanity check

```bash
python -m scripts.smoke_test
```

Builds the model, runs a forward+backward pass on synthetic data, decodes
peaks. If this fails, the environment is broken; fix before training.

### 5.4. Train

```bash
python -m src.train --config configs/default.yaml
```

Checkpoints are written to `runs/default/`:

- `best.pt`: lowest validation mean pixel error.
- `last.pt`: final epoch.
- `epoch_XXX.pt`: periodic snapshots every `train.save_every` epochs.
- `tb/`: TensorBoard logs (`tensorboard --logdir runs/default/tb`).

On a single RTX 3090 / A10, ~60 epochs takes roughly 3-5 hours depending on
dataset size.

### 5.5. Inference → `predictions.json`

```bash
python -m src.infer \
    --config configs/default.yaml \
    --checkpoint runs/default/best.pt \
    --test-dir data/test_dataset \
    --output predictions.json
```

Output format matches the training labels exactly, plus a `confidence` field:

```json
{
  "231129_CTD/231129_CTD_GDA94/2/DJI_20231129130135_0428.JPG": {
    "mark": {"x": 2884.95, "y": 888.54},
    "verified_shape": "Cross",
    "confidence": 0.94
  }
}
```

### 5.6. Model weights

After training, place `runs/default/best.pt` on a cloud bucket and drop the
link here. Example:

> **Download:** `https://<your-bucket>/gcp-detection/best.pt`
>
> SHA-256: `<fill-in-after-upload>`

---

## 6. Challenges encountered & mitigations

| Challenge | Mitigation |
|---|---|
| **JSON has missing files, missing fields, and a `None` class.** | `_safe_get` walks the dict safely; `load_records` drops bad entries and logs counts. `_normalize_shape` maps label variants. |
| **Markers are 2-5 px wide** — most architectures lose them on downsampling. | Input resolution kept at 1024×576 (not 224×224); U-Net decoder preserves full resolution; target sigma tuned to 3 px. |
| **Massive positive/negative pixel imbalance** (~30 / 590 000). | Gaussian focal loss (CornerNet) instead of MSE. |
| **Risk of train/val leakage** — multiple photos of the same physical marker. | Split by `project/survey`, not per-image. |
| **Photometric variation across flights** (brightness 80-160, contrast 10-65). | Brightness/contrast/CLAHE/HSV augmentations. |
| **Marker cropped out by aggressive augmentation** → no target to supervise. | Loss applies a per-sample mask (`has_kp`) so cropped-out samples contribute only implicitly via BN stats. |
| **Sub-pixel accuracy required**, but argmax is integer-grid. | Soft-argmax refinement in a 5×5 window around the integer peak. |
| **Rotational ambiguity** of the three classes. | Flips + 90° rotations are all enabled — they're label-preserving for Cross, Square, and L-Shape. |

---

## 7. Possible extensions

- **Coarse-to-fine two-stage model**: a cheap full-image locator + a
  high-resolution 256×256 refinement crop around the peak. This is the
  standard trick when sub-pixel accuracy matters; easy to add on top of the
  current pipeline.
- **Larger encoder** (EfficientNet-B2 or ConvNeXt-Tiny) if the hardware
  budget allows.
- **Hard-negative mining**: crops with no marker, to teach the network to
  emit flat heatmaps for empty regions, reducing false positives in the
  wild.
- **Offset regression head** (CenterNet-style) instead of soft-argmax for
  even finer sub-pixel accuracy.
- **Distillation** to a smaller model for production if latency matters.

---

## 8. License

Internal take-home submission.
