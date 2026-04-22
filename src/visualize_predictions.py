"""
visualize_predictions.py
------------------------
Annotates test images using predictions.json and saves:
  1. Downscaled full image with marker, shape label, and confidence overlaid
  2. A tight crop centred on the predicted marker
  3. A summary grid of all crops

Handles real-world conditions:
  - Any image resolution (4000x3000, 5472x3648, etc.)
  - Markers near edges (crops are clamped + offset, never black-padded)
  - Adaptive annotation scaling based on actual image resolution

Defaults:
    --predictions   ./predictions.json
    --dataset_dir   ./GCP_Assignment_Datasets/test_dataset
    --output_dir    ../viz_test_predictions

Usage:
    python visualize_predictions.py
    python visualize_predictions.py --predictions ./predictions.json --dataset_dir ./test --output_dir ./out
    python visualize_predictions.py --crop_size 512 --no_grid
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# Colours (BGR for OpenCV)
# ─────────────────────────────────────────────
SHAPE_COLORS = {
    "Cross":    (50,  205,  50),   # lime green
    "Square":   (0,   165, 255),   # orange
    "L-Shape":  (147,  20, 255),   # purple
    "L-Shaped": (147,  20, 255),   # purple (alias)
    "Unknown":  (180, 180, 180),   # grey fallback
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# Adaptive drawing helpers
# (scale annotation sizes to image resolution)
# ─────────────────────────────────────────────

def _scale(img: np.ndarray, base: float) -> int:
    """Scale a base pixel value relative to image diagonal vs 2048px baseline."""
    H, W = img.shape[:2]
    diag = (H ** 2 + W ** 2) ** 0.5
    return max(1, int(round(base * diag / 2896)))  # 2896 ≈ diag of 2048x2048


def draw_crosshair(img, cx, cy, color, base_radius=28):
    r = _scale(img, base_radius)
    t = max(2, _scale(img, 3))
    x, y = int(round(cx)), int(round(cy))
    cv2.circle(img, (x, y), r, color, t, cv2.LINE_AA)
    cv2.line(img, (x - r, y), (x + r, y), color, t, cv2.LINE_AA)
    cv2.line(img, (x, y - r), (x, y + r), color, t, cv2.LINE_AA)
    # Small filled centre dot
    cv2.circle(img, (x, y), max(2, t), color, -1, cv2.LINE_AA)


def draw_label_badge(img, text, cx, cy, color, base_font=0.9):
    H, W = img.shape[:2]
    diag = (H ** 2 + W ** 2) ** 0.5
    fs   = base_font * diag / 2896
    ft   = max(1, int(fs * 2))
    r    = _scale(img, 28)

    (tw, th), bl = cv2.getTextSize(text, FONT, fs, ft)
    pad  = max(4, _scale(img, 6))
    x, y = int(round(cx)), int(round(cy))

    # Position badge above crosshair; flip below if too close to top
    by1 = y - r - pad * 2 - th
    by0 = y - r - pad * 3 - th - 4
    if by0 < 0:
        by0 = y + r + 4
        by1 = by0 + th + pad * 2

    bx0 = max(0, x - tw // 2 - pad)
    bx1 = min(W - 1, x + tw // 2 + pad)
    by0 = max(0, by0)
    by1 = min(H - 1, by1)

    cv2.rectangle(img, (bx0, by0), (bx1, by1), color, -1)
    cv2.putText(img, text, (bx0 + pad, by1 - bl // 2 - 1),
                FONT, fs, (15, 15, 15), ft, cv2.LINE_AA)


def draw_coord_text(img, cx, cy, color, base_font=0.65):
    H, W = img.shape[:2]
    diag = (H ** 2 + W ** 2) ** 0.5
    fs   = base_font * diag / 2896
    ft   = max(1, int(fs * 1.8))
    r    = _scale(img, 28)
    x, y = int(round(cx)), int(round(cy))
    txt  = f"({cx:.1f}, {cy:.1f})"
    tx   = min(x + r + 6, W - 200)
    ty   = min(y + 10,    H - 10)
    cv2.putText(img, txt, (tx, ty), FONT, fs, color, ft, cv2.LINE_AA)


def draw_confidence_bar(img, cx, cy, conf, color, base_font=0.60):
    """Small confidence bar drawn below the coord text."""
    H, W = img.shape[:2]
    diag = (H ** 2 + W ** 2) ** 0.5
    fs   = base_font * diag / 2896
    ft   = max(1, int(fs * 1.5))
    r    = _scale(img, 28)
    x, y = int(round(cx)), int(round(cy))

    bar_w = _scale(img, 80)
    bar_h = max(4, _scale(img, 8))
    bx    = min(x + r + 6, W - bar_w - 4)
    by    = min(y + 10 + _scale(img, 24), H - bar_h - 4)

    # Background track
    cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h),
                  (60, 60, 60), -1)
    # Filled portion
    filled = int(bar_w * conf)
    # Colour shifts green→orange→red with confidence (inverted: high conf = green)
    r_val  = int(255 * (1 - conf))
    g_val  = int(255 * conf)
    bar_color = (0, g_val, r_val)
    cv2.rectangle(img, (bx, by), (bx + filled, by + bar_h), bar_color, -1)
    cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h),
                  (120, 120, 120), 1)

    conf_txt = f"{conf:.2f}"
    cv2.putText(img, conf_txt, (bx + bar_w + 6, by + bar_h),
                FONT, fs, color, ft, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Annotation entry point
# ─────────────────────────────────────────────

def annotate_full(img: np.ndarray, markers: list[dict]) -> np.ndarray:
    vis = img.copy()
    for m in markers:
        x, y   = m["x"], m["y"]
        shape  = m.get("shape", "Unknown")
        conf   = m.get("confidence", None)
        gcp_id = m.get("gcp_id", "")
        color  = SHAPE_COLORS.get(shape, SHAPE_COLORS["Unknown"])

        label = f"{gcp_id} · {shape}" if gcp_id else shape

        draw_crosshair(vis, x, y, color)
        draw_label_badge(vis, label, x, y, color)
        draw_coord_text(vis, x, y, color)
        if conf is not None:
            draw_confidence_bar(vis, x, y, conf, color)

    return vis


def make_crop(img: np.ndarray, cx: float, cy: float,
              half: int) -> tuple[np.ndarray, float, float]:
    """
    Extract a (2*half x 2*half) crop centred on (cx, cy).
    The window is shifted inward if the marker is near an edge —
    no black padding, the marker stays visible.
    Returns (crop, local_cx, local_cy).
    """
    H, W = img.shape[:2]
    size = half * 2

    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half

    # Clamp window so it never goes out of bounds
    x0 = max(0, min(x0, W - size))
    y0 = max(0, min(y0, H - size))
    x1 = x0 + size
    y1 = y0 + size

    crop  = img[y0:y1, x0:x1].copy()
    lx    = cx - x0
    ly    = cy - y0
    return crop, lx, ly


# ─────────────────────────────────────────────
# Label / prediction parsing
# ─────────────────────────────────────────────

def parse_predictions(raw: dict) -> dict[str, list[dict]]:
    """
    Normalise predictions.json into:
        { "relative/path.JPG": [ {x, y, shape, confidence?, gcp_id?}, ... ] }

    Handles both single-entry and list-of-entries per key.
    """
    from collections import defaultdict
    grouped = defaultdict(list)

    for key, val in raw.items():
        entries = val if isinstance(val, list) else [val]
        for entry in entries:
            if "mark" in entry:
                x = entry["mark"]["x"]
                y = entry["mark"]["y"]
            elif "x" in entry and "y" in entry:
                x, y = entry["x"], entry["y"]
            else:
                print(f"  ⚠  Unrecognised entry for '{key}': {entry}")
                continue

            shape = entry.get("verified_shape",
                    entry.get("predicted_shape",
                    entry.get("shape", "Unknown")))
            conf  = entry.get("confidence", None)

            parts  = Path(key).parts
            gcp_id = parts[-2] if len(parts) >= 2 else None

            m = {"x": x, "y": y, "shape": shape}
            if conf  is not None: m["confidence"] = conf
            if gcp_id:            m["gcp_id"]     = gcp_id

            grouped[key].append(m)

    return dict(grouped)


# ─────────────────────────────────────────────
# Per-image worker
# ─────────────────────────────────────────────

def process_image(rel_path: str, markers: list[dict],
                  dataset_dir: Path, output_dir: Path,
                  crop_half: int, downscale: float) -> dict:
    img_path = dataset_dir / rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot decode: {img_path}")

    H, W = img.shape[:2]

    # ── Full annotated image (downscaled for disk) ──
    full_ann = annotate_full(img, markers)
    if downscale < 1.0:
        nW = int(W * downscale)
        nH = int(H * downscale)
        full_ann = cv2.resize(full_ann, (nW, nH), interpolation=cv2.INTER_AREA)

    safe = rel_path.replace("/", "__").replace("\\", "__")
    stem = Path(safe).with_suffix("").name

    full_out = output_dir / "full" / f"{stem}.jpg"
    full_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(full_out), full_ann, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ── Crops (one per marker, at native resolution) ──
    crop_paths = []
    for i, m in enumerate(markers):
        crop, lx, ly = make_crop(img, m["x"], m["y"], crop_half)

        # Annotate crop with a smaller crosshair
        color = SHAPE_COLORS.get(m.get("shape", "Unknown"), SHAPE_COLORS["Unknown"])
        r = max(10, crop_half // 8)
        t = max(1, r // 6)
        xi, yi = int(round(lx)), int(round(ly))
        cv2.circle(crop, (xi, yi), r, color, t, cv2.LINE_AA)
        cv2.line(crop, (xi - r, yi), (xi + r, yi), color, t, cv2.LINE_AA)
        cv2.line(crop, (xi, yi - r), (xi, yi + r), color, t, cv2.LINE_AA)
        cv2.circle(crop, (xi, yi), max(2, t), color, -1, cv2.LINE_AA)

        # Confidence + shape label top-left of crop
        shape = m.get("shape", "?")
        conf  = m.get("confidence")
        label = f"{shape}  {conf:.3f}" if conf is not None else shape
        fs    = max(0.4, crop_half / 300)
        ft    = max(1, int(fs * 2))
        cv2.rectangle(crop, (0, 0), (int(len(label) * 10 * fs) + 8, int(24 * fs) + 6),
                      (20, 20, 20), -1)
        cv2.putText(crop, label, (4, int(20 * fs) + 2),
                    FONT, fs, color, ft, cv2.LINE_AA)

        # Coloured border
        cv2.rectangle(crop, (0, 0),
                      (crop.shape[1] - 1, crop.shape[0] - 1), color, t + 1)

        suffix = f"_m{i}" if len(markers) > 1 else ""
        crop_out = output_dir / "crops" / f"{stem}{suffix}.jpg"
        crop_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        crop_paths.append(crop_out)

    return {"full": full_out, "crops": crop_paths}


# ─────────────────────────────────────────────
# Summary grid
# ─────────────────────────────────────────────

def build_summary_grid(crop_paths: list[Path], output_dir: Path,
                       thumb: int = 200, cols: int = 8) -> None:
    thumbs = []
    for p in sorted(crop_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w   = img.shape[:2]
        scale  = thumb / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        canvas = np.zeros((thumb, thumb, 3), dtype=np.uint8)
        yo = (thumb - nh) // 2
        xo = (thumb - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_AREA)
        thumbs.append(canvas)

    if not thumbs:
        return

    blank = np.zeros((thumb, thumb, 3), dtype=np.uint8)
    while len(thumbs) % cols:
        thumbs.append(blank)

    rows = [np.hstack(thumbs[i * cols:(i + 1) * cols])
            for i in range(len(thumbs) // cols)]
    grid = np.vstack(rows)
    out  = output_dir / "_summary_grid_crops.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"📊  Crop grid  →  {out}  ({cols} × {len(rows)} tiles)")


# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────

def print_stats(grouped: dict) -> None:
    all_m   = [m for ms in grouped.values() for m in ms]
    shapes  = Counter(m.get("shape", "Unknown") for m in all_m)
    confs   = [m["confidence"] for m in all_m if "confidence" in m]
    mpi     = Counter(len(v) for v in grouped.values())

    print("\n── Prediction Statistics ────────────────────────")
    print(f"  Unique images       : {len(grouped)}")
    print(f"  Total GCP markers   : {len(all_m)}")
    print(f"  Markers/image dist  : {dict(sorted(mpi.items()))}")
    if confs:
        print(f"  Confidence  min/avg/max : "
              f"{min(confs):.3f} / {sum(confs)/len(confs):.3f} / {max(confs):.3f}")
    print()
    mx = max(shapes.values()) if shapes else 1
    for shape, count in sorted(shapes.items()):
        bar = "█" * max(1, count * 30 // mx)
        print(f"  {shape:<12} {count:>5}  {bar}")
    print("─────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise predictions.json on test images.")
    parser.add_argument("--predictions", default="./predictions.json",
                        help="Path to predictions.json (default: ./predictions.json)")
    parser.add_argument("--dataset_dir", default="/home/numair/Desktop/IISc/misc/skylark_assn/GCP_Assignment_Datasets/test_dataset",
                        help="Root of test_dataset (default: /home/numair/Desktop/IISc/misc/skylark_assn/GCP_Assignment_Datasets/test_dataset")
    parser.add_argument("--output_dir",  default="./viz_test_predictions",
                        help="Output directory (default: ./viz_test_predictions)")
    parser.add_argument("--crop_size",   type=int, default=512,
                        help="Full side length of saved crop in native pixels (default: 512)")
    parser.add_argument("--downscale",   type=float, default=1.0,
                        help="Scale factor for saved full images, e.g. 0.25 = quarter res (default: 0.25)")
    parser.add_argument("--workers",     type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--grid_cols",   type=int, default=8,
                        help="Columns in summary grid (default: 8)")
    parser.add_argument("--no_grid",     action="store_true",
                        help="Skip building the summary grid")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir  = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂  Test dataset : {dataset_dir}")
    print(f"🏷   Predictions  : {args.predictions}")
    print(f"💾  Output       : {output_dir}")

    with open(args.predictions) as f:
        raw = json.load(f)

    grouped = parse_predictions(raw)
    print_stats(grouped)

    crop_half   = args.crop_size // 2
    errors      = []
    all_crops   = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_image, rel, markers, dataset_dir,
                        output_dir, crop_half, args.downscale): rel
            for rel, markers in grouped.items()
        }
        with tqdm(total=len(futures), desc="Annotating", unit="img") as pbar:
            for fut in as_completed(futures):
                rel = futures[fut]
                try:
                    result = fut.result()
                    all_crops.extend(result["crops"])
                except Exception as e:
                    errors.append((rel, str(e)))
                finally:
                    pbar.update(1)

    if errors:
        print(f"\n⚠️   {len(errors)} image(s) failed:")
        for rel, msg in errors[:20]:
            print(f"   {rel}: {msg}")
        if len(errors) > 20:
            print(f"   … and {len(errors) - 20} more")

    if not args.no_grid and all_crops:
        build_summary_grid(all_crops, output_dir, cols=args.grid_cols)

    ok = len(grouped) - len(errors)
    print(f"\n✅  {ok}/{len(grouped)} images annotated → {output_dir}")
    print(f"   full/   ← downscaled full images  ({int(args.downscale*100)}% of native res)")
    print(f"   crops/  ← {args.crop_size}×{args.crop_size}px native-res crops around each marker\n")


if __name__ == "__main__":
    main()