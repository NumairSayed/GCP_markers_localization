"""One-off helper: validate that every annotated image exists and that every
record is parseable. Writes a cleaned JSON with only usable entries and prints
a summary. Run this before training if the source JSON is suspicious.

Usage:
    python -m scripts.prepare_data \
        --annotations data/train_dataset/curated_gcp_marks.json \
        --data-root data/train_dataset \
        --out data/train_dataset/curated_gcp_marks.clean.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataset import load_records  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = load_records(args.annotations, args.data_root)
    cleaned = {
        r.rel_path: {
            "mark": {"x": r.x, "y": r.y},
            "verified_shape": r.shape,
        }
        for r in records
    }
    with open(args.out, "w") as f:
        json.dump(cleaned, f, indent=2)
    logging.info("Wrote %d clean records → %s", len(cleaned), args.out)


if __name__ == "__main__":
    main()
