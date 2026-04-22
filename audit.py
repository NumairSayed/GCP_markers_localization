"""Quick audit of the train_dataset directory.

Run from inside the train_dataset directory (the one that contains
curated_gcp_marks.json):

    python check_data.py

Or pass paths explicitly:

    python check_data.py --root . --ann curated_gcp_marks.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="train_dataset root (default: cwd)")
    p.add_argument("--ann", default="curated_gcp_marks.json",
                   help="annotation JSON filename (relative to --root)")
    args = p.parse_args()

    root = Path(args.root).resolve()
    ann_path = root / args.ann

    # 1. Count images on disk (recursive)
    on_disk = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print(f"Root: {root}")
    print(f"Images on disk: {len(on_disk)}")

    # Depth breakdown (project/survey/gcp/image)
    depth_counts = Counter()
    for p in on_disk:
        rel_parts = p.relative_to(root).parts
        depth_counts[len(rel_parts)] += 1
    print(f"Depth histogram (parts incl. filename): {dict(depth_counts)}")

    # Top-level projects
    projects = sorted({p.relative_to(root).parts[0] for p in on_disk
                       if len(p.relative_to(root).parts) > 1})
    print(f"Top-level folders ({len(projects)}): {projects[:10]}"
          f"{' ...' if len(projects) > 10 else ''}")

    # 2. Load annotations
    if not ann_path.exists():
        print(f"\n⚠ annotation file not found: {ann_path}")
        return
    with open(ann_path, "r") as f:
        ann = json.load(f)
    print(f"\nAnnotation entries: {len(ann)}")

    # 3. Match annotations to files on disk
    on_disk_rel = {str(p.relative_to(root)).replace("\\", "/") for p in on_disk}

    matched = 0
    missing_direct = []
    missing_even_after_strip = []

    for rel in ann.keys():
        rel_norm = rel.replace("\\", "/")
        if rel_norm in on_disk_rel:
            matched += 1
            continue

        # Try stripping 1 or 2 leading path components
        parts = Path(rel_norm).parts
        found = False
        for strip in (1, 2):
            if len(parts) > strip:
                alt = "/".join(parts[strip:])
                if alt in on_disk_rel:
                    matched += 1
                    found = True
                    break
        if not found:
            missing_direct.append(rel_norm)
            missing_even_after_strip.append(rel_norm)

    print(f"\nMatched annotations → files on disk: {matched} / {len(ann)}")
    print(f"Missing (no match even after stripping prefixes): {len(missing_even_after_strip)}")

    if missing_even_after_strip:
        print("\nFirst 10 missing annotation paths:")
        for r in missing_even_after_strip[:10]:
            print(f"  {r}")

    # 4. Files on disk that have no annotation
    ann_basenames = {Path(k).name for k in ann.keys()}
    orphan_files = [p for p in on_disk if p.name not in ann_basenames]
    print(f"\nImage files on disk with no annotation match (by filename): "
          f"{len(orphan_files)}")
    for p in orphan_files[:10]:
        print(f"  {p.relative_to(root)}")


if __name__ == "__main__":
    main()

