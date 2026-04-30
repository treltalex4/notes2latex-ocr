"""Find duplicate page photos via hashing.

Usage:
    python -m labeling.find_duplicates --input my_dataset/pages_to_label
    python -m labeling.find_duplicates --input yolo_training/dataset/images --threshold 8

Output: prints clusters of likely-same pages so you can decide what to keep.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find duplicate page photos via pHash.")
    p.add_argument("--input", required=True, help="Folder with images.")
    p.add_argument("--threshold", type=int, default=6,
                   help="Max Hamming distance to consider duplicates "
                        "(0 = identical, 6 = near-duplicate, 12+ = different). "
                        "Default 6 catches different photos of the same page.")
    p.add_argument("--hash-size", type=int, default=16,
                   help="pHash resolution (16 = 256 bits, more precise; "
                        "8 = 64 bits, faster but coarser). Default 16.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("Missing dependencies. Install:  pip install imagehash pillow",
              file=sys.stderr)
        sys.exit(1)

    folder = Path(args.input)
    if not folder.exists():
        print(f"Not found: {folder}", file=sys.stderr)
        sys.exit(1)

    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Computing pHash for {len(images)} images (hash_size={args.hash_size})...")
    hashes: list[tuple[Path, "imagehash.ImageHash"]] = []
    for path in images:
        try:
            h = imagehash.phash(Image.open(path), hash_size=args.hash_size)
            hashes.append((path, h))
        except Exception as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)

    # ── Cluster by Hamming distance via union-find ────────────────────────────
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i][1] - hashes[j][1] <= args.threshold:
                union(i, j)

    clusters: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for i, (path, h) in enumerate(hashes):
        root = find(i)
        clusters[root].append((path, i))

    # ── Report ────────────────────────────────────────────────────────────────
    duplicate_groups = [c for c in clusters.values() if len(c) > 1]

    if not duplicate_groups:
        print(f"\nNo duplicates found (threshold={args.threshold}).")
        return

    print(f"\nFound {len(duplicate_groups)} duplicate group(s):\n")
    leakage_groups: list[list] = []

    for n, group in enumerate(duplicate_groups, 1):
        # Detect cross-folder matches: split = first folder under root
        splits = {p.relative_to(folder).parts[0] if len(p.relative_to(folder).parts) > 1
                  else "(root)" for p, _ in group}
        is_cross = len(splits) > 1
        is_leakage = "train" in splits and "val" in splits

        if is_leakage:
            marker = "  ⚠ DATA LEAKAGE (train ↔ val)"
        elif is_cross:
            marker = f"  ⚠ CROSS-FOLDER ({' ↔ '.join(sorted(splits))})"
        else:
            marker = ""
        print(f"── Group {n} ({len(group)} images){marker} ──")

        first_h = hashes[group[0][1]][1]
        for path, idx in group:
            dist = hashes[idx][1] - first_h
            print(f"  [dist={dist:2d}]  {path.relative_to(folder)}")
        print()

        if is_leakage:
            leakage_groups.append(group)

        if is_cross and not is_leakage:
            pass  # cross-folder match (e.g. new ↔ train) — informational

    total_dup = sum(len(g) - 1 for g in duplicate_groups)
    print(f"Total: {total_dup} potential duplicates that could be deduplicated "
          f"(keeping 1 per group).")

    if leakage_groups:
        print(f"\n⚠ WARNING: {len(leakage_groups)} group(s) span train AND val.")
        print(f"  This causes data leakage — move all duplicates into the same split.")


if __name__ == "__main__":
    main()
