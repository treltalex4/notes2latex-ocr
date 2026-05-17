"""
Download and prepare im2latex-100k dataset from Zenodo (record 56198).

Usage:
    python download_im2latex.py [--data-dir data_raw]

No authentication required.

Expected output structure (matches prepare_data.py expectations):
    data_raw/
        im2latex_formulas.lst
        im2latex_train.lst
        im2latex_validate.lst
        im2latex_test.lst
        formula_images/
            *.png
"""

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path


ZENODO_BASE = "https://zenodo.org/api/records/56198/files"

LST_FILES = [
    "im2latex_formulas.lst",
    "im2latex_train.lst",
    "im2latex_validate.lst",
    "im2latex_test.lst",
]

IMAGES_ARCHIVE = "formula_images.tar.gz"


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        def _progress(count, block_size, total):
            if total > 0:
                pct = min(100, count * block_size * 100 // total)
                mb = count * block_size / 1_048_576
                total_mb = total / 1_048_576
                print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()
        tmp.rename(dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"\n  ERROR: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download im2latex-100k from Zenodo")
    parser.add_argument("--data-dir", default="data_raw")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    formula_images_dir = data_dir / "formula_images"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if already done ──────────────────────────────────────────────────
    if not args.force:
        lst_ok = all((data_dir / f).exists() for f in LST_FILES)
        imgs_ok = formula_images_dir.exists() and any(formula_images_dir.iterdir())
        if lst_ok and imgs_ok:
            n = sum(1 for _ in formula_images_dir.glob("*.png"))
            print(f"Dataset already present ({n} images). Use --force to re-download.")
            return

    # ── Download .lst files ────────────────────────────────────────────────────
    print("\n=== Downloading .lst files ===")
    for fname in LST_FILES:
        dest = data_dir / fname
        if dest.exists() and not args.force:
            print(f"  {fname} already exists, skipping")
            continue
        _download(f"{ZENODO_BASE}/{fname}/content", dest)

    # ── Download and extract formula_images.tar.gz ─────────────────────────────
    archive_path = data_dir / IMAGES_ARCHIVE

    if not archive_path.exists() or args.force:
        print("\n=== Downloading formula_images.tar.gz (~292 MB) ===")
        _download(f"{ZENODO_BASE}/{IMAGES_ARCHIVE}/content", archive_path)
    else:
        print(f"\n  {IMAGES_ARCHIVE} already downloaded, skipping")

    print("\n=== Extracting formula_images.tar.gz ===")
    print("  This may take a few minutes ...")
    if formula_images_dir.exists() and args.force:
        shutil.rmtree(formula_images_dir)

    formula_images_dir.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            if member.isfile() and member.name.endswith(".png"):
                member.name = Path(member.name).name  # strip any path prefix
                tf.extract(member, formula_images_dir)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{total} files ...", flush=True)

    archive_path.unlink()  # free space

    # ── Verify ─────────────────────────────────────────────────────────────────
    print("\n=== Verifying ===")
    ok = True
    for fname in LST_FILES:
        path = data_dir / fname
        if path.exists():
            lines = path.read_text(encoding="latin-1").strip().splitlines()
            print(f"  ✓ {fname}: {len(lines)} lines")
        else:
            print(f"  ✗ MISSING: {fname}")
            ok = False

    if formula_images_dir.exists():
        n = sum(1 for _ in formula_images_dir.glob("*.png"))
        print(f"  ✓ formula_images/: {n} PNG files")
        if n < 50_000:
            print(f"  WARN: expected ~100k images, got {n}")
    else:
        print("  ✗ MISSING: formula_images/")
        ok = False

    if ok:
        print(f"\nDone. Dataset ready at: {data_dir.resolve()}")
        print("Next: python prepare_data.py --profile rtx5090_32gb --datasets im2latex")
    else:
        print("\nWARN: some files missing")
        sys.exit(1)


if __name__ == "__main__":
    main()
