"""
Download and prepare im2latex-100k dataset from Kaggle.

Usage:
    python download_im2latex.py [--data-dir data_raw]

Requires:
    pip install kaggle
    ~/.kaggle/kaggle.json  (from https://www.kaggle.com/settings → API → Create New Token)

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
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


KAGGLE_DATASET = "shahrukhkhan/im2latex-100k"

# Expected files after download
REQUIRED_LST = [
    "im2latex_formulas.lst",
    "im2latex_train.lst",
    "im2latex_validate.lst",
    "im2latex_test.lst",
]


def _check_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle not installed. Run: pip install kaggle")
        sys.exit(1)

    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        print(
            f"ERROR: Kaggle credentials not found at {cred_path}\n"
            "  1. Go to https://www.kaggle.com/settings\n"
            "  2. API → Create New Token\n"
            "  3. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "  4. chmod 600 ~/.kaggle/kaggle.json"
        )
        sys.exit(1)


def _run(cmd: str) -> None:
    print(f"  $ {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"ERROR: command failed (exit {ret.returncode})")
        sys.exit(1)


def _extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract tar.gz or zip into dest/."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        print(f"  Extracting {archive_path.name} ...")
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest)
    elif name.endswith(".tar"):
        print(f"  Extracting {archive_path.name} ...")
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(dest)
    elif name.endswith(".zip"):
        print(f"  Extracting {archive_path.name} ...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest)
    else:
        print(f"  Unknown archive format: {archive_path.name} — skipping extraction")


def _find_files(root: Path, extensions: tuple) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in extensions]


def _find_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def _collect_images(src_dir: Path, dest_dir: Path) -> int:
    """Move all .png files from src_dir (recursively) into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for png in src_dir.rglob("*.png"):
        target = dest_dir / png.name
        if not target.exists():
            shutil.move(str(png), str(target))
            moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download im2latex-100k from Kaggle")
    parser.add_argument("--data-dir", default="data_raw", help="Target directory")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already present"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    formula_images_dir = data_dir / "formula_images"

    # ── Check if already done ──────────────────────────────────────────────────
    if not args.force:
        lst_ok = all((data_dir / f).exists() for f in REQUIRED_LST)
        imgs_ok = formula_images_dir.exists() and any(formula_images_dir.iterdir())
        if lst_ok and imgs_ok:
            n = sum(1 for _ in formula_images_dir.glob("*.png"))
            print(f"Dataset already present: {data_dir} ({n} images). Use --force to re-download.")
            return

    _check_kaggle()

    # ── Download ───────────────────────────────────────────────────────────────
    tmp_dir = data_dir / "_download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {KAGGLE_DATASET} → {tmp_dir} ...")
    _run(f"kaggle datasets download -d {KAGGLE_DATASET} -p {tmp_dir}")

    # ── Extract top-level zip/tar ──────────────────────────────────────────────
    archives = list(tmp_dir.glob("*.zip")) + list(tmp_dir.glob("*.tar.gz")) + list(tmp_dir.glob("*.tar"))
    if not archives:
        print("ERROR: no archives found in download dir. Check Kaggle download output.")
        sys.exit(1)

    extract_dir = tmp_dir / "extracted"
    for arch in archives:
        _extract_archive(arch, extract_dir)

    # ── Extract nested archives (some versions have formula_images.tar.gz) ─────
    for nested in extract_dir.rglob("*.tar.gz"):
        print(f"  Found nested archive: {nested.name}")
        _extract_archive(nested, nested.parent)
        nested.unlink()

    for nested in extract_dir.rglob("*.tar"):
        print(f"  Found nested archive: {nested.name}")
        _extract_archive(nested, nested.parent)
        nested.unlink()

    # ── Copy .lst files ────────────────────────────────────────────────────────
    print("\nCopying .lst files ...")
    data_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for lst_name in REQUIRED_LST:
        src = _find_file(extract_dir, lst_name)
        if src is None:
            missing.append(lst_name)
            continue
        dst = data_dir / lst_name
        shutil.copy2(str(src), str(dst))
        print(f"  {lst_name} → {dst}")

    if missing:
        print(f"\nWARN: could not find: {missing}")
        print("  Check extract_dir manually:", extract_dir)

    # ── Collect all PNGs → formula_images/ ────────────────────────────────────
    print("\nCollecting formula images ...")

    # First try: there's already a formula_images/ directory somewhere
    img_src = None
    for candidate in extract_dir.rglob("formula_images"):
        if candidate.is_dir():
            img_src = candidate
            break

    if img_src:
        print(f"  Found formula_images at {img_src}")
        if img_src != formula_images_dir:
            if formula_images_dir.exists():
                shutil.rmtree(formula_images_dir)
            shutil.move(str(img_src), str(formula_images_dir))
    else:
        # Fallback: collect all .png files from anywhere in extract_dir
        print("  No formula_images/ dir found — collecting all .png recursively ...")
        n = _collect_images(extract_dir, formula_images_dir)
        print(f"  Collected {n} images")

    # ── Verify ─────────────────────────────────────────────────────────────────
    print("\nVerifying ...")
    ok = True
    for lst_name in REQUIRED_LST:
        path = data_dir / lst_name
        if path.exists():
            lines = path.read_text(encoding="latin-1").strip().splitlines()
            print(f"  ✓ {lst_name}: {len(lines)} lines")
        else:
            print(f"  ✗ MISSING: {lst_name}")
            ok = False

    if formula_images_dir.exists():
        n_imgs = sum(1 for _ in formula_images_dir.glob("*.png"))
        print(f"  ✓ formula_images/: {n_imgs} PNG files")
        if n_imgs < 50_000:
            print(f"  WARN: expected ~100k images, got {n_imgs} — download may be incomplete")
    else:
        print("  ✗ MISSING: formula_images/")
        ok = False

    # ── Cleanup ────────────────────────────────────────────────────────────────
    print(f"\nCleaning up {tmp_dir} ...")
    shutil.rmtree(tmp_dir)

    if ok:
        print(f"\nDone. Dataset ready at: {data_dir.resolve()}")
        print("Next step: python prepare_data.py --profile rtx5090_32gb --datasets im2latex")
    else:
        print("\nWARN: some files missing — check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
