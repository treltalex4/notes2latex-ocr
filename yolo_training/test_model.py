"""Run inference on all test pages and save annotated results.

All parameters are in config_y.py — edit that file to change anything.
Results are saved to yolo_training/test_results/<model_stem>/.

Usage:
    python -m yolo_training.test_model
    python -m yolo_training.test_model --weights path/to/custom.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
RUNS_DIR = THIS_DIR / "runs"
TEST_PAGES_DIR = THIS_DIR.parent / "my_dataset" / "test_pages"
RESULTS_DIR = THIS_DIR / "test_results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_best_pt(model_stem: str) -> Path:
    candidates = sorted(
        RUNS_DIR.glob(f"{model_stem}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trained best.pt found under {RUNS_DIR}/{model_stem}*/weights/.\n"
            f"Train first:  python -m yolo_training.train_yolo"
        )
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test YOLO line detector on all test pages.")
    p.add_argument("--weights", default=None,
                   help="Explicit path to .pt file (overrides auto-detection).")
    return p.parse_args()


def main() -> None:
    from yolo_training.config_y import cfg

    args = parse_args()
    model_stem = Path(cfg.model).stem

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"Error: weights not found: {weights_path}", file=sys.stderr)
            sys.exit(1)
    else:
        weights_path = find_best_pt(model_stem)

    images = [p for p in TEST_PAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"No images found in {TEST_PAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Weights : {weights_path}")
    print(f"Images  : {len(images)} pages in {TEST_PAGES_DIR}")
    print(f"Results : {RESULTS_DIR / model_stem}")
    print()

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    for img_path in sorted(images):
        model.predict(
            source=str(img_path),
            conf=cfg.test_conf,
            imgsz=cfg.test_imgsz,
            device=cfg.device,
            save=True,
            save_txt=True,
            project=str(RESULTS_DIR),
            name=model_stem,
            exist_ok=True,
            verbose=False,
        )
        print(f"  {img_path.name}")

    print(f"\nDone. Annotated images saved to: {RESULTS_DIR / model_stem}")


if __name__ == "__main__":
    main()
