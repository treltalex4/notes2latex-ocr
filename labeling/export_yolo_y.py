"""Label page images using a trained YOLO model to train YOLO model(Active Learning).

Output structure:
    <output>/
        images/          ← copies of original page images
        labels/          ← one .txt per image (YOLO format)
        labels.txt       ← class names (one class: "line")
        debug/           ← overlay images (optional, --debug)

Usage:
    python -m labeling.export_yolo_y --input my_dataset/pages_to_label --output yolo_labels
    python -m labeling.export_yolo_y --input my_dataset/pages_to_label --output yolo_labels --model yolo26s.pt --debug
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "yolo_models"
DEFAULT_MODEL = "yolo26s.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0
CLASS_NAME = "line"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _save_debug_overlay(
    img_bgr: np.ndarray,
    bboxes_px: list[tuple[int, int, int, int]],
    out_path: Path,
) -> None:
    canvas = img_bgr.copy()
    for idx, (x1, y1, x2, y2) in enumerate(bboxes_px, 1):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(canvas, str(idx), (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pseudo-label pages using a trained YOLO model → YOLO format."
    )
    p.add_argument("--input", required=True, help="Folder with page images.")
    p.add_argument("--output", required=True, help="Output folder.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model filename in {MODELS_DIR.name}/ (default: {DEFAULT_MODEL}).")
    p.add_argument("--conf", type=float, default=0.30,
                   help="Confidence threshold.")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--debug", action="store_true",
                   help="Save overlay images to <output>/debug/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    weights_path = MODELS_DIR / args.model
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model not found: {weights_path}\n"
            f"Place a trained best.pt at this path (e.g. copy from "
            f"yolo_training/runs/<run>/weights/best.pt and rename to {args.model})."
        )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    images = list(_iter_images(input_path))
    if not images:
        raise ValueError(f"No images found in: {input_path}")

    out = Path(args.output)
    images_dir = out / "images"
    labels_dir = out / "labels"
    debug_dir = out / "debug" if args.debug else None
    for d in [images_dir, labels_dir]:
        _ensure_dir(d)
    if debug_dir:
        _ensure_dir(debug_dir)

    (out / "labels.txt").write_text(CLASS_NAME + "\n", encoding="utf-8")

    print(f"Model   : {weights_path}")
    print(f"Input   : {input_path} ({len(images)} images)")
    print(f"Output  : {out.resolve()}")
    print()

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    total_lines = 0
    for img_path in images:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        r = results[0]
        h_page, w_page = r.orig_shape  # (h, w)

        # YOLO returns xywhn (normalised cx, cy, w, h) directly — perfect for export.
        yolo_lines: list[str] = []
        bboxes_px: list[tuple[int, int, int, int]] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xywhn = r.boxes.xywhn.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (cx, cy, w, h), (x1, y1, x2, y2) in zip(xywhn, xyxy):
                cx = float(np.clip(cx, 0.0, 1.0))
                cy = float(np.clip(cy, 0.0, 1.0))
                w = float(np.clip(w, 0.0, 1.0))
                h = float(np.clip(h, 0.0, 1.0))
                yolo_lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                bboxes_px.append((int(x1), int(y1), int(x2), int(y2)))

        label_path = labels_dir / (img_path.stem + ".txt")
        label_path.write_text(
            ("\n".join(yolo_lines) + "\n") if yolo_lines else "",
            encoding="utf-8",
        )

        shutil.copy2(img_path, images_dir / img_path.name)

        if debug_dir is not None:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                _save_debug_overlay(img_bgr, bboxes_px,
                                    debug_dir / (img_path.stem + "_overlay.png"))

        total_lines += len(yolo_lines)
        print(f"  {img_path.name}: {len(yolo_lines)} lines")

    print(f"\nDone. {len(images)} pages → {total_lines} line boxes total.")


if __name__ == "__main__":
    main()
