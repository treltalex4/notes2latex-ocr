"""YOLO-based line slicer for handwritten page images.

Output structure:
    <output>/
        crops/                ← <stem>_line_NN.png
        debug/                ← overlay images (if cfg.save_debug)
        slices_meta.jsonl     ← per-line metadata (if cfg.save_metadata)

Usage:
    python -m labeling.slicer --input my_dataset/pages_to_label --output my_dataset/line_crops
    python -m labeling.slicer --input my_dataset/pages_to_label --output my_dataset/line_crops --model yolo26s.pt
    python -m labeling.slicer --input page.jpg --output my_dataset/line_crops --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from labeling.config_s import SlicerConfig

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "yolo_models"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _load_image(path: Path, max_w: int | None = None) -> np.ndarray:
    """Load page image at full resolution. Optionally downscale ONLY if width
    exceeds max_w (use generously — YOLO does its own letterboxing internally,
    so downscaling pre-detection just throws away pixels for crop quality)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if max_w is not None and max_w > 0:
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _sort_reading_order(
    bboxes: list[tuple[int, int, int, int]],
    cfg: SlicerConfig,
) -> list[int]:
    """Sort bbox indices in reading order using horizontal-band grouping.

    Lines on (almost) the same vertical level are grouped into a band and
    sorted left-to-right inside it. Bands themselves are sorted top-to-bottom.
    """
    if not bboxes:
        return []

    heights = [y2 - y1 for (_, y1, _, y2) in bboxes]
    median_h = float(np.median(heights)) if heights else 1.0
    tol = cfg.band_tolerance * median_h

    centres = [((y1 + y2) / 2.0, i) for i, (_, y1, _, y2) in enumerate(bboxes)]
    centres.sort()

    bands: list[list[int]] = []
    band_centre: list[float] = []
    for cy, idx in centres:
        if bands and abs(cy - band_centre[-1]) <= tol:
            bands[-1].append(idx)
            # Update running mean of the band centre
            n = len(bands[-1])
            band_centre[-1] = band_centre[-1] + (cy - band_centre[-1]) / n
        else:
            bands.append([idx])
            band_centre.append(cy)

    order: list[int] = []
    for band in bands:
        band.sort(key=lambda i: bboxes[i][0])  # left-to-right by x1
        order.extend(band)
    return order


def _apply_contour_mask(
    crop: np.ndarray,
    crop_origin: tuple[int, int],
    own_bbox: tuple[int, int, int, int],
    other_bboxes: list[tuple[int, int, int, int]],
    cfg: SlicerConfig,
) -> np.ndarray:
    """Whiten pixels inside this crop that overlap with neighbouring line
    bboxes (so parts of adjacent lines that bled into this crop are removed).

    crop_origin = (x_off, y_off) — top-left of the crop in original image coords.
    """
    cx_off, cy_off = crop_origin
    h, w = crop.shape[:2]

    # Start with everything = "neighbour territory", carve out our own bbox.
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # Whiten parts that overlap neighbours BUT NOT our own bbox.
    own_x1, own_y1, own_x2, own_y2 = own_bbox
    for (ox1, oy1, ox2, oy2) in other_bboxes:
        # Skip if no overlap with our crop region
        if ox2 < cx_off or oy2 < cy_off or ox1 >= cx_off + w or oy1 >= cy_off + h:
            continue
        # Compute overlap rect inside crop coords
        rx1 = max(0, ox1 - cx_off)
        ry1 = max(0, oy1 - cy_off)
        rx2 = min(w, ox2 - cx_off)
        ry2 = min(h, oy2 - cy_off)
        # Subtract our own bbox from the area to be whitened
        # (so we don't whiten our own ink)
        # Convert own bbox into crop coords with dilation-friendly inflation
        dx = cfg.contour_mask_dilate_px
        ox_start = max(rx1, own_x1 - cx_off - dx)
        oy_start = max(ry1, own_y1 - cy_off - dx)
        ox_end   = min(rx2, own_x2 - cx_off + dx)
        oy_end   = min(ry2, own_y2 - cy_off + dx)

        # Whiten the entire overlap rect first
        if rx2 > rx1 and ry2 > ry1:
            mask[ry1:ry2, rx1:rx2] = 0
        # Then restore our own bbox region (with dilation tolerance)
        if ox_end > ox_start and oy_end > oy_start:
            mask[oy_start:oy_end, ox_start:ox_end] = 255

    if mask.min() == 255:
        return crop  # nothing to mask

    out = crop.copy()
    out[mask == 0] = 255  # whiten neighbour territory
    return out


def _save_debug_overlay(
    img_bgr: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    order: list[int],
    out_path: Path,
    cfg: SlicerConfig,
) -> None:
    canvas = img_bgr.copy()
    for rank, idx in enumerate(order, 1):
        x1, y1, x2, y2 = bboxes[idx]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), cfg.debug_box_color, 2)
        cv2.putText(canvas, str(rank), (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    cfg.debug_text_color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _slice_one_page(
    img_path: Path,
    yolo_model,
    crops_dir: Path,
    debug_dir: Path | None,
    meta_path: Path | None,
    cfg: SlicerConfig,
) -> tuple[int, list[tuple[int, int]]]:
    img_bgr = _load_image(img_path, cfg.max_width or None)
    h_page, w_page = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    results = yolo_model.predict(
        source=img_bgr,
        conf=cfg.conf,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        device=cfg.device,
        verbose=False,
    )
    r = results[0]

    bboxes: list[tuple[int, int, int, int]] = []
    confs: list[float] = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, cs):
            x1i = max(0, int(x1)); y1i = max(0, int(y1))
            x2i = min(w_page - 1, int(x2)); y2i = min(h_page - 1, int(y2))
            if x2i - x1i < 2 or y2i - y1i < cfg.min_line_height_px:
                continue
            aspect = (x2i - x1i) / max(1, (y2i - y1i))
            if aspect < cfg.min_crop_aspect_ratio:
                continue
            bboxes.append((x1i, y1i, x2i, y2i))
            confs.append(float(c))

    if not bboxes:
        logger.warning("%s: no lines detected", img_path.name)
        return 0, []

    order = _sort_reading_order(bboxes, cfg)

    if debug_dir is not None:
        _save_debug_overlay(img_bgr, bboxes, order,
                            debug_dir / f"{img_path.stem}_overlay.png", cfg)

    saved = 0
    crop_widths: list[tuple[int, int]] = []
    meta_lines: list[dict] = []
    for rank, idx in enumerate(order, 1):
        x1, y1, x2, y2 = bboxes[idx]
        # Apply padding
        px1 = max(0, x1 - cfg.pad_x)
        py1 = max(0, y1 - cfg.pad_y)
        px2 = min(w_page - 1, x2 + cfg.pad_x)
        py2 = min(h_page - 1, y2 + cfg.pad_y)

        crop = gray[py1:py2 + 1, px1:px2 + 1].copy()
        if crop.size == 0:
            continue

        if cfg.contour_mask:
            others = [bboxes[j] for j in range(len(bboxes)) if j != idx]
            crop = _apply_contour_mask(
                crop, (px1, py1), (x1, y1, x2, y2), others, cfg,
            )

        # Ink density check (after masking)
        dark = np.count_nonzero(crop < 180)
        density = dark / max(1, crop.size)
        if density < cfg.min_ink_density:
            continue

        ch, cw = crop.shape[:2]
        crop_name = f"{img_path.stem}_line_{rank:02d}.png"
        cv2.imwrite(str(crops_dir / crop_name), crop)
        saved += 1
        crop_widths.append((cw, ch))

        if cfg.save_metadata and meta_path is not None:
            meta_lines.append({
                "line_id":     f"{img_path.stem}_line_{rank:02d}",
                "file":        crop_name,
                "page":        img_path.name,
                "rank":        rank,
                "bbox_tight":  [x1, y1, x2, y2],
                "bbox_padded": [px1, py1, px2, py2],
                "confidence":  round(confs[idx], 4),
                "crop_size":   [cw, ch],
                "ink_density": round(density, 4),
                "page_size":   [w_page, h_page],
            })

    if meta_lines and meta_path is not None:
        with meta_path.open("a", encoding="utf-8") as f:
            for rec in meta_lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return saved, crop_widths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO-based slicer: cut handwritten pages into line crops."
    )
    p.add_argument("--input", required=True,
                   help="Input image file or folder.")
    p.add_argument("--output", required=True,
                   help="Output folder (will contain crops/, debug/, meta).")
    p.add_argument("--model", default=None,
                   help="Override config model. Filename inside "
                        f"{MODELS_DIR.name}/ (e.g. best.pt, yolo26s.pt).")
    p.add_argument("--conf", type=float, default=None,
                   help="Override confidence threshold.")
    p.add_argument("--iou", type=float, default=None,
                   help="Override NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Override inference image size.")
    p.add_argument("--device", default=None,
                   help="Override device ('0', 'cpu').")
    p.add_argument("--no-mask", action="store_true",
                   help="Disable contour masking.")
    p.add_argument("--no-debug", action="store_true",
                   help="Skip writing debug overlays.")
    p.add_argument("--no-meta", action="store_true",
                   help="Skip writing slices_meta.jsonl.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = SlicerConfig()
    if args.model:    cfg.model = args.model
    if args.conf:     cfg.conf = args.conf
    if args.iou:      cfg.iou = args.iou
    if args.imgsz:    cfg.imgsz = args.imgsz
    if args.device:   cfg.device = args.device
    if args.no_mask:  cfg.contour_mask = False
    if args.no_debug: cfg.save_debug = False
    if args.no_meta:  cfg.save_metadata = False

    weights_path = MODELS_DIR / cfg.model
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model not found: {weights_path}\n"
            f"Place a trained best.pt at this path (e.g. copy from "
            f"yolo_training/runs/<run>/weights/best.pt and rename to {cfg.model})."
        )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    images = list(_iter_images(input_path))
    if not images:
        raise ValueError(f"No images found in: {input_path}")

    out = Path(args.output)
    crops_dir = out / "crops"
    debug_dir = out / "debug" if cfg.save_debug else None
    meta_path = out / "slices_meta.jsonl" if cfg.save_metadata else None

    _ensure_dir(crops_dir)
    if debug_dir:
        _ensure_dir(debug_dir)
    # Truncate meta on a fresh run so we don't append to old data
    if meta_path and meta_path.exists():
        meta_path.unlink()

    print(f"Model   : {weights_path}")
    print(f"Conf    : {cfg.conf}    IoU: {cfg.iou}    imgsz: {cfg.imgsz}")
    print(f"Input   : {input_path} ({len(images)} images)")
    print(f"Output  : {out.resolve()}")
    print()

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    TARGET_H = 128   # recognition model's target_height (config.py)
    MAX_W    = 2048  # recognition model's max_width    (config.py)

    total = 0
    all_widths: list[tuple[int, int]] = []  # (crop_w, crop_h)
    for img_path in images:
        n, wh = _slice_one_page(img_path, model, crops_dir, debug_dir, meta_path, cfg)
        total += n
        all_widths.extend(wh)
        print(f"  {img_path.name}: {n} lines")

    print(f"\nDone. {len(images)} pages -> {total} line crops.")
    print(f"Crops:  {crops_dir.resolve()}")
    if debug_dir:
        print(f"Debug:  {debug_dir.resolve()}")
    if meta_path:
        print(f"Meta:   {meta_path.resolve()}")

    # Width statistics — how crops will look after resize to target_height
    if all_widths:
        eff_widths = sorted(cw * TARGET_H / max(1, ch) for cw, ch in all_widths)
        n = len(eff_widths)
        over = sum(1 for w in eff_widths if w > MAX_W)
        print(f"\n--- Crop width after resize to h={TARGET_H} ---")
        print(f"  min={eff_widths[0]:.0f}  "
              f"median={eff_widths[n//2]:.0f}  "
              f"p90={eff_widths[int(0.9*n)]:.0f}  "
              f"p99={eff_widths[min(int(0.99*n), n-1)]:.0f}  "
              f"max={eff_widths[-1]:.0f}")
        pct = over / n * 100
        if over == 0:
            print(f"  All crops fit within max_width={MAX_W} ✓")
        else:
            print(f"  {over}/{n} ({pct:.1f}%) exceed max_width={MAX_W} -> will be squished!")
            print(f"  Consider raising max_width in config.py or splitting long lines.")


if __name__ == "__main__":
    main()
