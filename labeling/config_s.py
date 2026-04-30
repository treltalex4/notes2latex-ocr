"""Configuration for the YOLO-based line slicer (slicer.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlicerConfig:
    # ── YOLO model ────────────────────────────────────────────────────────────
    model: str = "yolo26s.pt"
    """Weights filename inside labeling/yolo_models/.
    Place a trained best.pt there (e.g. copy from
    yolo_training/runs/<run>/weights/best.pt and rename)."""

    conf: float = 0.3
    """Minimum confidence to keep a line detection.
    Lower (0.20) = recall more faint/edge lines but adds noise.
    Higher (0.50) = only confident detections, may miss valid lines."""

    iou: float = 0.5
    """NMS IoU threshold for overlapping predictions.
    Lower = stricter de-duplication (drops more overlapping boxes).
    For tightly-packed lines, 0.4-0.5 works best."""

    imgsz: int = 1280
    """Inference image size (must be multiple of 32).
    Match the training imgsz of the model for best results.
    1280 is good for A4 scans; 1024 is faster but coarser."""

    device: str = "0"
    """GPU index ('0'), multiple ('0,1'), or 'cpu'."""

    # ── Image loading ─────────────────────────────────────────────────────────
    max_width: int = 0
    """Safety cap on input image width (px). 0 = no cap (use full resolution).
    YOLO does its own letterboxing internally, so downscaling here only hurts
    crop quality. Set to e.g. 4000 only if you hit OpenCV/PIL memory limits."""

    # ── Crop geometry ─────────────────────────────────────────────────────────
    pad_x: int = 28
    """Horizontal padding (px) added around each line bbox crop."""

    pad_y: int = 22
    """Vertical padding (px) added around each line bbox crop.
    Generous so superscripts/subscripts not in the bbox are still captured."""

    # ── Contour masking (clean overlapping neighbors) ─────────────────────────
    contour_mask: bool = True
    """If True, pixels inside the crop that belong to NEIGHBOUR line bboxes
    (overlapping with this line's bbox) are whitened. Produces cleaner
    training crops without parts of the line above/below leaking in."""

    contour_mask_dilate_px: int = 5
    """Dilation (px) added to this line's region before masking, so that
    thin strokes near the bbox edge are not accidentally erased."""

    # ── Garbage filtering ─────────────────────────────────────────────────────
    min_ink_density: float = 0.003
    """Minimum ratio of dark pixels to total crop area. Crops below this
    are considered empty/noise and discarded."""

    min_crop_aspect_ratio: float = 0.3
    """Minimum width/height ratio. Very tall-narrow crops (vertical lines,
    margin artifacts) are discarded."""

    min_line_height_px: int = 8
    """Absolute minimum height in pixels for a line crop to be saved."""

    # ── Reading order ─────────────────────────────────────────────────────────
    band_tolerance: float = 0.5
    """Two lines are in the same horizontal band (read left-to-right
    together) if their vertical centres differ by less than this fraction
    of the median line height. Helps with side-by-side formulas."""

    # ── Output ────────────────────────────────────────────────────────────────
    save_metadata: bool = True
    """Write a JSONL file with bbox, page info, and crop size for each line."""

    save_debug: bool = True
    """Save overlay images showing detected line boxes (for visual QA)."""

    debug_box_color: tuple = (0, 200, 0)
    """BGR colour for line bboxes in debug overlay (green by default)."""

    debug_text_color: tuple = (0, 0, 255)
    """BGR colour for line index numbers in debug overlay (red by default)."""
