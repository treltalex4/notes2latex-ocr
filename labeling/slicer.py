from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np

if TYPE_CHECKING:
    from config import Config


@dataclass
class SliceOptions:
    # ── Loading & deskew ──────────────────────────────────────────────────
    max_width: int = 2200
    deskew: bool = True

    # ── Foreground extraction ─────────────────────────────────────────────
    detect_dark_threshold: int = 75
    expand_dark_threshold: int = 130
    border_margin_px: int = 4

    # ── Line constraints ──────────────────────────────────────────────────
    min_line_height: int = 20
    min_line_width: int = 60

    # ── RLSA core segmentation ────────────────────────────────────────────
    rlsa_h_kernel_factor: float = 6.0
    rlsa_v_kernel_factor: float = 0.20
    min_cc_area_for_scale: int = 25

    # ── Orphan attachment (sub/superscripts, dots, accents) ──────────────
    orphan_v_distance_factor: float = 1.5
    orphan_h_tolerance_px: int = 28
    orphan_min_area: int = 3
    orphan_max_height_factor: float = 2.0

    # ── Splitting clearly merged lines ────────────────────────────────────
    split_height_factor: float = 3.0
    split_valley_ratio: float = 0.08
    split_min_run_factor: float = 0.30

    # ── Merging fragments of the same line ────────────────────────────────
    merge_y_overlap_ratio: float = 0.30

    # ── Edge inflation when ink is touching the bbox border ───────────────
    edge_touch_ratio: float = 0.012
    edge_expand_x: int = 14
    edge_expand_y: int = 10
    max_edge_expand_iters: int = 3

    # ── Final padding around each saved crop ──────────────────────────────
    pad_x: int = 16
    pad_y: int = 14

    # ── Perspective correction ────────────────────────────────────────────
    perspective_correction: bool = False
    perspective_min_lines: int = 6

    # ── Per-line deskew (OBB) ─────────────────────────────────────────────
    per_line_deskew: bool = True
    per_line_deskew_min_angle: float = 0.5

    # ── Baseline estimation for orphan assignment ─────────────────────────
    use_baseline_assignment: bool = True
    baseline_bin_width: int = 25

    # ── Garbage filtering ─────────────────────────────────────────────────
    min_ink_density: float = 0.015
    min_ink_components: int = 3
    min_crop_width_ratio: float = 0.04
    min_crop_aspect_ratio: float = 0.3

    # ── Contour masking ───────────────────────────────────────────────────
    contour_mask_lines: bool = True
    contour_mask_dilate: int = 3

    # ── Directed RLSA ─────────────────────────────────────────────────────
    directed_rlsa: bool = True

    # ── Misc ──────────────────────────────────────────────────────────────
    single_line: bool = False
    save_debug: bool = False

    @classmethod
    def from_config(cls, config: "Config") -> "SliceOptions":
        """Build SliceOptions from a Config object (slice_* fields)."""
        return cls(
            max_width=config.slice_max_width,
            deskew=config.slice_deskew,
            detect_dark_threshold=config.slice_detect_dark_threshold,
            expand_dark_threshold=config.slice_expand_dark_threshold,
            border_margin_px=config.slice_border_margin_px,
            min_line_height=config.slice_min_line_height,
            min_line_width=config.slice_min_line_width,
            rlsa_h_kernel_factor=config.slice_rlsa_h_kernel_factor,
            rlsa_v_kernel_factor=config.slice_rlsa_v_kernel_factor,
            min_cc_area_for_scale=config.slice_min_cc_area_for_scale,
            orphan_v_distance_factor=config.slice_orphan_v_distance_factor,
            orphan_h_tolerance_px=config.slice_orphan_h_tolerance_px,
            orphan_min_area=config.slice_orphan_min_area,
            orphan_max_height_factor=config.slice_orphan_max_height_factor,
            split_height_factor=config.slice_split_height_factor,
            split_valley_ratio=config.slice_split_valley_ratio,
            split_min_run_factor=config.slice_split_min_run_factor,
            merge_y_overlap_ratio=config.slice_merge_y_overlap_ratio,
            edge_touch_ratio=config.slice_edge_touch_ratio,
            edge_expand_x=config.slice_edge_expand_x,
            edge_expand_y=config.slice_edge_expand_y,
            max_edge_expand_iters=config.slice_max_edge_expand_iters,
            pad_x=config.slice_pad_x,
            pad_y=config.slice_pad_y,
            perspective_correction=config.slice_perspective_correction,
            perspective_min_lines=config.slice_perspective_min_lines,
            per_line_deskew=config.slice_per_line_deskew,
            per_line_deskew_min_angle=config.slice_per_line_deskew_min_angle,
            use_baseline_assignment=config.slice_use_baseline_assignment,
            baseline_bin_width=config.slice_baseline_bin_width,
            min_ink_density=config.slice_min_ink_density,
            min_ink_components=config.slice_min_ink_components,
            min_crop_width_ratio=config.slice_min_crop_width_ratio,
            min_crop_aspect_ratio=config.slice_min_crop_aspect_ratio,
            contour_mask_lines=config.slice_contour_mask_lines,
            contour_mask_dilate=config.slice_contour_mask_dilate,
            directed_rlsa=config.slice_directed_rlsa,
        )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_input_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _load_gray(image_path: Path, max_width: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _to_binary(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 15,
    )
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _foreground_without_grid(gray: np.ndarray, dark_threshold: int) -> np.ndarray:
    """Return cleaned foreground mask (255 = ink, 0 = background)."""
    h, w = gray.shape[:2]
    base = _to_binary(gray)

    _, dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    fg = cv2.bitwise_and(cv2.bitwise_not(base), dark_mask)

    # Detect and subtract notebook ruling lines.
    k_h = max(15, w // 40)
    k_v = max(15, h // 40)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v))
    horiz = cv2.morphologyEx(fg, cv2.MORPH_OPEN, h_kernel)
    vert = cv2.morphologyEx(fg, cv2.MORPH_OPEN, v_kernel)
    ruling = cv2.bitwise_or(horiz, vert)
    cleaned = cv2.bitwise_and(fg, cv2.bitwise_not(ruling))

    # Light denoise + reconnect strokes broken by binarization.
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    cleaned = cv2.dilate(
        cleaned,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    for label in range(1, n_labels):
        x, y, cw, ch, area = stats[label]
        if area < 10 or cw < 2 or ch < 2:
            continue
        filtered[labels == label] = 255
    return filtered


def _suppress_border_artifacts(mask: np.ndarray, border_margin_px: int) -> np.ndarray:
    h, w = mask.shape[:2]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()

    margin = max(1, border_margin_px)
    area_ratio_limit = 0.0015
    for label in range(1, n_labels):
        x, y, cw, ch, area = stats[label]
        x2 = x + cw - 1
        y2 = y + ch - 1
        touches = (
            x <= margin or y <= margin
            or x2 >= (w - 1 - margin) or y2 >= (h - 1 - margin)
        )
        if not touches:
            continue
        area_ratio = area / float(max(1, h * w))
        very_wide_thin = cw > int(0.30 * w) and ch < int(0.04 * h)
        very_tall_thin = ch > int(0.25 * h) and cw < int(0.03 * w)
        if area_ratio > area_ratio_limit and (very_wide_thin or very_tall_thin):
            out[labels == label] = 0
    return out


def _estimate_skew_angle(binary: np.ndarray) -> float:
    ys, xs = np.where(binary < 128)
    if len(xs) < 100:
        return 0.0
    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle -= 90
    return float(angle)


def _rotate_keep_canvas(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

# ──────────────────────────────────────────────────────────────────────────
# §3.2  Perspective correction
# ──────────────────────────────────────────────────────────────────────────

def _line_intersection(
    l1: tuple[int, int, int, int],
    l2: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (float(px), float(py))


def _correct_perspective(gray: np.ndarray, options: SliceOptions) -> np.ndarray:
    """Correct trapezoidal distortion using detected grid lines."""
    if not options.perspective_correction:
        return gray
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=w // 5, maxLineGap=10,
    )
    if lines is None or len(lines) < options.perspective_min_lines:
        return gray

    h_lines: list[tuple[int, int, int, int]] = []
    v_lines: list[tuple[int, int, int, int]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        length = np.sqrt(dx * dx + dy * dy)
        if angle < 15 and length > w * 0.2:
            h_lines.append((x1, y1, x2, y2))
        elif angle > 75 and length > h * 0.15:
            v_lines.append((x1, y1, x2, y2))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return gray

    h_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    v_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    top, bottom = h_lines[0], h_lines[-1]
    left, right = v_lines[0], v_lines[-1]

    corners = [
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    ]
    if any(c is None for c in corners):
        return gray
    src = np.array(corners, dtype=np.float32)

    # Sanity: corners must be inside the image (with margin)
    margin = max(w, h) * 0.1
    for cx, cy in src:
        if cx < -margin or cy < -margin or cx > w + margin or cy > h + margin:
            return gray

    dst_w = max(
        np.linalg.norm(src[1] - src[0]),
        np.linalg.norm(src[2] - src[3]),
    )
    dst_h = max(
        np.linalg.norm(src[3] - src[0]),
        np.linalg.norm(src[2] - src[1]),
    )
    if dst_w < 100 or dst_h < 100:
        return gray

    dst = np.array(
        [[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(
        gray, M, (int(dst_w), int(dst_h)),
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return corrected


# ──────────────────────────────────────────────────────────────────────────
# §3.6  Directed RLSA — dominant text angle estimation
# ──────────────────────────────────────────────────────────────────────────

def _estimate_dominant_text_angle(foreground_mask: np.ndarray) -> float:
    """Estimate dominant text-line angle via projection-profile analysis."""
    scale = 0.25
    h, w = foreground_mask.shape[:2]
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(foreground_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

    best_angle = 0.0
    best_score = 0.0
    # Coarse search
    for angle_deg in np.linspace(-10, 10, 41):
        M = cv2.getRotationMatrix2D((sw // 2, sh // 2), angle_deg, 1.0)
        rotated = cv2.warpAffine(small, M, (sw, sh), borderValue=0)
        proj = np.sum(rotated > 0, axis=1).astype(np.float64)
        score = float(np.sum(proj ** 2))
        if score > best_score:
            best_score = score
            best_angle = angle_deg
    # Fine search
    for angle_deg in np.linspace(best_angle - 1, best_angle + 1, 21):
        M = cv2.getRotationMatrix2D((sw // 2, sh // 2), angle_deg, 1.0)
        rotated = cv2.warpAffine(small, M, (sw, sh), borderValue=0)
        proj = np.sum(rotated > 0, axis=1).astype(np.float64)
        score = float(np.sum(proj ** 2))
        if score > best_score:
            best_score = score
            best_angle = angle_deg
    return best_angle


def _rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a binary mask by *angle_deg* (counter-clockwise positive)."""
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(mask, M, (w, h), borderValue=0)


# ──────────────────────────────────────────────────────────────────────────
# §3.4  Baseline estimation (slope, intercept) per line cluster
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class _LineBaseline:
    slope: float
    intercept: float

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept


def _estimate_baseline(
    foreground_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    bin_width: int = 25,
) -> _LineBaseline:
    """Fit a baseline through the ink-density peaks of a line bbox."""
    x1, y1, x2, y2 = bbox
    roi = foreground_mask[y1:y2 + 1, x1:x2 + 1]
    ys_local, xs_local = np.where(roi > 0)
    if xs_local.size < 10:
        return _LineBaseline(0.0, float((y1 + y2) / 2))

    # Bin ink pixels by X, find peak Y per bin
    peak_points: list[tuple[float, float]] = []
    roi_w = x2 - x1 + 1
    for bx in range(0, roi_w, bin_width):
        mask_bin = (xs_local >= bx) & (xs_local < bx + bin_width)
        if mask_bin.sum() < 5:
            continue
        bin_ys = ys_local[mask_bin]
        n_bins_hist = max(5, (int(bin_ys.max()) - int(bin_ys.min())) // 3 + 1)
        hist, edges = np.histogram(bin_ys, bins=n_bins_hist)
        peak_idx = int(np.argmax(hist))
        peak_y = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0
        peak_points.append((float(bx + bin_width / 2 + x1), float(peak_y + y1)))

    if len(peak_points) < 2:
        mid_y = float((y1 + y2) / 2)
        return _LineBaseline(0.0, mid_y)

    pts = np.array(peak_points)
    xs_fit, ys_fit = pts[:, 0], pts[:, 1]

    # RANSAC-like: fit, remove outliers, refit
    coeffs = np.polyfit(xs_fit, ys_fit, 1)
    residuals = np.abs(ys_fit - np.polyval(coeffs, xs_fit))
    threshold = max(3.0, float(np.median(residuals)) * 2.5)
    inliers = residuals < threshold
    if inliers.sum() >= 2:
        coeffs = np.polyfit(xs_fit[inliers], ys_fit[inliers], 1)

    return _LineBaseline(slope=float(coeffs[0]), intercept=float(coeffs[1]))


# ──────────────────────────────────────────────────────────────────────────
# §3.5  Garbage line filtering
# ──────────────────────────────────────────────────────────────────────────

def _filter_garbage_lines(
    foreground_mask: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    page_shape: tuple[int, int],
    options: SliceOptions,
) -> list[tuple[int, int, int, int]]:
    """Drop bboxes that contain too little ink or have degenerate shapes."""
    h_page, w_page = page_shape
    filtered: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in bboxes:
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        roi = foreground_mask[y1:y2 + 1, x1:x2 + 1]
        ink_count = int(np.count_nonzero(roi))
        ink_density = ink_count / max(1, bw * bh)

        if ink_density < options.min_ink_density:
            continue
        if bw < options.min_crop_width_ratio * w_page:
            continue
        if bh > 0 and bw / bh < options.min_crop_aspect_ratio:
            continue

        # Check minimum number of connected components
        n_cc, _, _, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        if (n_cc - 1) < options.min_ink_components:
            continue

        filtered.append((x1, y1, x2, y2))
    return filtered


# ──────────────────────────────────────────────────────────────────────────
# §3.1  Per-line OBB deskew crop
# ──────────────────────────────────────────────────────────────────────────

def _per_line_deskew_crop(
    gray: np.ndarray,
    foreground_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    options: SliceOptions,
) -> np.ndarray:
    """Crop a single line using its oriented bounding box for deskew."""
    x1, y1, x2, y2 = bbox

    if not options.per_line_deskew:
        return gray[y1:y2 + 1, x1:x2 + 1]

    roi_mask = foreground_mask[y1:y2 + 1, x1:x2 + 1]
    ys_local, xs_local = np.where(roi_mask > 0)
    if xs_local.size < 20:
        return gray[y1:y2 + 1, x1:x2 + 1]

    # OBB in page coordinates
    pts = np.column_stack((xs_local + x1, ys_local + y1)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    # Normalize angle to [-45, 45]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    if abs(angle) < options.per_line_deskew_min_angle:
        return gray[y1:y2 + 1, x1:x2 + 1]

    # Expand ROI to avoid clipping during rotation
    cx, cy = rect[0]
    rw, rh = rect[1]
    side = int(max(rw, rh) * 1.3)
    half = side // 2
    h_page, w_page = gray.shape[:2]
    rx1 = max(0, int(cx) - half)
    ry1 = max(0, int(cy) - half)
    rx2 = min(w_page, int(cx) + half)
    ry2 = min(h_page, int(cy) + half)

    roi_gray = gray[ry1:ry2, rx1:rx2].copy()
    local_cx = cx - rx1
    local_cy = cy - ry1
    M = cv2.getRotationMatrix2D((local_cx, local_cy), angle, 1.0)
    rotated = cv2.warpAffine(
        roi_gray, M, (roi_gray.shape[1], roi_gray.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )

    # Also rotate the mask to find tight ink bbox
    roi_fg = foreground_mask[ry1:ry2, rx1:rx2].copy()
    rotated_mask = cv2.warpAffine(
        roi_fg, M, (roi_fg.shape[1], roi_fg.shape[0]),
        borderValue=0,
    )

    ys_r, xs_r = np.where(rotated_mask > 0)
    if xs_r.size < 4:
        return gray[y1:y2 + 1, x1:x2 + 1]

    crop = rotated[ys_r.min():ys_r.max() + 1, xs_r.min():xs_r.max() + 1]
    if crop.size == 0:
        return gray[y1:y2 + 1, x1:x2 + 1]
    return crop


# ──────────────────────────────────────────────────────────────────────────
# §3.3  Contour masking — white-out non-ink regions in crops
# ──────────────────────────────────────────────────────────────────────────

def _contour_mask_crop(
    crop_gray: np.ndarray,
    crop_mask: np.ndarray,
    dilate_px: int = 3,
) -> np.ndarray:
    """White-out pixels in *crop_gray* that don't belong to this line's ink."""
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        expanded = cv2.dilate(crop_mask, kernel, iterations=1)
    else:
        expanded = crop_mask
    result = crop_gray.copy()
    result[expanded == 0] = 255
    return result


# ──────────────────────────────────────────────────────────────────────────
# Core segmentation
# ──────────────────────────────────────────────────────────────────────────

def _estimate_text_scale(foreground_mask: np.ndarray, options: SliceOptions) -> float:
    """Return the typical character height (median of mid-sized CC heights).

    All downstream kernels and thresholds scale with this, so the algorithm
    works for both small and large handwriting without retuning.
    """
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
    heights: list[int] = []
    for i in range(1, n_labels):
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < options.min_cc_area_for_scale or h < 4:
            continue
        heights.append(h)
    if not heights:
        return 28.0  # safe fallback

    arr = np.asarray(heights, dtype=np.float32)
    # Trim extremes (very tall brackets/dividers, accents).
    p20, p80 = np.percentile(arr, [20, 80])
    middle = arr[(arr >= p20) & (arr <= p80)]
    if middle.size == 0:
        return float(np.median(arr))
    return float(np.median(middle))


def _segment_lines_rlsa(
    foreground_mask: np.ndarray,
    median_h: float,
    options: SliceOptions,
) -> list[tuple[int, int, int, int]]:
    """RLSA-based line clustering with optional directed (angle-aware) mode.

    When *directed_rlsa* is True the foreground mask is first rotated so that
    the dominant text direction becomes horizontal.  RLSA is applied on the
    aligned image and the resulting bboxes are mapped back to the original
    coordinate system.
    """
    work_mask = foreground_mask
    text_angle = 0.0
    if options.directed_rlsa:
        text_angle = _estimate_dominant_text_angle(foreground_mask)
        if abs(text_angle) > 0.3:
            work_mask = _rotate_mask(foreground_mask, text_angle)

    h_kw = max(int(median_h * options.rlsa_h_kernel_factor), 40)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kw, 1))
    smeared = cv2.morphologyEx(work_mask, cv2.MORPH_CLOSE, h_kernel)

    v_kh = max(3, int(median_h * options.rlsa_v_kernel_factor)) | 1  # odd
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kh))
    smeared = cv2.morphologyEx(smeared, cv2.MORPH_CLOSE, v_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(smeared, connectivity=8)

    # If we rotated, prepare inverse map to original coords
    need_unrotate = options.directed_rlsa and abs(text_angle) > 0.3
    if need_unrotate:
        hh, ww = foreground_mask.shape[:2]
        M_inv = cv2.getRotationMatrix2D((ww // 2, hh // 2), -text_angle, 1.0)

    bboxes: list[tuple[int, int, int, int]] = []
    for label in range(1, n_labels):
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        if ch < options.min_line_height or cw < options.min_line_width:
            continue
        cluster_mask = (labels == label).astype(np.uint8) * 255
        ink = cv2.bitwise_and(work_mask, cluster_mask)
        ys, xs = np.where(ink > 0)
        if xs.size < 8:
            continue
        bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
        if xs.size / max(1, bbox_area) < 0.006:
            continue

        if need_unrotate:
            # Map ink pixel coordinates back to original image
            pts = np.column_stack((xs, ys)).astype(np.float64)
            ones = np.ones((pts.shape[0], 1), dtype=np.float64)
            pts_h = np.hstack((pts, ones))
            mapped = (M_inv @ pts_h.T).T
            mx, my = mapped[:, 0], mapped[:, 1]
            # Clip to image bounds
            hh, ww = foreground_mask.shape[:2]
            mx = np.clip(mx, 0, ww - 1)
            my = np.clip(my, 0, hh - 1)
            bboxes.append((int(mx.min()), int(my.min()), int(mx.max()), int(my.max())))
        else:
            bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    bboxes.sort(key=lambda b: b[1])
    return bboxes


def _split_merged_lines(
    foreground_mask: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    median_h: float,
    options: SliceOptions,
) -> list[tuple[int, int, int, int]]:
    """Split bboxes that are clearly multiple lines glued together.

    Triggers only when the bbox is much taller than median_h AND a clear ink
    valley exists in its row profile. This avoids cutting tall but legitimate
    single-line content (matrices, big sigma, fractions).
    """
    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in bboxes:
        h = y2 - y1 + 1
        if h < median_h * options.split_height_factor:
            out.append((x1, y1, x2, y2))
            continue

        roi = (foreground_mask[y1:y2 + 1, x1:x2 + 1] > 0).astype(np.float32)
        row_sum = roi.sum(axis=1)
        peak = float(row_sum.max())
        if peak <= 0:
            out.append((x1, y1, x2, y2))
            continue

        win = max(3, int(median_h * 0.35)) | 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        row_smooth = np.convolve(row_sum, kernel, mode="same")

        valley_thr = peak * options.split_valley_ratio
        is_valley = row_smooth < valley_thr

        # Ignore valleys touching the top/bottom edges — those are just padding.
        margin = int(options.min_line_height * 0.6)
        is_valley[:margin] = False
        is_valley[-margin:] = False

        min_run = max(3, int(median_h * options.split_min_run_factor))
        runs: list[tuple[int, int]] = []
        i = 0
        while i < h:
            if not is_valley[i]:
                i += 1
                continue
            j = i
            while j < h and is_valley[j]:
                j += 1
            if (j - i) >= min_run:
                runs.append((i, j - 1))
            i = j

        if not runs:
            out.append((x1, y1, x2, y2))
            continue

        midpoints = [(rs + re) // 2 for rs, re in runs]
        boundaries = [0] + midpoints + [h - 1]
        for k in range(len(boundaries) - 1):
            sy1 = y1 + boundaries[k]
            sy2 = y1 + boundaries[k + 1]
            if sy2 - sy1 + 1 < options.min_line_height:
                continue
            out.append((x1, sy1, x2, sy2))

    out.sort(key=lambda b: b[1])
    return out


def _attach_orphans(
    foreground_mask: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    median_h: float,
    options: SliceOptions,
    baselines: list[_LineBaseline] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Pull nearby small CCs into the closest line bbox.

    When *baselines* are provided (one per bbox), distance is measured from the
    orphan's centre to the baseline rather than to the AABB edges.  This is far
    more accurate for tilted text.
    """
    if not bboxes:
        return bboxes

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
    components: list[tuple[int, int, int, int]] = []
    for i in range(1, n_labels):
        x, y, cw, ch, area = stats[i]
        if area < options.orphan_min_area:
            continue
        components.append((int(x), int(y), int(x + cw - 1), int(y + ch - 1)))

    v_max = median_h * options.orphan_v_distance_factor
    h_tol = options.orphan_h_tolerance_px
    max_comp_h = median_h * options.orphan_max_height_factor

    use_bl = baselines is not None and len(baselines) == len(bboxes)

    growth: list[list[tuple[int, int, int, int]]] = [[] for _ in bboxes]

    for cx1, cy1, cx2, cy2 in components:
        comp_h = cy2 - cy1 + 1
        comp_cx = (cx1 + cx2) / 2.0
        comp_cy = (cy1 + cy2) / 2.0
        best_li = -1
        best_v_dist = v_max + 1
        already_inside = False

        for li, (x1, y1, x2, y2) in enumerate(bboxes):
            inside = (cx1 >= x1 and cx2 <= x2 and cy1 >= y1 and cy2 <= y2)
            if inside:
                already_inside = True
                break

            if cx2 < x1 - h_tol or cx1 > x2 + h_tol:
                continue
            if comp_h > max_comp_h:
                continue

            if use_bl:
                bl_y = baselines[li].y_at(comp_cx)
                v_dist = abs(comp_cy - bl_y)
            else:
                if cy2 < y1:
                    v_dist = y1 - cy2
                elif cy1 > y2:
                    v_dist = cy1 - y2
                else:
                    v_dist = 0

            if v_dist > v_max:
                continue
            if v_dist < best_v_dist:
                best_v_dist = v_dist
                best_li = li

        if already_inside:
            continue
        if best_li >= 0:
            growth[best_li].append((cx1, cy1, cx2, cy2))

    expanded: list[tuple[int, int, int, int]] = []
    for li, (x1, y1, x2, y2) in enumerate(bboxes):
        ex1, ey1, ex2, ey2 = x1, y1, x2, y2
        for cx1, cy1, cx2, cy2 in growth[li]:
            ex1 = min(ex1, cx1)
            ey1 = min(ey1, cy1)
            ex2 = max(ex2, cx2)
            ey2 = max(ey2, cy2)
        expanded.append((ex1, ey1, ex2, ey2))
    return expanded


def _resolve_overlaps(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Trim pairs of vertically overlapping bboxes at their gap midpoint.

    After orphan attachment, bboxes can grow toward each other (e.g. a line's
    bbox expands down to capture integral bounds while the next line expands up
    for its own subscripts). Without trimming, the merge step would incorrectly
    collapse two real lines into one.  Splitting at the midpoint of the overlap
    gives each line its fair share of the shared region while ensuring no bbox
    eats into its neighbor's primary content.
    """
    if len(bboxes) < 2:
        return list(bboxes)

    bboxes = sorted(bboxes, key=lambda b: b[1])
    out: list[list[int]] = [list(b) for b in bboxes]
    for i in range(len(out) - 1):
        y2a = out[i][3]
        y1b = out[i + 1][1]
        if y2a >= y1b:  # overlap exists
            mid = (y2a + y1b) // 2
            out[i][3] = mid - 1
            out[i + 1][1] = mid
    return [(r[0], r[1], r[2], r[3]) for r in out if r[3] > r[1]]


def _merge_overlapping_lines(
    bboxes: list[tuple[int, int, int, int]],
    options: SliceOptions,
) -> list[tuple[int, int, int, int]]:
    """Merge bboxes whose Y-overlap is high — fragments of the same line.

    Runs multiple passes until no further merges happen, so a chain of
    partially-overlapping fragments (A∩B, B∩C but not A∩C directly) all
    collapse into one box.
    """
    if len(bboxes) < 2:
        return list(bboxes)

    current = sorted(bboxes, key=lambda b: b[1])
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = [current[0]]
        for x1, y1, x2, y2 in current[1:]:
            px1, py1, px2, py2 = merged[-1]
            oy1 = max(py1, y1)
            oy2 = min(py2, y2)
            overlap = max(0, oy2 - oy1 + 1)
            h_min = min(py2 - py1 + 1, y2 - y1 + 1)
            if h_min > 0 and (overlap / float(h_min)) >= options.merge_y_overlap_ratio:
                merged[-1] = (
                    min(px1, x1), min(py1, y1),
                    max(px2, x2), max(py2, y2),
                )
                changed = True
            else:
                merged.append((x1, y1, x2, y2))
        current = merged
    return current


def _inflate_bbox_if_edge_touched(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    options: SliceOptions,
) -> tuple[int, int, int, int]:
    h_page, w_page = mask.shape[:2]
    x1, y1, x2, y2 = bbox

    for _ in range(options.max_edge_expand_iters):
        roi = mask[y1:y2 + 1, x1:x2 + 1]
        total = int(np.count_nonzero(roi))
        if total == 0:
            break
        eh = min(2, roi.shape[0])
        ew = min(2, roi.shape[1])
        top = float(np.count_nonzero(roi[:eh, :])) / total
        bot = float(np.count_nonzero(roi[-eh:, :])) / total
        lft = float(np.count_nonzero(roi[:, :ew])) / total
        rgt = float(np.count_nonzero(roi[:, -ew:])) / total

        changed = False
        if top > options.edge_touch_ratio and y1 > 0:
            y1 = max(0, y1 - options.edge_expand_y); changed = True
        if bot > options.edge_touch_ratio and y2 < (h_page - 1):
            y2 = min(h_page - 1, y2 + options.edge_expand_y); changed = True
        if lft > options.edge_touch_ratio and x1 > 0:
            x1 = max(0, x1 - options.edge_expand_x); changed = True
        if rgt > options.edge_touch_ratio and x2 < (w_page - 1):
            x2 = min(w_page - 1, x2 + options.edge_expand_x); changed = True
        if not changed:
            break
    return (x1, y1, x2, y2)


def _apply_padding(
    bboxes: list[tuple[int, int, int, int]],
    page_shape: tuple[int, int],
    options: SliceOptions,
) -> list[tuple[int, int, int, int]]:
    h, w = page_shape
    out: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in bboxes:
        out.append((
            max(0, x1 - options.pad_x),
            max(0, y1 - options.pad_y),
            min(w - 1, x2 + options.pad_x),
            min(h - 1, y2 + options.pad_y),
        ))
    return out


def _save_debug_overlay(
    gray: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    out_path: Path,
) -> None:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            canvas, str(idx),
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), canvas)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ──────────────────────────────────────────────────────────────────────────

def _slice_one_page(
    image_path: Path,
    output_dir: Path,
    meta_path: Path,
    debug_dir: Path | None,
    options: SliceOptions,
) -> list[str]:
    gray = _load_gray(image_path, max_width=options.max_width)

    # §3.2 — Perspective correction (before everything else)
    gray = _correct_perspective(gray, options)

    detect_mask = _foreground_without_grid(gray, options.detect_dark_threshold)
    expand_mask = _foreground_without_grid(gray, options.expand_dark_threshold)
    detect_mask = _suppress_border_artifacts(detect_mask, options.border_margin_px)
    expand_mask = _suppress_border_artifacts(expand_mask, options.border_margin_px)

    # Global deskew (kept as a coarse first pass)
    angle = 0.0
    if options.deskew:
        angle = _estimate_skew_angle(cv2.bitwise_not(detect_mask))
        if abs(angle) > 0.3:
            gray = _rotate_keep_canvas(gray, angle)
            detect_mask = _foreground_without_grid(gray, options.detect_dark_threshold)
            expand_mask = _foreground_without_grid(gray, options.expand_dark_threshold)
            detect_mask = _suppress_border_artifacts(detect_mask, options.border_margin_px)
            expand_mask = _suppress_border_artifacts(expand_mask, options.border_margin_px)

    median_h = _estimate_text_scale(detect_mask, options)

    # Pipeline:
    # 1. §3.6 — Directed RLSA: rotate mask to align text, cluster, map back.
    # 2. Split if a cluster is clearly multiple lines glued together.
    # 3. §3.4 — Estimate baselines for each line cluster.
    # 4. §3.4 — Pull in orphans using baseline distance (not AABB).
    # 5. Resolve overlaps, merge fragments, inflate edges.
    # 6. §3.5 — Filter garbage lines (low ink, degenerate shape).
    # 7. §3.1 — Per-line OBB deskew + §3.3 contour masking when cropping.
    bboxes = _segment_lines_rlsa(detect_mask, median_h, options)
    bboxes = _split_merged_lines(detect_mask, bboxes, median_h, options)

    # §3.4 — Baseline estimation for smarter orphan attachment
    baselines: list[_LineBaseline] | None = None
    if options.use_baseline_assignment and bboxes:
        baselines = [
            _estimate_baseline(detect_mask, b, options.baseline_bin_width)
            for b in bboxes
        ]

    bboxes = _attach_orphans(expand_mask, bboxes, median_h, options, baselines)
    bboxes = _resolve_overlaps(bboxes)
    bboxes = _merge_overlapping_lines(bboxes, options)
    bboxes = [_inflate_bbox_if_edge_touched(expand_mask, b, options) for b in bboxes]
    bboxes = _resolve_overlaps(bboxes)

    # §3.5 — Garbage filtering
    bboxes = _filter_garbage_lines(expand_mask, bboxes, gray.shape[:2], options)

    bboxes = _apply_padding(bboxes, gray.shape[:2], options)
    bboxes.sort(key=lambda b: b[1])

    if options.single_line and bboxes:
        bboxes = [max(bboxes, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))]

    if debug_dir is not None:
        _ensure_dir(debug_dir)
        _save_debug_overlay(gray, bboxes, debug_dir / f"{image_path.stem}_overlay.png")

    saved_files: list[str] = []
    with meta_path.open("a", encoding="utf-8") as meta_f:
        for i, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
            # §3.1 — Per-line OBB deskew crop
            crop = _per_line_deskew_crop(gray, expand_mask, (x1, y1, x2, y2), options)
            if crop.size == 0:
                continue

            # §3.3 — Contour masking: white-out non-ink pixels
            if options.contour_mask_lines:
                crop_mask = expand_mask[y1:y2 + 1, x1:x2 + 1]
                # If per-line deskew changed the crop shape, fall back to no masking
                if crop.shape == crop_mask.shape:
                    crop = _contour_mask_crop(crop, crop_mask, options.contour_mask_dilate)

            line_name = f"{image_path.stem}_line_{i:02d}.png"
            line_path = output_dir / line_name
            cv2.imwrite(str(line_path), crop)
            saved_files.append(str(line_path))

            record = {
                "line_id": f"{image_path.stem}_line_{i:02d}",
                "file": line_name,
                "page": image_path.name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "angle": round(angle, 4),
                "median_h": round(median_h, 1),
                "quality": "ok",
            }
            meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return saved_files


def slice_page_into_lines(
    image_path: str,
    output_dir: str,
    config: "Config | None" = None,
) -> list[str]:
    """Convenience wrapper for direct imports.

    Pass a Config object to use project-wide slicer settings; omit for defaults.
    """
    input_path = Path(image_path)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    meta_path = out_dir.parent / "slices_meta.jsonl"
    options = SliceOptions.from_config(config) if config is not None else SliceOptions()
    return _slice_one_page(
        image_path=input_path,
        output_dir=out_dir,
        meta_path=meta_path,
        debug_dir=None,
        options=options,
    )


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Slice page images into line crops.")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output", required=True, help="Output directory for sliced lines.")
    parser.add_argument("--meta", default="my_dataset/slices_meta.jsonl",
                        help="Path to JSONL file with slice metadata.")
    parser.add_argument("--debug-dir", default="my_dataset/debug_slices",
                        help="Directory for debug overlays.")

    parser.add_argument("--max-width", type=int, default=2200)
    parser.add_argument("--detect-dark-threshold", type=int, default=75)
    parser.add_argument("--expand-dark-threshold", type=int, default=130)
    parser.add_argument("--border-margin", type=int, default=4)
    parser.add_argument("--min-line-height", type=int, default=20)
    parser.add_argument("--min-line-width", type=int, default=60)

    parser.add_argument("--rlsa-h-factor", type=float, default=6.0,
                        help="Horizontal RLSA kernel = factor x median_h.")
    parser.add_argument("--rlsa-v-factor", type=float, default=0.20)

    parser.add_argument("--orphan-v-factor", type=float, default=1.5)
    parser.add_argument("--orphan-h-tolerance", type=int, default=28)
    parser.add_argument("--orphan-min-area", type=int, default=3)
    parser.add_argument("--orphan-max-height-factor", type=float, default=2.0)

    parser.add_argument("--split-height-factor", type=float, default=3.0)
    parser.add_argument("--split-valley-ratio", type=float, default=0.08)
    parser.add_argument("--split-min-run-factor", type=float, default=0.30)

    parser.add_argument("--merge-y-overlap-ratio", type=float, default=0.30)

    parser.add_argument("--edge-touch-ratio", type=float, default=0.012)
    parser.add_argument("--edge-expand-x", type=int, default=14)
    parser.add_argument("--edge-expand-y", type=int, default=10)
    parser.add_argument("--max-edge-expand-iters", type=int, default=3)

    parser.add_argument("--pad-x", type=int, default=16)
    parser.add_argument("--pad-y", type=int, default=14)

    # New feature flags
    parser.add_argument("--perspective", choices=["on", "off"], default="off",
                        help="Enable perspective correction for notebook photos.")
    parser.add_argument("--perspective-min-lines", type=int, default=6)
    parser.add_argument("--per-line-deskew", choices=["on", "off"], default="on",
                        help="Enable per-line OBB deskew.")
    parser.add_argument("--per-line-deskew-min-angle", type=float, default=0.5)
    parser.add_argument("--baseline-assignment", choices=["on", "off"], default="on",
                        help="Use baseline distance for orphan attachment.")
    parser.add_argument("--baseline-bin-width", type=int, default=25)
    parser.add_argument("--min-ink-density", type=float, default=0.015)
    parser.add_argument("--min-ink-components", type=int, default=3)
    parser.add_argument("--min-crop-width-ratio", type=float, default=0.04)
    parser.add_argument("--min-crop-aspect-ratio", type=float, default=0.3)
    parser.add_argument("--contour-mask", choices=["on", "off"], default="on",
                        help="White-out non-ink pixels in each crop.")
    parser.add_argument("--contour-mask-dilate", type=int, default=3)
    parser.add_argument("--directed-rlsa", choices=["on", "off"], default="on",
                        help="Rotate mask to align text before RLSA.")

    parser.add_argument("--single-line", action="store_true",
                        help="Force one output line per image (largest bbox).")
    parser.add_argument("--deskew", choices=["on", "off"], default="on",
                        help="Enable/disable global skew correction.")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save overlay images of detected line boxes.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    output_dir = Path(args.output)
    _ensure_dir(output_dir)
    meta_path = Path(args.meta)
    _ensure_dir(meta_path.parent)

    options = SliceOptions(
        max_width=args.max_width,
        detect_dark_threshold=args.detect_dark_threshold,
        expand_dark_threshold=args.expand_dark_threshold,
        border_margin_px=args.border_margin,
        min_line_height=args.min_line_height,
        min_line_width=args.min_line_width,
        rlsa_h_kernel_factor=args.rlsa_h_factor,
        rlsa_v_kernel_factor=args.rlsa_v_factor,
        orphan_v_distance_factor=args.orphan_v_factor,
        orphan_h_tolerance_px=args.orphan_h_tolerance,
        orphan_min_area=args.orphan_min_area,
        orphan_max_height_factor=args.orphan_max_height_factor,
        split_height_factor=args.split_height_factor,
        split_valley_ratio=args.split_valley_ratio,
        split_min_run_factor=args.split_min_run_factor,
        merge_y_overlap_ratio=args.merge_y_overlap_ratio,
        edge_touch_ratio=args.edge_touch_ratio,
        edge_expand_x=args.edge_expand_x,
        edge_expand_y=args.edge_expand_y,
        max_edge_expand_iters=args.max_edge_expand_iters,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        perspective_correction=args.perspective == "on",
        perspective_min_lines=args.perspective_min_lines,
        per_line_deskew=args.per_line_deskew == "on",
        per_line_deskew_min_angle=args.per_line_deskew_min_angle,
        use_baseline_assignment=args.baseline_assignment == "on",
        baseline_bin_width=args.baseline_bin_width,
        min_ink_density=args.min_ink_density,
        min_ink_components=args.min_ink_components,
        min_crop_width_ratio=args.min_crop_width_ratio,
        min_crop_aspect_ratio=args.min_crop_aspect_ratio,
        contour_mask_lines=args.contour_mask == "on",
        contour_mask_dilate=args.contour_mask_dilate,
        directed_rlsa=args.directed_rlsa == "on",
        single_line=args.single_line,
        deskew=args.deskew == "on",
        save_debug=args.save_debug,
    )

    debug_dir = Path(args.debug_dir) if args.save_debug else None
    if debug_dir is not None:
        _ensure_dir(debug_dir)

    all_saved: list[str] = []
    images = list(_iter_input_images(input_path))
    if not images:
        raise ValueError(f"No images found in input path: {input_path}")

    for img_path in images:
        saved = _slice_one_page(
            image_path=img_path,
            output_dir=output_dir,
            meta_path=meta_path,
            debug_dir=debug_dir,
            options=options,
        )
        all_saved.extend(saved)
        print(f"[slicer] {img_path.name}: {len(saved)} lines")

    print(f"[slicer] done. total lines: {len(all_saved)}")
    print(f"[slicer] output dir: {output_dir}")
    print(f"[slicer] meta file: {meta_path}")
    if debug_dir is not None:
        print(f"[slicer] debug dir: {debug_dir}")


if __name__ == "__main__":
    main()
