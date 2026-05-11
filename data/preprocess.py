import random

import albumentations as A
import cv2
import numpy as np
import torch


PAD_VALUE = 255  # белый фон


def load_image(image_path: str) -> np.ndarray | None:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def crop_to_content(img: np.ndarray, threshold: int = 250, padding: int = 8) -> np.ndarray:
    mask = img < threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = img.shape
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding + 1)

    return img[rmin:rmax, cmin:cmax]


def binarize(img: np.ndarray) -> np.ndarray:
    if len(np.unique(img)) <= 2:
        return img

    # blockSize scales with image height so the neighborhood covers ~1/6 of height;
    # must be odd and at least 11
    h = img.shape[0]
    block_size = max(11, (h // 6) | 1)

    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=5,
    )
    return cv2.medianBlur(binary, 3)


def resize_preserve_aspect(img: np.ndarray, target_h: int, max_w: int) -> np.ndarray:
    """Resize с СТРОГИМ сохранением аспекта.

    Если естественная new_w (при scale=target_h/h) превышает max_w —
    масштабируем по min ratio чтобы вписаться в (target_h, max_w), и
    паддим высоту белым до target_h. Так модель никогда не видит
    horizontally-squished картинок.
    """
    h, w = img.shape
    scale_h = target_h / h
    scale_w = max_w / w
    scale = min(scale_h, scale_w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Pad по высоте до target_h белым (PAD_VALUE=255). Большинству формул
    # этот pad не нужен — паддятся только экстремально широкие (aspect >
    # max_w/target_h).
    if new_h < target_h:
        pad_total = target_h - new_h
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=PAD_VALUE,
        )

    return img


def overlay_grid(
    img: np.ndarray,
    intensity_range: tuple[int, int] = (160, 220),
    cell_range: tuple[int, int] = (20, 50),
    thickness_choices: tuple[int, ...] = (1, 1, 1, 2),
    jitter: float = 0.04,
    line_noise: int = 15,
) -> np.ndarray:
    """См. документацию ниже — defaults дублируют config.grid_* для backward-compat."""
    """Накладывает тетрадную сетку на grayscale изображение.

    Имитирует разные тетради: размер клетки, толщина линий, насыщенность
    варьируются. Линии слегка дрожат и имеют шум интенсивности — реальные
    тетради тоже не идеально прямые.

    intensity_range: насколько тёмные линии (160-220 = серый, текст обычно 50-100).
    cell_range:      размер клетки в пикселях.
    thickness_choices: 1px чаще чем 2px.
    jitter:          доля cell_size — насколько линии могут смещаться.
    line_noise:      ± шум интенсивности отдельных линий.
    """
    h, w = img.shape
    cell_size = random.randint(*cell_range)
    intensity = random.randint(*intensity_range)
    thickness = random.choice(thickness_choices)

    x_offset = random.randint(0, cell_size)
    y_offset = random.randint(0, cell_size)

    jitter_px = max(1, int(cell_size * jitter))
    grid = np.full_like(img, 255)

    # Вертикальные линии
    for x in range(x_offset, w + 1, cell_size):
        dx = random.randint(-jitter_px, jitter_px)
        x_drawn = max(0, min(w - 1, x + dx))
        li = max(80, min(245, intensity + random.randint(-line_noise, line_noise)))
        cv2.line(grid, (x_drawn, 0), (x_drawn, h - 1), li, thickness)

    # Горизонтальные линии
    for y in range(y_offset, h + 1, cell_size):
        dy = random.randint(-jitter_px, jitter_px)
        y_drawn = max(0, min(h - 1, y + dy))
        li = max(80, min(245, intensity + random.randint(-line_noise, line_noise)))
        cv2.line(grid, (0, y_drawn), (w - 1, y_drawn), li, thickness)

    # min: тёмный текст остаётся тёмным, светлый фон затемняется до интенсивности сетки.
    # Это семантически правильно: грид появляется только на фоне, не поверх чернил.
    return np.minimum(img, grid)


def apply_augmentations(
    img: np.ndarray,
    dataset_type: str,
    elastic_p: float,
    elastic_alpha: int,
    elastic_sigma: int,
    strength: float,
    elastic_factor: float = 1.0,
    grid_aug_prob: float = 0.0,
    # === Все параметры аугментаций — дефолты дублируют config.py ===
    dilate_erode_prob: float = 0.4,
    noise_prob: float = 0.3,
    noise_sigma: float = 8.0,
    blur_prob: float = 0.3,
    brightness_prob: float = 0.3,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    affine_prob: float = 0.3,
    affine_rotate_deg: float = 2.0,
    affine_scale_pct: float = 0.05,
    grid_cell_range: tuple[int, int] = (20, 50),
    grid_intensity_range: tuple[int, int] = (160, 220),
    grid_jitter: float = 0.04,
    grid_line_noise: int = 15,
) -> np.ndarray:
    """Применяет все аугментации в порядке: grid → elastic → dilate/erode →
    noise → blur → brightness → affine. Каждая бросает кости отдельно."""
    # Grid augmentation: рисуем тетрадную клетку. Применяется ПЕРВЫМ —
    # дальше elastic warpит грид вместе с текстом (страница как под углом).
    if grid_aug_prob > 0 and random.random() < grid_aug_prob:
        img = overlay_grid(
            img,
            intensity_range=grid_intensity_range,
            cell_range=grid_cell_range,
            jitter=grid_jitter,
            line_noise=grid_line_noise,
        )

    # ElasticTransform: эффективный p = elastic_p × elastic_factor.
    effective_p = elastic_p * elastic_factor
    if effective_p > 0:
        h = img.shape[0]
        scaled_alpha = float(elastic_alpha) * h / 128.0
        scaled_sigma = float(elastic_sigma) * h / 128.0
        img = A.ElasticTransform(
            alpha=scaled_alpha,
            sigma=scaled_sigma,
            p=effective_p,
        )(image=img)["image"]

    if strength <= 0:
        return img

    # Толщина чернил: dilate (тоньше) или erode (толще).
    if random.random() < dilate_erode_prob * strength:
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        if random.random() < 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)

    # Шум камеры.
    if random.random() < noise_prob * strength:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Расфокус.
    if random.random() < blur_prob * strength:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Яркость и контраст.
    if random.random() < brightness_prob * strength:
        img = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=1.0,
        )(image=img)["image"]

    # Лёгкий наклон и масштаб (белый фон при заполнении).
    if random.random() < affine_prob * strength:
        img = A.Affine(
            rotate=(-affine_rotate_deg, affine_rotate_deg),
            scale=(1.0 - affine_scale_pct, 1.0 + affine_scale_pct),
            fill=PAD_VALUE,
            p=1.0,
        )(image=img)["image"]

    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = (tensor - 0.5) / 0.5   # [0,1] → [-1,1]
    return tensor.unsqueeze(0)       # [H,W] → [1,H,W]


def preprocess_image(
    image_path: str,
    target_h: int,
    max_w: int,
    augment: bool = False,
    dataset_type: str = "im2latex",
    elastic_p: float = 0.0,
    elastic_alpha: int = 0,
    elastic_sigma: int = 0,
    strength: float = 0.0,
    grid_aug_prob: float = 0.0,
) -> torch.Tensor | None:
    """Pipeline: load → crop_to_content → resize_preserve_aspect → augment → tensor.
    NB: binarize намеренно отсутствует — модель работает с grayscale.
    Грид/шум устраняется через grid_aug + elastic в augmentations, не через threshold.
    """
    img = load_image(image_path)
    if img is None:
        return None

    img = crop_to_content(img)
    img = resize_preserve_aspect(img, target_h, max_w)

    if augment:
        img = apply_augmentations(
            img, dataset_type, elastic_p, elastic_alpha, elastic_sigma, strength,
            grid_aug_prob=grid_aug_prob,
        )

    return to_tensor(img)
