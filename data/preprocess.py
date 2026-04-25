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

    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=5,
    )
    return cv2.medianBlur(binary, 3)


def resize_preserve_aspect(img: np.ndarray, target_h: int, max_w: int) -> np.ndarray:
    h, w = img.shape
    new_w = max(1, int(w * target_h / h))
    if new_w > max_w:
        new_w = max_w
    interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, target_h), interpolation=interp)


def apply_augmentations(
    img: np.ndarray,
    dataset_type: str,  # "im2latex" | "synthetic" | "handwritten"
    elastic_p: float,   # 0.0 во время warmup-фазы
    elastic_alpha: int,
    elastic_sigma: int,
    strength: float,    # 0..1, линейно нарастает по эпохам
) -> np.ndarray:
    # ElasticTransform — только на im2latex и synthetic (не на handwritten)
    if elastic_p > 0 and dataset_type != "handwritten":
        img = A.ElasticTransform(
            alpha=float(elastic_alpha),
            sigma=float(elastic_sigma),
            p=elastic_p,
        )(image=img)["image"]

    if strength <= 0:
        return img

    # Толщина чернил: dilate (тоньше) или erode (толще)
    if random.random() < 0.4 * strength:
        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        if random.random() < 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)

    # Шум камеры
    if random.random() < 0.3 * strength:
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Расфокус
    if random.random() < 0.3 * strength:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Яркость и контраст
    if random.random() < 0.3 * strength:
        img = A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=1.0,
        )(image=img)["image"]

    # Лёгкий наклон и масштаб (белый фон при заполнении)
    if random.random() < 0.3 * strength:
        img = A.Affine(
            rotate=(-2, 2),
            scale=(0.95, 1.05),
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
) -> torch.Tensor | None:
    img = load_image(image_path)
    if img is None:
        return None

    img = crop_to_content(img)

    if augment:
        img = apply_augmentations(
            img, dataset_type, elastic_p, elastic_alpha, elastic_sigma, strength
        )

    img = resize_preserve_aspect(img, target_h, max_w)
    img = binarize(img)
    return to_tensor(img)
