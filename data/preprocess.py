"""
Предобработка изображений формул для подачи в нейросеть.

Pipeline:
    1. Загрузка в grayscale
    2. Обрезка по содержимому
    3. Бинаризация (адаптивная или по порогу)
    4. Resize до фиксированного размера с сохранением пропорций (pad белым цветом)
    5. Нормализация в тензор PyTorch

Выходной формат: torch.Tensor shape [1, H, W] (по умолчанию [1, 64, 512]).
"""
import os

import cv2
import numpy as np
import torch
import random


# --- Константы по умолчанию ---
TARGET_HEIGHT = 128
TARGET_WIDTH = 1024  # Максимальный лимит, а не фиксированный размер
PAD_VALUE = 255  # Считаем белый за фон


def load_image(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


def crop_to_content(img: np.ndarray, threshold: int = 250, padding: int = 8) -> np.ndarray:
    mask = img < threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        # Пустое изображение — вернуть как есть
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Добавляем отступ, ограничиваясь размерами изображения
    h, w = img.shape
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding + 1)

    return img[rmin:rmax, cmin:cmax]


def binarize(img: np.ndarray) -> np.ndarray:
    unique_count = len(np.unique(img))

    if unique_count <= 2:
        # Уже бинарное — ничего не делаем
        return img

    # Адаптивная бинаризация для фотографий
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=5,
    )

    # Медианный фильтр — убирает мелкий шум
    binary = cv2.medianBlur(binary, 3)

    return binary


def resize_preserve_aspect(
    img: np.ndarray,
    target_h: int = TARGET_HEIGHT,
    max_w: int = TARGET_WIDTH,
) -> np.ndarray:
    h, w = img.shape

    # Масштаб по высоте
    scale = target_h / h
    new_w = max(1, int(w * scale))

    # Защита от бесконечно широких строк
    if new_w > max_w:
        new_w = max_w

    # Resize
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    return resized

def apply_augmentations(img: np.ndarray) -> np.ndarray:
    # Изменение толщины "чернил"
    if random.random() < 0.4:
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() < 0.5:
            # Истончение линий (текст черный, фон белый -> dilate расширяет фон, истончая текст)
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            # Утолщение линий (erode сворачивает белый фон, утолщая черный текст)
            img = cv2.erode(img, kernel, iterations=1)
            
    # Шум камеры
    if random.random() < 0.3:
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    # Расфокус
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
        
    return img

def to_tensor(img: np.ndarray) -> torch.Tensor:
    # uint8 [0, 255] → float [0.0, 1.0]
    tensor = torch.from_numpy(img).float() / 255.0

    # Нормализация: (x - 0.5) / 0.5 → диапазон [-1, 1]
    tensor = (tensor - 0.5) / 0.5

    # Добавляем канальное измерение: (H, W) → (1, H, W)
    tensor = tensor.unsqueeze(0)

    return tensor



def preprocess_image(
    image_path: str,
    target_h: int = TARGET_HEIGHT,
    max_w: int = TARGET_WIDTH,
    augment: bool = False,
) -> torch.Tensor | None:
    """Полный пайплайн: загрузка -> обрезка -> [аугментация] -> бинаризация -> ресайз -> тензор."""
    img = load_image(image_path)
    if img is None:
        return None

    img = crop_to_content(img)
    
    if augment:
        img = apply_augmentations(img)
        
    img = binarize(img)
    img = resize_preserve_aspect(img, target_h, max_w=max_w)
    tensor = to_tensor(img)

    return tensor
