"""Визуальная проверка preprocess pipeline'а в боевых условиях.

Что показывает:
  - Картинку ТАК, как её увидит модель: после crop_to_content +
    resize_preserve_aspect (с правильным аспектом и паддингом).
  - Тот же сэмпл с применёнными аугментациями (elastic + grid_aug +
    dilate/erode + noise/blur/affine).

Что НЕ показывает: LaTeX-подписи, чтобы не загромождать кадр.

Использование:
    python test_pipeline.py                         # все три датасета по 10
    python test_pipeline.py --n 5                   # 5 на датасет
    python test_pipeline.py --skip handwritten      # пропустить датасет
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import load_config
from data.preprocess import (
    apply_augmentations, crop_to_content, load_image,
    resize_preserve_aspect,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEST_DIR = Path(__file__).parent / "test"


# ──────────────────────────────────────────────────────────────────────────────
# Источники сэмплов
# ──────────────────────────────────────────────────────────────────────────────

def _im2latex_samples(config, n: int, seed: int) -> list[Path]:
    """Берёт N случайных сырых PNG из data_raw/formula_images."""
    raw_dir = Path(config.data_dir) / "formula_images"
    if not raw_dir.exists():
        return []
    files = [p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files)))


def _synthetic_samples(config, n: int, seed: int) -> list[Path]:
    """Берёт N случайных PNG из data_synthetic/images."""
    images_dir = Path(config.synthetic_dir) / "images"
    if not images_dir.exists():
        return []
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files)))


def _handwritten_samples(config, n: int, seed: int) -> list[Path]:
    """Берёт N случайных line crops из my_dataset/line_crops/crops/."""
    crops_dir = Path(config.my_dataset_dir) / "line_crops" / "crops"
    if not crops_dir.exists():
        return []
    files = [p for p in crops_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    rng = random.Random(seed)
    return rng.sample(files, min(n, len(files)))


# ──────────────────────────────────────────────────────────────────────────────
# Обработка одного сэмпла
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_clean(img_path: Path, target_h: int, max_w: int) -> np.ndarray | None:
    """Чистый preprocess: crop_to_content + resize_preserve_aspect. БЕЗ augmentation."""
    img = load_image(str(img_path))
    if img is None:
        return None
    img = crop_to_content(img)
    img = resize_preserve_aspect(img, target_h, max_w)
    return img


def _preprocess_aug(img: np.ndarray, dataset_type: str, config,
                    seed: int) -> np.ndarray:
    """Применяет full augmentation pipeline как в stage 2 train'е."""
    random.seed(seed)
    np.random.seed(seed)

    # Эмулируем "середину stage 2" — все аугментации на полную
    elastic_factor_map = {
        "im2latex":    config.elastic_factor_im2latex,
        "synthetic":   config.elastic_factor_synthetic,
        "handwritten": config.elastic_factor_handwritten,
    }
    grid_factor_map = {
        "im2latex":    config.grid_aug_factor_im2latex,
        "synthetic":   config.grid_aug_factor_synthetic,
        "handwritten": config.grid_aug_factor_handwritten,
    }

    return apply_augmentations(
        img.copy(),
        dataset_type=dataset_type,
        elastic_p=0.5, elastic_alpha=15, elastic_sigma=5,
        strength=0.5,
        elastic_factor=elastic_factor_map.get(dataset_type, 1.0),
        grid_aug_prob=config.grid_aug_prob * grid_factor_map.get(dataset_type, 1.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Сохранение пары clean+aug
# ──────────────────────────────────────────────────────────────────────────────

def _save_sample(clean: np.ndarray, aug: np.ndarray,
                 out_path: Path, name: str) -> None:
    """Сохраняет 2 строки (clean, aug) в один PNG без LaTeX-подписей."""
    w = clean.shape[1]
    # Ширина фигуры пропорциональна ширине картинки
    fig_w = max(6.0, min(22.0, w / 100.0))
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 3.2))

    axes[0].imshow(clean, cmap="gray", aspect="equal", vmin=0, vmax=255)
    axes[0].set_title(f"{name}  CLEAN  shape={clean.shape}",
                      fontsize=9, loc="left", family="monospace")
    axes[0].axis("off")

    axes[1].imshow(aug, cmap="gray", aspect="equal", vmin=0, vmax=255)
    axes[1].set_title("AUGMENTED (elastic + grid + dilate/erode + noise)",
                      fontsize=9, loc="left", family="monospace")
    axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Обработка датасета целиком
# ──────────────────────────────────────────────────────────────────────────────

def process_dataset(name: str, samples: list[Path], config, seed: int) -> None:
    out_dir = TEST_DIR / name
    # Чистим старые результаты
    if out_dir.exists():
        for f in out_dir.glob("*.png"):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        print(f"[{name}] нет сэмплов (источник пуст или не найден) — пропускаю.")
        return

    print(f"[{name}] обработка {len(samples)} сэмплов -> {out_dir}")
    target_h = config.target_height
    max_w    = config.max_width

    for i, p in enumerate(samples, 1):
        clean = _preprocess_clean(p, target_h, max_w)
        if clean is None:
            print(f"  [skip] не удалось загрузить: {p.name}")
            continue
        aug = _preprocess_aug(clean, dataset_type=name, config=config, seed=seed * 100 + i)
        out_path = out_dir / f"{i:03d}_{p.stem}.png"
        _save_sample(clean, aug, out_path, name=p.name)
        print(f"  {out_path.name}  ({clean.shape[1]}px wide)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Визуальная проверка preprocess + augmentation.")
    parser.add_argument("--n", type=int, default=10,
                        help="Сколько сэмплов на датасет (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["im2latex", "synthetic", "handwritten"],
                        help="Какие датасеты пропустить")
    parser.add_argument("--profile", default="rtx4060_8gb")
    args = parser.parse_args()

    config = load_config(args.profile)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Config:        target_h={config.target_height}  max_w={config.max_width}")
    print(f"Grid aug:      prob={config.grid_aug_prob}  factors: "
          f"i={config.grid_aug_factor_im2latex} "
          f"s={config.grid_aug_factor_synthetic} "
          f"h={config.grid_aug_factor_handwritten}")
    print(f"Output:        {TEST_DIR}")
    print()

    if "im2latex" not in args.skip:
        process_dataset("im2latex",
                        _im2latex_samples(config, args.n, args.seed),
                        config, args.seed)
    if "synthetic" not in args.skip:
        process_dataset("synthetic",
                        _synthetic_samples(config, args.n, args.seed),
                        config, args.seed)
    if "handwritten" not in args.skip:
        process_dataset("handwritten",
                        _handwritten_samples(config, args.n, args.seed),
                        config, args.seed)

    print("\nГотово.")


if __name__ == "__main__":
    main()
