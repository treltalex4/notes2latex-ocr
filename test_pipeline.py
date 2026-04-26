import os
import sys
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

import json

from data.tokenizer import LaTeXTokenizer
from data.preprocess import (
    load_image, crop_to_content, binarize,
    resize_preserve_aspect, apply_augmentations, preprocess_image,
)
from config import load_config

# ──────────────────────────────────────────────
N_IMAGES_IM2LATEX  = 10   # sample_*.png из im2latex
N_IMAGES_SYNTHETIC =  5   # synthetic_*.png из сгенерированного датасета
# ──────────────────────────────────────────────

DATA_DIR      = r"d:\notes2latex-ocr\data_raw"
SYNTHETIC_DIR = r"d:\notes2latex-ocr\data_synthetic"
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
os.makedirs(TEST_DIR, exist_ok=True)


def _save_img(img_np, title: str, path: str, vmax=255):
    h, w = img_np.shape[:2]
    fig_w = max(6.0, w / 30)
    fig_h = max(2.0, h / 30) + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9, family="monospace", pad=5)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────
# Минимальный датасет, читающий сырые PNG без кэша — только для тестов
# ──────────────────────────────────────────────

class _RawIm2LatexDataset(Dataset):
    """Reads directly from raw PNG files. Used only in test_pipeline.py."""
    dataset_type = "im2latex"

    def __init__(self, data_dir: str, split: str, target_h: int, target_w: int) -> None:
        self.target_h = target_h
        self.target_w = target_w

        formulas_path = os.path.join(data_dir, "im2latex_formulas.lst")
        with open(formulas_path, encoding="latin-1", newline="\n") as f:
            self.formulas = [line.replace("\r", "").strip() for line in f]

        split_path = os.path.join(data_dir, f"im2latex_{split}.lst")
        self.samples: list[tuple[str, int]] = []
        with open(split_path, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                formula_idx = int(parts[0])
                image_name = parts[1]
                image_path = os.path.join(data_dir, "formula_images", image_name + ".png")
                if formula_idx < len(self.formulas) and self.formulas[formula_idx]:
                    self.samples.append((image_path, formula_idx))

        self.lengths = [len(self.formulas[idx]) for _, idx in self.samples]
        self.max_length = 512

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path, formula_idx = self.samples[idx]
        formula = self.formulas[formula_idx]
        image = preprocess_image(image_path, self.target_h, self.target_w, augment=False)
        if image is None:
            image = torch.zeros(1, self.target_h, 32)
        return image, formula


# ──────────────────────────────────────────────

def test_tokenizer(data_dir: str) -> LaTeXTokenizer:
    print("=== 1. Токенизатор ===")
    formulas_path = os.path.join(data_dir, "im2latex_formulas.lst")
    with open(formulas_path, encoding="latin-1", newline="\n") as f:
        formulas = [next(f).replace("\r", "").strip() for _ in range(2000)]

    tok = LaTeXTokenizer()
    tok.build_vocab(formulas, min_freq=2)
    print(f"  словарь: {tok.vocab_size} токенов (на 2000 формул)")

    samples = [
        (r"\frac{1}{2} + \alpha",                             "LaTeX"),
        (r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}", "LaTeX"),
        # Кириллица: FAIL ожидаем — vocab собран только из im2latex (без синтетики)
        ("Лемма 1. Пусть f непрерывна на [a, b]", "Cyrillic (ожидаем FAIL без синтетики)"),
    ]
    for s, note in samples:
        decoded = tok.decode(tok.encode(s))
        ok = decoded == s
        print(f"  {'OK  ' if ok else 'FAIL'} [{note}]  {repr(s[:50])}")
    return tok


def test_preprocess(data_dir: str) -> list[str]:
    print("\n=== 2. Предобработка ===")
    images_dir = os.path.join(data_dir, "formula_images")
    all_files = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.endswith(".png")
    ][:N_IMAGES_IM2LATEX]

    config = load_config()
    ok_count = 0
    for path in all_files:
        img = load_image(path)
        if img is None:
            print(f"  SKIP (broken): {os.path.basename(path)}")
            continue
        img = crop_to_content(img)
        img = resize_preserve_aspect(img, config.target_height, config.max_width)
        img = binarize(img)
        ok_count += 1

    print(f"  обработано {ok_count}/{len(all_files)} без ошибок")
    return all_files


def test_augmentations(image_path: str):
    print("\n=== 3. Аугментации ===")
    config = load_config()
    img = load_image(image_path)
    if img is None:
        print("  SKIP: не удалось загрузить изображение")
        return

    img = crop_to_content(img)
    img = resize_preserve_aspect(img, config.target_height, config.max_width)
    img = binarize(img)

    # p=1.0 гарантирует применение; alpha увеличен для наглядности
    cases = [
        ("aug_1_original.png",    "original",                  img, "im2latex",    0.0,  0,  0, 0.0),
        ("aug_2_warmup.png",      "warmup  (no aug, str=0.7)", img, "im2latex",    0.0,  0,  0, 0.7),
        ("aug_3_im2latex.png",    "im2latex elastic (a=60)",   img, "im2latex",    1.0, 60,  8, 1.0),
        ("aug_4_synthetic.png",   "synthetic elastic (a=120)", img, "synthetic",   1.0,120, 10, 1.0),
        ("aug_5_handwritten.png", "handwritten (no elastic)",  img, "handwritten", 1.0,120, 10, 1.0),
    ]

    for fname, label, src, dtype, ep, ea, es, st in cases:
        out = src if ep == 0.0 and st == 0.0 else apply_augmentations(src.copy(), dtype, ep, ea, es, st)
        _save_img(out, label, os.path.join(TEST_DIR, fname))
        print(f"  {fname}")


def test_dataloader(data_dir: str, tok: LaTeXTokenizer):
    print("\n=== 4. DataLoader ===")
    config = load_config()

    from data.dataset import CollateFunction, BucketBatchSampler

    dataset = _RawIm2LatexDataset(
        data_dir, "train",
        target_h=config.target_height,
        target_w=config.max_width,
    )
    collate_fn = CollateFunction(tok, max_len=config.tokenizer_max_len)
    train_loader = DataLoader(
        dataset,
        batch_sampler=BucketBatchSampler(dataset, base_batch_size=4, shuffle=True),
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"  батчей в train: {len(train_loader)}")

    for idx in range(min(N_IMAGES_IM2LATEX, len(dataset))):
        image_path, formula_idx = dataset.samples[idx]
        filename = os.path.basename(image_path)
        formula = dataset.formulas[formula_idx]

        img_tensor, _ = dataset[idx]
        img_np = img_tensor[0].numpy() * 0.5 + 0.5   # [-1,1] → [0,1]

        formula_wrapped = textwrap.fill(formula, width=100)
        title = f"{filename}\n{formula_wrapped}"

        save_path = os.path.join(TEST_DIR, f"sample_{idx + 1:02d}.png")
        _save_img(img_np, title, save_path, vmax=1)

    print(f"  сохранено {min(N_IMAGES_IM2LATEX, len(dataset))} изображений в {TEST_DIR}/")


def test_synthetic_samples(synthetic_dir: str):
    print("\n=== 5. Синтетический датасет ===")
    labels_path = os.path.join(synthetic_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"  SKIP: {labels_path} не найден.")
        print(f"  Запустите: python generate_synthetic.py --count 200")
        return

    with open(labels_path, encoding="utf-8") as f:
        labels: dict[str, str] = json.load(f)

    config = load_config()
    samples = list(labels.items())[:N_IMAGES_SYNTHETIC]
    saved = 0

    for i, (fname, formula) in enumerate(samples, 1):
        image_path = os.path.join(synthetic_dir, "images", fname)
        img = preprocess_image(image_path, config.target_height, config.max_width, augment=False)
        if img is None:
            print(f"  SKIP (broken): {fname}")
            continue

        img_np = img[0].numpy() * 0.5 + 0.5   # [-1,1] → [0,1]
        formula_wrapped = textwrap.fill(formula, width=100)
        title = f"{fname}\n{formula_wrapped}"

        save_path = os.path.join(TEST_DIR, f"synthetic_{i:02d}.png")
        _save_img(img_np, title, save_path, vmax=1)
        print(f"  synthetic_{i:02d}.png")
        saved += 1

    print(f"  сохранено {saved} изображений из синтетики в {TEST_DIR}/")


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"Датасет не найден: {DATA_DIR}")
        print("Скачайте im2latex-100k и распакуйте в data_raw/")
        sys.exit(1)

    tok = test_tokenizer(DATA_DIR)
    image_files = test_preprocess(DATA_DIR)
    if image_files:
        test_augmentations(image_files[0])
    test_dataloader(DATA_DIR, tok)
    test_synthetic_samples(SYNTHETIC_DIR)
    print("\nВсе проверки завершены.")


if __name__ == "__main__":
    main()
