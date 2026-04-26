import os
import sys
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.tokenizer import LaTeXTokenizer
from data.preprocess import (
    load_image, crop_to_content, binarize,
    resize_preserve_aspect, apply_augmentations,
)
from config import load_config

# ──────────────────────────────────────────────
N_IMAGES = 10   # количество тестовых sample_*.png в выходной папке
# ──────────────────────────────────────────────

DATA_DIR = r"d:\notes2latex-ocr\data_raw"
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

def test_tokenizer(data_dir: str) -> LaTeXTokenizer:
    print("=== 1. Токенизатор ===")
    formulas_path = os.path.join(data_dir, "im2latex_formulas.lst")
    with open(formulas_path, encoding="latin-1", newline="\n") as f:
        formulas = [next(f).replace("\r", "").strip() for _ in range(2000)]

    tok = LaTeXTokenizer()
    tok.build_vocab(formulas, min_freq=2)
    print(f"  словарь: {tok.vocab_size} токенов (на 2000 формул)")

    samples = [
        (r"\frac{1}{2} + \alpha",                         "LaTeX"),
        (r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}", "LaTeX"),
        # Кириллица: FAIL ожидаем — vocab собран только из im2latex (без синтетики)
        ("Лемма 1. Пусть f непрерывна на [a, b]",         "Cyrillic (ожидаем FAIL без синтетики)"),
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
    ][:N_IMAGES]

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
        ("aug_1_original.png",   "original",                  img, "im2latex",    0.0,  0,  0, 0.0),
        ("aug_2_warmup.png",     "warmup  (no aug, str=0.7)", img, "im2latex",    0.0,  0,  0, 0.7),
        ("aug_3_im2latex.png",   "im2latex elastic (a=60)",   img, "im2latex",    1.0, 60,  8, 1.0),
        ("aug_4_synthetic.png",  "synthetic elastic (a=120)",  img, "synthetic",   1.0,120, 10, 1.0),
        ("aug_5_handwritten.png","handwritten (no elastic)",   img, "handwritten", 1.0,120, 10, 1.0),
    ]

    for fname, label, src, dtype, ep, ea, es, st in cases:
        out = src if ep == 0.0 and st == 0.0 else apply_augmentations(src.copy(), dtype, ep, ea, es, st)
        _save_img(out, label, os.path.join(TEST_DIR, fname))
        print(f"  {fname}")


def test_dataloader(data_dir: str, tok: LaTeXTokenizer):
    print("\n=== 4. DataLoader ===")
    config = load_config()

    from data.dataset import RawIm2LatexDataset, build_dataloaders

    # Считаем число батчей через даталоадер
    train_loader, _, _ = build_dataloaders(
        data_dir=data_dir,
        tokenizer=tok,
        batch_size=4,
        max_len=config.tokenizer_max_len,
        num_workers=0,
    )
    print(f"  батчей в train: {len(train_loader)}")

    # Для сохранения sample-изображений идём по датасету напрямую —
    # только так можно получить имя исходного файла
    dataset = RawIm2LatexDataset(
        data_dir, split="train",
        target_h=config.target_height,
        target_w=config.max_width,
    )

    for idx in range(min(N_IMAGES, len(dataset))):
        image_path, formula_idx = dataset.samples[idx]
        filename = os.path.basename(image_path)
        formula = dataset.formulas[formula_idx]

        img_tensor, _ = dataset[idx]
        img_np = img_tensor[0].numpy() * 0.5 + 0.5   # [-1,1] → [0,1]

        formula_wrapped = textwrap.fill(formula, width=100)
        title = f"{filename}\n{formula_wrapped}"

        save_path = os.path.join(TEST_DIR, f"sample_{idx + 1:02d}.png")
        _save_img(img_np, title, save_path, vmax=1)

    print(f"  сохранено {min(N_IMAGES, len(dataset))} изображений в {TEST_DIR}/")


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
    print("\nВсе проверки завершены.")


if __name__ == "__main__":
    main()
