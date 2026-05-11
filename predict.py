"""Прогон обученной модели на тестовых строках с визуализацией.

Использование:
    # По дефолту: папка test_lines/ + последний best чекпоинт
    python predict.py --checkpoint checkpoints/best_pretrain.pth

    # Своя папка
    python predict.py --checkpoint ... --input my_dataset/line_crops/crops

    # Готовые сэмплы из кэша im2latex (с GT для сравнения)
    python predict.py --checkpoint ... --mode cached --dataset im2latex --n 10

    # Из synthetic
    python predict.py --checkpoint ... --mode cached --dataset synthetic --n 10

Результат — PNG-файл, где для каждой строки показано: изображение слева,
предсказание модели снизу (и groundtruth в режиме cached).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import load_config
from data.preprocess import preprocess_image, to_tensor
from data.tokenizer import EOS_ID, LaTeXTokenizer, SOS_ID
from model.model import Notes2LaTeX


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка входов: два режима
# ──────────────────────────────────────────────────────────────────────────────

def _load_folder(input_dir: Path, config, n: int | None) -> list[tuple[str, torch.Tensor, str | None]]:
    """Читает PNG из папки, прогоняет через preprocess. Возвращает [(name, tensor, gt_or_None)]."""
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"Нет изображений в {input_dir} (расширения: {IMAGE_EXTS})")
    if n is not None:
        files = files[:n]

    samples: list[tuple[str, torch.Tensor, str | None]] = []
    for path in files:
        tensor = preprocess_image(str(path), config.target_height, config.max_width)
        if tensor is None:
            print(f"  [skip] не удалось загрузить: {path.name}")
            continue
        samples.append((path.name, tensor, None))
    return samples


def _load_cached(dataset_name: str, config, n: int, seed: int) -> list[tuple[str, torch.Tensor, str | None]]:
    """Случайные n сэмплов из data_cache/<dataset>/manifest.json (с GT-формулой)."""
    manifest_path = Path(config.cache_dir) / dataset_name / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Кэш не найден: {manifest_path}\n"
            f"Запусти: python prepare_data.py --datasets {dataset_name}"
        )
    with manifest_path.open(encoding="utf-8") as f:
        entries = json.load(f)
    # train-сплит как наиболее представительный
    train_entries = [e for e in entries if e.get("split", "train") == "train"]
    rng = random.Random(seed)
    picked = rng.sample(train_entries, min(n, len(train_entries)))

    samples: list[tuple[str, torch.Tensor, str | None]] = []
    for e in picked:
        img = np.load(e["npy_path"])
        tensor = to_tensor(img)
        name = os.path.basename(e["npy_path"])
        samples.append((name, tensor, e["formula"]))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(model, tensor: torch.Tensor, tokenizer, device, max_len: int = 300) -> str:
    """Greedy decode одной картинки."""
    model.eval()
    image = tensor.unsqueeze(0).to(device)               # [1, 1, H, W]
    memory, memory_kpm = model.encoder(image)            # без src_kpm (одна картинка)

    generated = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model.decoder(generated, memory, memory_key_padding_mask=memory_kpm)
        next_id = logits[0, -1].argmax().item()
        generated = torch.cat(
            [generated, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1,
        )
        if next_id == EOS_ID:
            break
    return tokenizer.decode(generated[0].tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Визуализация
# ──────────────────────────────────────────────────────────────────────────────

def _render_one(name: str, tensor: torch.Tensor, gt: str | None,
                pred: str, save_path: Path) -> bool:
    """Сохраняет PNG для одного сэмпла. Возвращает True если pred совпал с gt
    (для подсчёта exact match по выборке)."""
    img_np = tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5  # [-1,1] → [0,1]
    w_px = img_np.shape[1]

    # Подгоняем размер фигуры под форму строки: широкие строки -> широкая figure
    fig_w = max(8.0, min(20.0, w_px / 80.0))
    fig, ax = plt.subplots(figsize=(fig_w, 2.6))
    ax.imshow(img_np, cmap="gray", aspect="auto", vmin=0, vmax=1)

    is_match = gt is not None and pred == gt
    if gt is not None:
        status = "OK  " if is_match else "DIFF"
        title = (f"[{status}] {name}\n"
                 f"  GT:   {gt}\n"
                 f"  PRED: {pred}")
    else:
        title = f"{name}\n  PRED: {pred}"

    ax.set_title(title, fontsize=9, loc="left", family="monospace")
    ax.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return is_match


def _render_all(samples_with_preds: list[tuple[str, torch.Tensor, str | None, str]],
                out_dir: Path) -> None:
    n = len(samples_with_preds)
    if n == 0:
        print("Нечего рендерить.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n_match = 0
    has_gt = samples_with_preds[0][2] is not None

    for i, (name, tensor, gt, pred) in enumerate(samples_with_preds, 1):
        # Имя файла: <NN>_<original_stem>.png — нумерация даёт стабильный порядок
        stem = Path(name).stem
        out_path = out_dir / f"{i:03d}_{stem}.png"
        n_match += int(_render_one(name, tensor, gt, pred, out_path))

    summary = f"Saved {n} predictions -> {out_dir}"
    if has_gt:
        summary += f"  (exact match: {n_match}/{n})"
    print(summary)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quick visual test модели на строках.")
    parser.add_argument("--checkpoint", required=True,
                        help="Путь к .pth (например checkpoints/best_pretrain.pth)")
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--profile", default="rtx4060_8gb")

    parser.add_argument("--mode", choices=["folder", "cached"], default="folder",
                        help="folder = PNG из --input, cached = сэмплы из data_cache")
    parser.add_argument("--input", default="test_lines",
                        help="Папка с PNG (для --mode folder)")
    parser.add_argument("--dataset", choices=["im2latex", "synthetic", "handwritten"],
                        default="im2latex",
                        help="Из какого кэша брать (для --mode cached)")
    parser.add_argument("--n", type=int, default=10,
                        help="Сколько сэмплов взять (для cached) или максимум из папки")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=300,
                        help="Максимум токенов при greedy decode")
    parser.add_argument("--output", default=None,
                        help="Папка для сохранения PNG (по одному файлу на строку). "
                             "По умолчанию: test_lines/results/ для folder-режима, "
                             "test_lines/results_<dataset>/ для cached.")

    parser.add_argument("--max-width", type=int, default=None,
                        help="Override config.max_width (для слабого железа)")

    args = parser.parse_args()

    overrides: dict = {}
    if args.max_width is not None:
        overrides["max_width"] = args.max_width
    config = load_config(args.profile, **overrides)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")

    # Tokenizer
    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab:      {tokenizer.vocab_size} tokens ({args.tokenizer})")

    # Model + checkpoint
    model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Checkpoint: {args.checkpoint}  "
          f"(stage={state.get('stage_name', '?')} epoch={state.get('epoch', '?')})")

    # Загрузка сэмплов
    if args.mode == "folder":
        input_dir = Path(args.input)
        if not input_dir.exists():
            raise FileNotFoundError(f"Папка не найдена: {input_dir}")
        samples = _load_folder(input_dir, config, n=args.n)
        default_out_dir = Path("test_lines") / "results"
        source_desc = f"folder: {input_dir} ({len(samples)} imgs)"
    else:
        samples = _load_cached(args.dataset, config, n=args.n, seed=args.seed)
        default_out_dir = Path("test_lines") / f"results_{args.dataset}"
        source_desc = f"cached: {args.dataset} ({len(samples)} imgs, seed={args.seed})"
    print(f"Source:     {source_desc}")

    if not samples:
        print("Нет сэмплов для прогона.")
        return

    # Inference
    print(f"Decoding {len(samples)} samples (greedy, max_len={args.max_len})...")
    samples_with_preds: list = []
    for name, tensor, gt in samples:
        pred = greedy_decode(model, tensor, tokenizer, device, max_len=args.max_len)
        samples_with_preds.append((name, tensor, gt, pred))
        print(f"  {name}")

    # Render — отдельный PNG на каждую строку, в общую папку
    out_dir = Path(args.output) if args.output else default_out_dir
    _render_all(samples_with_preds, out_dir)


if __name__ == "__main__":
    main()
