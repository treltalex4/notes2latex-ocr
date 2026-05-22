"""Прогон обученной модели + вертикально-стэковая визуализация.

Для каждого сэмпла рисуется 4-строчный коллаж (одна колонка на всю ширину):
  ┌─────────────────────────────────────────┐
  │ GT (LaTeX code)                          │
  ├─────────────────────────────────────────┤
  │ Original image (input)                   │
  ├─────────────────────────────────────────┤
  │ PRED (LaTeX code)                        │
  ├─────────────────────────────────────────┤
  │ Rendered PRED                            │
  └─────────────────────────────────────────┘

Картинки получают полную ширину фигуры, ширина и высота фигуры подбираются
динамически под реальные пропорции изображений (формулы im2latex часто 128×2500
— нужна широкая figure чтобы текст был читаемым).

Это даёт сравнить модель **визуально**: даже если PRED отличается от GT
буквально (другой порядок токенов, лишние пробелы), отрендеренный PNG может
выглядеть идентично — модель «правильна семантически».

Рендер PRED через pdflatex (требует TeX Live / MiKTeX в PATH). Если LaTeX
от нейронки битый — в правом нижнем углу пишется "[render failed]".

Использование:
    # Случайные сэмплы из кэша im2latex (с GT для сравнения):
    python predict.py --checkpoint checkpoints/best_pretrain.pth --mode cached --n 10

    # Из synthetic:
    python predict.py --checkpoint ... --mode cached --dataset synthetic --n 10

    # Свои PNG (без GT — только PRED + rendered):
    python predict.py --checkpoint ... --mode folder --input test_lines/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import load_config
from data.preprocess import crop_to_content, preprocess_image, to_tensor
from data.tokenizer import LaTeXTokenizer
from model.model import Notes2LaTeX
from utils.beam_search import beam_search
from train import greedy_decode_batch   # KV-cached greedy, единый источник правды

try:
    from pdf2image import convert_from_path
    _HAS_PDF2IMAGE = True
except ImportError:
    _HAS_PDF2IMAGE = False


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX rendering для PRED
# ──────────────────────────────────────────────────────────────────────────────

# Используем \[ ... \] (display math) вместо $...$: безопаснее, не ломается на
# одиночных $ или % в формулах. amsmath/amssymb/amsfonts — на случай редких
# команд из im2latex (\widetilde, \rm, \lbrace и т.п.).
_RENDER_DOC_PREFIX = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm,amsfonts}
\usepackage{geometry}
\geometry{paperwidth=55cm,paperheight=10cm,left=0.5cm,right=0.5cm,top=0.5cm,bottom=0.5cm}
\pagestyle{empty}
\parindent=0pt
\begin{document}
\noindent \["""

_RENDER_DOC_SUFFIX = r"""\]
\end{document}
"""


def _check_pdflatex() -> bool:
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def render_latex_formula(formula: str, dpi: int = 200) -> np.ndarray | None:
    """Рендерит LaTeX-формулу в grayscale uint8 image через pdflatex + pdf2image.
    Возвращает None если рендер не удался — нормально для битого LaTeX от нейронки."""
    if not _HAS_PDF2IMAGE:
        return None
    doc = _RENDER_DOC_PREFIX + formula + _RENDER_DOC_SUFFIX
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "f.tex")
        pdf_path = os.path.join(tmpdir, "f.pdf")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0 or not os.path.exists(pdf_path):
            return None
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, grayscale=True)
        except Exception:
            return None
        if not pages:
            return None
        img = np.array(pages[0])
        img = crop_to_content(img)
        return img if img.size > 0 else None


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка входов
# ──────────────────────────────────────────────────────────────────────────────

def _load_folder(input_dir: Path, config, n: int | None) -> list[tuple[str, torch.Tensor, str | None]]:
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


def _load_cached(dataset_name: str, split: str, config, n: int,
                 seed: int) -> list[tuple[str, torch.Tensor, str | None]]:
    """Случайные n сэмплов из data_cache/<dataset>/manifest.json.

    split: 'test' (default) / 'validate' / 'train'. Для визуальной оценки
    обученной модели правильно брать test/validate — модель их не видела
    при обучении, метрика честная. train — для отладки «модель запомнила?».
    """
    manifest_path = Path(config.cache_dir) / dataset_name / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Кэш не найден: {manifest_path}\n"
            f"Запусти: python prepare_data.py --datasets {dataset_name}"
        )
    with manifest_path.open(encoding="utf-8") as f:
        entries = json.load(f)
    # У synthetic нет splits (всё в train). Для im2latex есть train/validate/test.
    filtered = [e for e in entries if e.get("split", "train") == split]
    if not filtered:
        available = sorted({e.get("split", "train") for e in entries})
        raise ValueError(
            f"В кэше '{dataset_name}' нет сэмплов со split='{split}'. "
            f"Доступные splits: {available}"
        )
    rng = random.Random(seed)
    picked = rng.sample(filtered, min(n, len(filtered)))

    samples: list[tuple[str, torch.Tensor, str | None]] = []
    for e in picked:
        img = np.load(e["npy_path"])
        tensor = to_tensor(img)
        name = os.path.basename(e["npy_path"])
        samples.append((name, tensor, e["formula"]))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Inference — поддерживает оба decoder'а (KV-cached в обоих случаях)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_one(model, tensor: torch.Tensor, tokenizer, config, device,
               decoder: str = "beam") -> str:
    """decoder: 'beam' (по дефолту, лучше качество) или 'greedy' (быстрее в ~5×)."""
    model.eval()
    image = tensor.unsqueeze(0).to(device)   # [1, 1, H, W]
    # src_key_padding_mask=None: одна картинка после preprocess не имеет паддинга
    # (preprocess_image возвращает точный размер).
    if decoder == "greedy":
        # greedy_decode_batch возвращает list[str] для батча — берём [0].
        return greedy_decode_batch(
            model, image, src_kpm=None, tokenizer=tokenizer, device=device,
            max_len=config.beam_max_len,
        )[0]
    return beam_search(model, image, tokenizer, config, src_key_padding_mask=None)


# ──────────────────────────────────────────────────────────────────────────────
# Side-by-side рендеринг
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_displayable(tensor: torch.Tensor) -> np.ndarray:
    """Конвертирует [-1, 1] тензор модели в [0, 1] numpy для imshow."""
    img = tensor.squeeze(0).cpu().numpy()   # [H, W]
    return img * 0.5 + 0.5


def _show_image_ax(ax, img: np.ndarray | None, title: str,
                   fail_text: str = "[нет картинки]") -> None:
    if img is not None:
        # uint8 ([0, 255]) или float ([0, 1]) — определяем по dtype.
        vmax = 255 if img.dtype == np.uint8 else 1.0
        ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
    else:
        ax.text(0.5, 0.5, fail_text, ha="center", va="center",
                fontsize=11, family="monospace", color="gray",
                transform=ax.transAxes)
    ax.set_title(title, fontsize=9, family="monospace", loc="left")
    ax.axis("off")


def _show_text_ax(ax, text: str, title: str) -> None:
    """Длинный LaTeX-текст без рендера — для GT/PRED заголовков."""
    ax.set_title(title, fontsize=9, family="monospace", loc="left")
    ax.axis("off")
    # wrap=True переносит длинные формулы по словам автоматически.
    ax.text(0.0, 1.0, text, ha="left", va="top",
            fontsize=8, family="monospace", wrap=True,
            transform=ax.transAxes)


# Параметры динамического sizing. Подобраны под формулы im2latex (128×100..2800).
_FIG_MIN_W_IN     = 10.0   # минимум ширины figure (короткие формулы)
_FIG_MAX_W_IN     = 18.0   # максимум (защита от слишком огромных PNG)
_PX_PER_INCH      = 130    # ~ширина mononspace 8pt × dpi=100 — эмпирический коэф
_TEXT_TITLE_H_IN  = 0.4    # высота на заголовок + padding для текстовой строки
_TEXT_LINE_H_IN   = 0.18   # высота на одну wrapped-строку monospace 8pt
_FAIL_PLACEHOLDER = 1.2    # высота для placeholder'а когда render PRED не удался


def _estimate_text_height(text: str, fig_w_in: float) -> float:
    """Грубая оценка высоты ячейки с wrap'нутым monospace 8pt текстом.

    Цель — дать достаточно вертикального места длинным формулам (~200 символов
    обычно). Перестраховываемся в сторону больше, чем меньше: лишний whitespace
    лучше обрезанной формулы.
    """
    char_per_line = max(60, int(fig_w_in * 16))   # ~16 monospace chars/inch
    n_lines = max(1, (len(text) + char_per_line - 1) // char_per_line)
    return _TEXT_TITLE_H_IN + n_lines * _TEXT_LINE_H_IN


def _image_height_in(img: np.ndarray | None, fig_w_in: float) -> float:
    """Высота картинки в дюймах при отображении на full figure width."""
    if img is None:
        return _FAIL_PLACEHOLDER
    h_px, w_px = img.shape[:2]
    return fig_w_in * (h_px / max(w_px, 1))


def render_comparison(name: str, tensor: torch.Tensor, gt: str | None,
                      pred: str, rendered_pred: np.ndarray | None,
                      save_path: Path) -> bool:
    """PNG 4-row stack (GT-text, GT-image, PRED-text, PRED-image).

    Ширина figure масштабируется под наиболее широкое изображение, высоты
    строк выставляются пропорционально реальным пропорциям картинок и длине
    текста — формулы 128×2500 не сжимаются в нечитаемую полоску.

    Возвращает True если pred == gt (EM).
    """
    is_match = gt is not None and pred == gt
    status = "" if gt is None else ("OK  " if is_match else "DIFF")

    orig_img = _tensor_to_displayable(tensor)
    gt_text = gt if gt is not None else "(no GT — folder mode)"

    # 1. Динамическая ширина figure: берём max ширину pixels из orig и rendered
    # и масштабируем через _PX_PER_INCH. Cap'ы защищают от крайностей.
    max_img_w_px = max(
        orig_img.shape[1],
        rendered_pred.shape[1] if rendered_pred is not None else 0,
    )
    fig_w_in = max(_FIG_MIN_W_IN,
                   min(_FIG_MAX_W_IN, max_img_w_px / _PX_PER_INCH))

    # 2. Высоты строк — пропорционально реальным aspect ratios.
    gt_text_h_in   = _estimate_text_height(gt_text, fig_w_in)
    orig_img_h_in  = _image_height_in(orig_img, fig_w_in)
    pred_text_h_in = _estimate_text_height(pred, fig_w_in)
    rend_img_h_in  = _image_height_in(rendered_pred, fig_w_in)

    total_h_in = (gt_text_h_in + orig_img_h_in
                  + pred_text_h_in + rend_img_h_in + 0.5)  # +suptitle

    fig = plt.figure(figsize=(fig_w_in, total_h_in))
    gs = fig.add_gridspec(
        4, 1, hspace=0.4,
        height_ratios=[gt_text_h_in, orig_img_h_in,
                       pred_text_h_in, rend_img_h_in],
    )

    suptitle = f"[{status}] {name}" if status else name
    fig.suptitle(suptitle, fontsize=10, family="monospace")

    # Row 0: GT text
    _show_text_ax(fig.add_subplot(gs[0]), gt_text, "GT (LaTeX code)")
    # Row 1: GT image (orig input)
    _show_image_ax(fig.add_subplot(gs[1]), orig_img, "Original image (input)")
    # Row 2: PRED text
    _show_text_ax(fig.add_subplot(gs[2]), pred, "PRED (LaTeX code)")
    # Row 3: PRED rendered
    _show_image_ax(fig.add_subplot(gs[3]), rendered_pred, "PRED rendered",
                   fail_text="[render failed — невалидный LaTeX]")

    fig.tight_layout(rect=(0, 0, 1, 0.97))   # место под suptitle
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return is_match


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visual test модели: GT vs PRED side-by-side.")
    parser.add_argument("--checkpoint", required=True,
                        help="Путь к .pth (например checkpoints/best_pretrain.pth)")
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--profile", default="rtx4060_8gb")

    parser.add_argument("--mode", choices=["folder", "cached"], default="cached",
                        help="folder = PNG из --input, cached = сэмплы из data_cache (с GT)")
    parser.add_argument("--input", default="test_lines",
                        help="Папка с PNG (для --mode folder)")
    parser.add_argument("--dataset", choices=["im2latex", "synthetic", "handwritten", "unimer"],
                        default="im2latex",
                        help="Из какого кэша брать (для --mode cached)")
    parser.add_argument("--split", choices=["train", "validate", "test"], default="test",
                        help="Какой split брать из кэша. Default 'validate' — те же "
                             "примеры что видит val_em во время обучения, можно сверять. "
                             "'test' — финальный held-out. 'train' — для отладки "
                             "«модель запомнила?». У synthetic есть только 'train'.")
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="beam",
                        help="Decoder для генерации PRED. beam: лучше качество "
                             "(использует beam_size из config), greedy: быстрее в ~5×. "
                             "Для финального визуального отчёта — beam, для быстрых "
                             "проверок — greedy.")
    parser.add_argument("--n", type=int, default=10,
                        help="Сколько сэмплов взять")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-dpi", type=int, default=200,
                        help="DPI рендера PRED (выше — крупнее, медленнее)")
    parser.add_argument("--output", default=None,
                        help="Папка для PNG (по одному файлу на строку). "
                             "По умолчанию: test_lines/results_<dataset>/ или "
                             "test_lines/results/ для folder.")
    parser.add_argument("--max-width", type=int, default=None,
                        help="Override config.max_width (для слабого железа)")
    args = parser.parse_args()

    # Проверяем pdflatex заранее — без него правая нижняя ячейка будет всегда пустой.
    if not _check_pdflatex():
        print("WARNING: pdflatex не найден в PATH — PRED не будет рендериться. "
              "Установи TeX Live / MiKTeX чтобы получить визуальное сравнение.")
    if not _HAS_PDF2IMAGE:
        print("WARNING: pdf2image не установлен — PRED не будет рендериться. "
              "pip install pdf2image (требует poppler).")

    overrides: dict = {}
    if args.max_width is not None:
        overrides["max_width"] = args.max_width
    config = load_config(args.profile, **overrides)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")

    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab:      {tokenizer.vocab_size} tokens ({args.tokenizer})")

    model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Checkpoint: {args.checkpoint}  "
          f"(stage={state.get('stage_name', '?')} epoch={state.get('epoch', '?')})")

    # Загрузка сэмплов.
    if args.mode == "folder":
        input_dir = Path(args.input)
        if not input_dir.exists():
            raise FileNotFoundError(f"Папка не найдена: {input_dir}")
        samples = _load_folder(input_dir, config, n=args.n)
        default_out_dir = Path("test_lines") / "results"
        source_desc = f"folder: {input_dir} ({len(samples)} imgs)"
    else:
        samples = _load_cached(args.dataset, args.split, config, n=args.n, seed=args.seed)
        default_out_dir = Path("test_lines") / f"results_{args.dataset}_{args.split}"
        source_desc = (f"cached: {args.dataset} split={args.split} "
                       f"({len(samples)} imgs, seed={args.seed})")
    print(f"Source:     {source_desc}")
    decoder_desc = (f"beam_search (beam_size={config.beam_size})"
                    if args.decoder == "beam" else "greedy_decode")
    print(f"Decoder:    {decoder_desc}")

    if not samples:
        print("Нет сэмплов для прогона.")
        return

    out_dir = Path(args.output) if args.output else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inference + render + save — по одному сэмплу для прогресса в реальном времени.
    n_match = 0
    n_render_failed = 0
    has_gt = samples[0][2] is not None
    for i, (name, tensor, gt) in enumerate(samples, 1):
        pred = decode_one(model, tensor, tokenizer, config, device,
                          decoder=args.decoder)
        rendered = render_latex_formula(pred, dpi=args.render_dpi)
        if rendered is None:
            n_render_failed += 1

        stem = Path(name).stem
        out_path = out_dir / f"{i:03d}_{stem}.png"
        is_match = render_comparison(name, tensor, gt, pred, rendered, out_path)
        n_match += int(is_match)

        mark = "OK  " if is_match else ("DIFF" if has_gt else "    ")
        render_mark = "" if rendered is not None else "  (render failed)"
        print(f"  [{i:3d}/{len(samples)}] {mark} {name}{render_mark}")

    summary = f"Saved {len(samples)} predictions -> {out_dir}"
    if has_gt:
        summary += f"  (exact match: {n_match}/{len(samples)})"
    if n_render_failed:
        summary += f"  (PRED render failed: {n_render_failed}/{len(samples)})"
    print(summary)


if __name__ == "__main__":
    main()
