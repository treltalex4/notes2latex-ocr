"""Оценка обученной модели на test split.

Использование:
    # Базово (beam search, метрики в stdout):
    python evaluate.py --checkpoint checkpoints/best_pretrain.pth --stage 1

    # Greedy (быстро, для регулярных оценок):
    python evaluate.py --checkpoint ... --greedy

    # Сравнить decoder'ы (одноразово, для понимания где упирается модель):
    python evaluate.py --checkpoint ... --compare-greedy

    # Сохранить worst-N худших предсказаний для ручного анализа:
    python evaluate.py --checkpoint ... --save-worst 30

    # Полный набор для финального отчёта:
    python evaluate.py --checkpoint ... --compare-greedy --save-worst 30 \\
        --save-predictions preds.json --plot-history
"""
import argparse
import glob
import json
import os
import random
import re
import time

import torch
from tqdm import tqdm

from config import load_config
from data.dataset import build_multi_dataloaders
from data.tokenizer import LaTeXTokenizer
from model.model import Notes2LaTeX, count_parameters
from utils.beam_search import beam_search_batch
from utils.metrics import (
    bleu_score, character_error_rate, edit_distance_score, exact_match,
)
from utils.visualization import plot_learning_curves, show_predictions
from train import greedy_decode_batch   # KV-cached декод — единый источник правды


@torch.no_grad()
def run_evaluation(model, loader, tokenizer, config, device,
                   limit_batches=None, use_greedy=False, compare_greedy=False):
    """Прогоняет loader через модель, возвращает предсказания.

    use_greedy=True: только greedy decode.
    compare_greedy=True: и greedy, и beam (одноразовое сравнение).
    иначе: только beam.

    Возвращает dict с ключами 'greedy', 'beam', 'references' (значения — list[str]).
    Пустой list если decoder не запускался.
    """
    model.eval()
    refs:   list[str] = []
    greedy: list[str] = []
    beam:   list[str] = []

    run_greedy = use_greedy or compare_greedy
    run_beam   = (not use_greedy) or compare_greedy

    pbar = tqdm(loader, total=limit_batches or len(loader), desc="eval")
    for batch_idx, (images, src_kpm, tgt_ids) in enumerate(pbar):
        images  = images.to(device)
        src_kpm = src_kpm.to(device)
        tgt_ids = tgt_ids.to(device)

        refs.extend(tokenizer.decode(ids.tolist()) for ids in tgt_ids)

        if run_greedy:
            greedy.extend(greedy_decode_batch(
                model, images, src_kpm, tokenizer, device,
                max_len=config.beam_max_len,
            ))
        if run_beam:
            beam.extend(beam_search_batch(
                model, images, src_kpm, tokenizer, config,
            ))

        if limit_batches and batch_idx + 1 >= limit_batches:
            break

    return {"greedy": greedy, "beam": beam, "references": refs}


def compute_all_metrics(predictions: list[str], references: list[str]) -> dict:
    return {
        "n_samples":            len(predictions),
        "exact_match":          exact_match(predictions, references),
        "character_error_rate": character_error_rate(predictions, references),
        "edit_distance_score":  edit_distance_score(predictions, references),
        "bleu_score":           bleu_score(predictions, references),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX normalization для honest-метрик
# ──────────────────────────────────────────────────────────────────────────────
#
# На шумных датасетах (im2latex) модель часто учится «лучше датасета»:
# нормализует `\label{...}`, упрощает скобки, унифицирует whitespace. Сырые
# метрики EM/BLEU/CER эту нормализацию НАКАЗЫВАЮТ — pred визуально совпадает с
# GT после рендера, но текст отличается → EM=0.
#
# normalize_latex приводит обе строки (pred и GT) к канонической форме перед
# сравнением. Это даёт «честные» метрики, отражающие реальное качество модели.
# Раз­ница между raw и normalized показывает «насколько метрика искажена шумом».

# Команды чисто для нумерации/оформления, без семантики формулы.
_LABEL_TAG_RE   = re.compile(r'\\(label|tag)\s*\{[^}]*\}')
# Бесаргументные оформительские команды (intsapce, не влияют на смысл).
_NOOP_CMD_RE    = re.compile(r'\\(nonumber|notag)\b')
# Spacing-команды LaTeX. Не критично для семантики (разный whitespace допустим).
_SPACE_CMD_RE   = re.compile(r'\\(quad|qquad|;|:|,|!)\b|\\[,;:!]')
# Множественные whitespace → один пробел.
_MULTI_WS_RE    = re.compile(r'\s+')
# Whitespace вокруг служебных символов (внутри `{a}^{2}` не нужны пробелы).
_PUNCT_WS_RE    = re.compile(r'\s*([{}^_])\s*')


def normalize_latex(s: str) -> str:
    """Канонизирует LaTeX-формулу для честных метрик.

    Убирает: `\\label{...}`, `\\tag{...}`, `\\nonumber`, `\\notag`,
             spacing-команды (`\\quad`, `\\,`, `\\!` и т.п.).
    Унифицирует: `\\lbrace`/`\\rbrace` → `{`/`}`, `\\lbrack`/`\\rbrack` → `[`/`]`,
                 множественные пробелы → один, пробелы вокруг `{}^_` — убраны.
    """
    s = _LABEL_TAG_RE.sub('', s)
    s = _NOOP_CMD_RE.sub('', s)
    s = _SPACE_CMD_RE.sub(' ', s)
    # Псевдонимы скобок (часто разный стиль в im2latex vs модель).
    s = s.replace(r'\lbrace', '{').replace(r'\rbrace', '}')
    s = s.replace(r'\lbrack', '[').replace(r'\rbrack', ']')
    s = _MULTI_WS_RE.sub(' ', s).strip()
    s = _PUNCT_WS_RE.sub(r'\1', s)
    return s


def compute_metrics_by_mode(predictions: list[str], references: list[str],
                            modes: list[str]) -> dict:
    """Возвращает {mode: metrics_dict} для запрошенных режимов из {raw, normalized}."""
    out: dict = {}
    if "raw" in modes:
        out["raw"] = compute_all_metrics(predictions, references)
    if "normalized" in modes:
        np_preds = [normalize_latex(p) for p in predictions]
        np_refs  = [normalize_latex(r) for r in references]
        out["normalized"] = compute_all_metrics(np_preds, np_refs)
    return out


_METRIC_LABELS = [
    ("exact_match",          "Exact Match"),
    ("character_error_rate", "Character Error Rate"),
    ("edit_distance_score",  "Edit Distance Score"),
    ("bleu_score",           "BLEU-4"),
]


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"\n=== {name} ({metrics['n_samples']} samples) ===")
    print(f"  Exact Match:           {metrics['exact_match']:.4f}")
    print(f"  Character Error Rate:  {metrics['character_error_rate']:.4f}")
    print(f"  Edit Distance Score:   {metrics['edit_distance_score']:.4f}  (1=identical)")
    print(f"  BLEU-4:                {metrics['bleu_score']:.4f}")


def _print_raw_vs_normalized(decoder_name: str, metrics_by_mode: dict) -> None:
    """Side-by-side таблица raw vs normalized для одного decoder'а.

    Показывает «насколько шум в датасете занижает метрики». Большая Δ → модель
    «умнее датасета», нормализованные метрики ближе к реальному качеству.
    """
    raw, norm = metrics_by_mode["raw"], metrics_by_mode["normalized"]
    print(f"\n=== {decoder_name} ({raw['n_samples']} samples) — raw vs normalized ===")
    print(f"  {'metric':<25}{'raw':>10}{'normalized':>14}{'Δ':>10}")
    print(f"  {'-' * 59}")
    for key, label in _METRIC_LABELS:
        r, n = raw[key], norm[key]
        delta = n - r
        # CER — "lower is better", инвертируем стрелку.
        arrow = "↑" if (delta > 0) ^ (key == "character_error_rate") else "↓"
        print(f"  {label:<25}{r:>10.4f}{n:>14.4f}{delta:>+9.4f}{arrow}")
    print()


def _print_comparison(greedy_metrics: dict, beam_metrics: dict,
                      mode_label: str = "") -> None:
    """Side-by-side таблица greedy vs beam с дельтами."""
    suffix = f" ({mode_label})" if mode_label else ""
    print(f"\n=== Greedy vs Beam comparison{suffix} ===")
    print(f"  {'metric':<25}{'greedy':>10}{'beam':>10}{'Δ':>10}")
    print(f"  {'-' * 55}")
    for key, label in _METRIC_LABELS:
        g, b = greedy_metrics[key], beam_metrics[key]
        delta = b - g
        # CER — "lower is better", инвертируем стрелку.
        arrow = "↑" if (delta > 0) ^ (key == "character_error_rate") else "↓"
        print(f"  {label:<25}{g:>10.4f}{b:>10.4f}{delta:>+9.4f}{arrow}")
    print()


def _levenshtein(s1: str, s2: str) -> int:
    """Простой Levenshtein per-pair для ранжирования worst predictions.
    O(len(s1) × len(s2)), для типичных формул (<200 символов) — миллисекунды."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,           # insert
                curr[j] + 1,                # delete
                prev[j] + (c1 != c2),       # substitute
            ))
        prev = curr
    return prev[-1]


def _collect_worst_and_random(predictions: list[str], references: list[str],
                              n_worst: int, n_random_correct: int) -> dict:
    """N худших по edit_distance + N случайных правильных (EM=1).

    Возвращает dict с двумя списками: worst и random_correct.
    Каждый элемент — {index, prediction, reference, edit_distance, length_diff}.
    """
    # Считаем edit_distance для каждой пары (тяжело только если их 10k+).
    scored = []
    correct_indices = []
    for i, (p, r) in enumerate(zip(predictions, references)):
        dist = _levenshtein(p, r)
        scored.append((i, dist, p, r))
        if p == r:
            correct_indices.append(i)

    # Worst по edit_distance (ties — по разнице длин).
    scored.sort(key=lambda x: (x[1], abs(len(x[2]) - len(x[3]))), reverse=True)
    worst = [{
        "index": i, "prediction": p, "reference": r,
        "edit_distance": d, "length_diff": len(p) - len(r),
    } for i, d, p, r in scored[:n_worst]]

    # N случайных правильных — sanity check что EM=1.0 не только тривиальные.
    random.seed(42)
    sample_n = min(n_random_correct, len(correct_indices))
    correct_sample = random.sample(correct_indices, sample_n) if sample_n else []
    random_correct = [{
        "index": i, "prediction": predictions[i], "reference": references[i],
    } for i in correct_sample]

    return {"worst": worst, "random_correct": random_correct}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="rtx4060_8gb")
    parser.add_argument("--checkpoint", required=True,
                        help="путь к чекпоинту (.pth)")
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="на каком datasets-наборе оценивать (1=im2latex test)")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--greedy", action="store_true",
                        help="только greedy decode (быстро, для регулярных оценок)")
    parser.add_argument("--compare-greedy", action="store_true",
                        help="запустить и greedy, и beam, показать дельты. "
                             "В 2× медленнее, имеет смысл только для финального отчёта "
                             "/ понимания упирается ли модель в decoder.")
    parser.add_argument("--normalize", choices=["raw", "normalized", "both"],
                        default="both",
                        help="raw: метрики на сырых строках (literal pred vs GT). "
                             "normalized: после normalize_latex (убирает \\label, "
                             "spacing-команды, унифицирует скобки) — честнее на "
                             "шумных датасетах. both (default): показать оба "
                             "вместе с дельтой — видно насколько метрики искажены "
                             "шумом в GT.")
    parser.add_argument("--save-predictions", default=None,
                        help="JSON-файл куда сохранить ВСЕ pred/ref пары")
    parser.add_argument("--save-worst", type=int, default=0,
                        help="Сохранить топ-N худших предсказаний (+ N/3 случайных "
                             "правильных) в <output>_worst.json для ручного анализа. "
                             "0=skip. Рекомендуется 30 — даёт картину failure modes "
                             "не перегружая ручную проверку.")
    parser.add_argument("--plot-history", action="store_true",
                        help="нарисовать графики обучения из history_*.json")
    parser.add_argument("--show-predictions", type=int, default=0,
                        help="нарисовать N примеров с GT/PRED в PNG (0=skip)")

    # для слабого железа
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=32,
                        help="override val_batch_size. На ноуте 24, на сервере 96.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.batch_size is not None:     overrides["batch_size"]     = args.batch_size
    if args.val_batch_size is not None: overrides["val_batch_size"] = args.val_batch_size
    if args.num_workers is not None:    overrides["num_workers"]    = args.num_workers
    if args.max_width is not None:      overrides["max_width"]      = args.max_width

    config = load_config(args.profile, **overrides)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
    print(f"Параметров: {count_parameters(model):,}")

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  stage={state.get('stage_name', '?')} epoch={state.get('epoch', '?')}")
    if "val_loss" in state:
        print(f"  train val_loss={state['val_loss']:.4f} val_acc={state.get('val_acc', 0):.3f}")

    # Test loader. Для stage 1 build_multi_dataloaders возвращает test_loader,
    # для stage 2/3 — None, тогда оцениваем на val.
    _, val_loader, test_loader = build_multi_dataloaders(
        config, tokenizer, stage=args.stage,
    )
    loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "val"
    print(f"Evaluating on {split_name} split: {len(loader)} batches")

    # Decode mode label для логов.
    if args.compare_greedy:
        mode = f"greedy + beam_size={config.beam_size}"
    elif args.greedy:
        mode = "greedy"
    else:
        mode = f"beam_size={config.beam_size}"
    print(f"Decode: {mode}")

    t0 = time.time()
    outputs = run_evaluation(
        model, loader, tokenizer, config, device,
        limit_batches=args.limit_batches,
        use_greedy=args.greedy,
        compare_greedy=args.compare_greedy,
    )
    dt = time.time() - t0
    print(f"\nEvaluation finished in {dt:.1f}s")

    # --- Метрики ---
    # modes: список normalize-режимов которые надо посчитать.
    if args.normalize == "both":
        modes = ["raw", "normalized"]
    else:
        modes = [args.normalize]

    # Структура: {decoder_name: {mode: metrics_dict}}.
    # Decoder'ы которые не запускались — отсутствуют.
    metrics_by_decoder: dict[str, dict[str, dict]] = {}
    if outputs["greedy"]:
        metrics_by_decoder["greedy"] = compute_metrics_by_mode(
            outputs["greedy"], outputs["references"], modes,
        )
    if outputs["beam"]:
        metrics_by_decoder["beam"] = compute_metrics_by_mode(
            outputs["beam"], outputs["references"], modes,
        )

    beam_label = f"Beam (size={config.beam_size})"
    decoder_labels = {"greedy": "Greedy", "beam": beam_label}

    # Печатаем per-decoder. Если оба mode'а (both) — таблица raw vs normalized.
    # Иначе — обычный блок одной метрики.
    for dec_key, metrics_by_mode in metrics_by_decoder.items():
        dec_label = decoder_labels[dec_key]
        if "raw" in metrics_by_mode and "normalized" in metrics_by_mode:
            _print_raw_vs_normalized(dec_label, metrics_by_mode)
        else:
            single_mode = next(iter(metrics_by_mode))   # "raw" или "normalized"
            _print_metrics(f"{dec_label} ({single_mode})", metrics_by_mode[single_mode])

    # Greedy vs Beam comparison — отдельно для каждого normalize-режима,
    # потому что Δ greedy→beam может вести себя по-разному в raw и normalized.
    if "greedy" in metrics_by_decoder and "beam" in metrics_by_decoder:
        for mode in modes:
            _print_comparison(
                metrics_by_decoder["greedy"][mode],
                metrics_by_decoder["beam"][mode],
                mode_label=mode,
            )

    # Какой набор предсказаний использовать для save-predictions / save-worst.
    # Если запускали оба — beam (это «правильный» decoder). Иначе тот что запустили.
    primary_preds = outputs["beam"] or outputs["greedy"]

    # --- Save full predictions ---
    if args.save_predictions:
        out = {
            "checkpoint":   args.checkpoint,
            "stage":        args.stage,
            "split":        split_name,
            "decode_mode":  mode,
            "normalize":    args.normalize,
            "elapsed_s":    round(dt, 1),
            # metrics_by_decoder: {"greedy": {"raw": {...}, "normalized": {...}}, "beam": ...}
            "metrics":      metrics_by_decoder,
            "samples": [
                {"prediction": p, "reference": r}
                for p, r in zip(primary_preds, outputs["references"])
            ],
        }
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {args.save_predictions}")

    # --- Save worst-N + random correct (для анализа failure modes) ---
    if args.save_worst > 0:
        n_random = max(args.save_worst // 3, 5)   # +1/3 от worst, но не меньше 5
        bundle = _collect_worst_and_random(
            primary_preds, outputs["references"],
            n_worst=args.save_worst, n_random_correct=n_random,
        )
        # Имя файла: на основе save_predictions или checkpoint.
        if args.save_predictions:
            worst_path = args.save_predictions.replace(".json", "_worst.json")
        else:
            base = os.path.splitext(os.path.basename(args.checkpoint))[0]
            worst_path = os.path.join(os.path.dirname(args.checkpoint) or ".",
                                      f"{base}_worst.json")
        with open(worst_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint":     args.checkpoint,
                "decoder":        "beam" if outputs["beam"] else "greedy",
                "n_worst":        len(bundle["worst"]),
                "n_random_correct": len(bundle["random_correct"]),
                "worst":          bundle["worst"],
                "random_correct": bundle["random_correct"],
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved worst-{args.save_worst} + random correct to {worst_path}")
        # Печатаем первые 3 worst прямо в stdout для быстрого осмотра.
        print("\n--- Sample worst predictions ---")
        for entry in bundle["worst"][:3]:
            print(f"  [#{entry['index']} edit_dist={entry['edit_distance']}]")
            print(f"    REF:  {entry['reference'][:200]}")
            print(f"    PRED: {entry['prediction'][:200]}")

    # --- Plot history ---
    if args.plot_history:
        stage_name = state.get("stage_name", "pretrain")
        runs_dir = os.path.join(config.checkpoint_dir, "runs")
        candidates = sorted(
            glob.glob(os.path.join(runs_dir, "*.json")),
            key=os.path.getmtime,
            reverse=True,
        )
        history_path = None
        for path in candidates:
            try:
                with open(path, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("stage_name") == stage_name:
                    history_path = path
                    break
            except Exception:
                continue
        if history_path:
            print(f"plotting history: {history_path}")
            plot_learning_curves(history_path, config.plots_dir)
        else:
            print(f"history not found for stage '{stage_name}' in {runs_dir}/")

    # --- Show predictions PNG ---
    if args.show_predictions > 0:
        dataset = loader.dataset
        out_path = os.path.join(config.plots_dir, f"predictions_{split_name}.png")
        os.makedirs(config.plots_dir, exist_ok=True)
        show_predictions(model, dataset, tokenizer, device,
                         n=args.show_predictions, save_path=out_path)


if __name__ == "__main__":
    main()
