import argparse
import hashlib
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from config import load_config, Config
from data.preprocess import load_image, crop_to_content, binarize, resize_preserve_aspect
from data.tokenizer import LaTeXTokenizer


def _stable_hash(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Listing raw samples per dataset type
# ──────────────────────────────────────────────────────────────────────────────

def _list_im2latex(config: Config) -> list[dict]:
    formulas_path = os.path.join(config.data_dir, "im2latex_formulas.lst")
    if not os.path.exists(formulas_path):
        print(f"  [WARN] не найден: {formulas_path}")
        return []

    with open(formulas_path, encoding="latin-1", newline="\n") as f:
        formulas = [line.replace("\r", "").strip() for line in f]

    samples = []
    n_bad_idx = 0
    for split in ("train", "validate", "test"):
        split_path = os.path.join(config.data_dir, f"im2latex_{split}.lst")
        if not os.path.exists(split_path):
            print(f"  [WARN] не найден: {split_path}")
            continue
        with open(split_path, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                formula_idx = int(parts[0])
                image_name = parts[1]
                if formula_idx >= len(formulas) or not formulas[formula_idx]:
                    n_bad_idx += 1
                    continue
                samples.append({
                    "image_path": os.path.join(config.data_dir, "formula_images", image_name + ".png"),
                    "formula": formulas[formula_idx],
                    "split": split,
                })
    if n_bad_idx:
        print(f"  [WARN] im2latex: пропущено {n_bad_idx} записей с невалидным индексом формулы")
    return samples


def _list_synthetic(config: Config) -> list[dict]:
    labels_path = os.path.join(config.synthetic_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"  [WARN] синтетика не найдена: {labels_path}")
        return []
    with open(labels_path, encoding="utf-8") as f:
        labels: dict[str, str] = json.load(f)
    return [
        {
            "image_path": os.path.join(config.synthetic_dir, "images", fname),
            "formula": latex,
            "split": "train",
        }
        for fname, latex in labels.items()
    ]


def _list_handwritten(config: Config) -> list[dict]:
    labels_path = os.path.join(config.my_dataset_dir, "labels.json")
    if not os.path.exists(labels_path):
        print(f"  [WARN] рукописный датасет не найден: {labels_path}")
        return []
    with open(labels_path, encoding="utf-8") as f:
        labels: dict[str, str] = json.load(f)
    images_dir = os.path.join(config.my_dataset_dir, "images")
    return [
        {
            "image_path": os.path.join(images_dir, fname),
            "formula": latex,
            "split": "train",
        }
        for fname, latex in labels.items()
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def _print_stats(name: str, manifest: list[dict], skipped: list[str]) -> None:
    if not manifest:
        print(f"  [{name}] нет валидных сэмплов.")
        return

    lengths = np.array([e["length"] for e in manifest])
    widths  = np.array([e["width"]  for e in manifest])

    print(f"\n  [{name}] сохранено: {len(manifest)}  пропущено: {len(skipped)}")
    print(f"  длина формул (токены): "
          f"min={lengths.min()}  "
          f"p50={np.percentile(lengths, 50):.0f}  "
          f"p95={np.percentile(lengths, 95):.0f}  "
          f"max={lengths.max()}")
    print(f"  ширина (px):           "
          f"min={widths.min()}  "
          f"p50={np.percentile(widths, 50):.0f}  "
          f"p95={np.percentile(widths, 95):.0f}  "
          f"max={widths.max()}")


def prepare_dataset(config: Config, dataset_name: str, force: bool = False) -> None:
    cache_subdir  = os.path.join(config.cache_dir, dataset_name)
    manifest_path = os.path.join(cache_subdir, "manifest.json")

    # Идемпотентность: пропустить если кэш существует
    if not force and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  [{dataset_name}] кэш существует ({len(existing)} записей). "
              f"Используйте --force для пересчёта.")
        return

    os.makedirs(cache_subdir, exist_ok=True)

    # При --force удаляем старые .npy чтобы не оставались orphan-файлы от прошлого запуска
    if force and os.path.isdir(cache_subdir):
        removed = 0
        for fname in os.listdir(cache_subdir):
            if fname.endswith(".npy"):
                os.remove(os.path.join(cache_subdir, fname))
                removed += 1
        if removed:
            print(f"  [{dataset_name}] удалено {removed} старых .npy")

    _listers = {
        "im2latex":    _list_im2latex,
        "synthetic":   _list_synthetic,
        "handwritten": _list_handwritten,
    }
    if dataset_name not in _listers:
        raise ValueError(f"Неизвестный датасет: {dataset_name!r}")

    raw_samples = _listers[dataset_name](config)
    if not raw_samples:
        print(f"  [{dataset_name}] нет сэмплов для обработки.")
        return

    print(f"\n[{dataset_name}] обработка {len(raw_samples)} изображений "
          f"(target_h={config.target_height}, max_w={config.max_width})...")

    manifest: list[dict] = []
    skipped:  list[str]  = []

    for sample in tqdm(raw_samples, unit="img"):
        raw_path = sample["image_path"]
        formula  = sample["formula"]
        split    = sample.get("split", "train")

        img = load_image(raw_path)
        if img is None:
            skipped.append(f"broken_file\t{raw_path}")
            continue

        img = crop_to_content(img)

        # Проверяем исходное соотношение сторон ДО resize (чтобы отловить
        # патологически вытянутые изображения: однопиксельные строки и т.п.).
        # Для handwritten порог выше, потому что строки реальных конспектов
        # после слайсера имеют отношение w/h до ~50 (длинная формула во всю A4).
        max_aspect = 80 if dataset_name == "handwritten" else 30
        h0, w0 = img.shape
        if h0 == 0 or w0 / max(h0, 1) > max_aspect:
            skipped.append(f"bad_aspect_ratio\tw={w0} h={h0}\t{raw_path}")
            continue

        img = resize_preserve_aspect(img, config.target_height, config.max_width)
        img = binarize(img)

        # Длина формулы в токенах
        token_len = len(LaTeXTokenizer.tokenize(formula))
        if token_len > config.tokenizer_max_len:
            skipped.append(f"formula_too_long\tlen={token_len}\t{raw_path}")
            continue

        npy_path = os.path.join(cache_subdir, _stable_hash(raw_path) + ".npy")
        np.save(npy_path, img)

        _, w = img.shape
        manifest.append({
            "npy_path": npy_path,
            "formula":  formula,
            "length":   token_len,
            "width":    w,
            "split":    split,
        })

    # Сохранить manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Сохранить skipped.log
    with open(os.path.join(cache_subdir, "skipped.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(skipped))

    # Сохранить stats.json
    if manifest:
        lengths = np.array([e["length"] for e in manifest])
        widths  = np.array([e["width"]  for e in manifest])
        stats = {
            "total":   len(manifest),
            "skipped": len(skipped),
            "length_tokens": {
                "min":  int(lengths.min()),
                "max":  int(lengths.max()),
                "mean": round(float(lengths.mean()), 1),
                "p50":  round(float(np.percentile(lengths, 50)), 1),
                "p95":  round(float(np.percentile(lengths, 95)), 1),
                "p99":  round(float(np.percentile(lengths, 99)), 1),
            },
            "width_px": {
                "min":  int(widths.min()),
                "max":  int(widths.max()),
                "mean": round(float(widths.mean()), 1),
                "p50":  round(float(np.percentile(widths, 50)), 1),
                "p95":  round(float(np.percentile(widths, 95)), 1),
                "p99":  round(float(np.percentile(widths, 99)), 1),
            },
        }
        with open(os.path.join(cache_subdir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    _print_stats(dataset_name, manifest, skipped)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Предобработка датасетов и построение кэша .npy"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["im2latex", "synthetic", "handwritten"],
        default=["im2latex"],
        metavar="DATASET",
        help="Какие датасеты обработать (im2latex | synthetic | handwritten)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Пересчитать кэш с нуля (игнорировать существующий)",
    )
    parser.add_argument(
        "--profile", default="rtx4060_8gb",
        choices=["rtx4060_8gb", "rtx5090_32gb"],
        help="GPU-профиль конфига",
    )
    args = parser.parse_args()

    config = load_config(args.profile)
    print(f"Профиль: {args.profile}")
    print(f"target_height={config.target_height}  max_width={config.max_width}  "
          f"tokenizer_max_len={config.tokenizer_max_len}")
    print(f"Кэш: {os.path.abspath(config.cache_dir)}\n")

    for ds_name in args.datasets:
        prepare_dataset(config, ds_name, force=args.force)

    print("\nГотово.")


if __name__ == "__main__":
    main()
