"""Адаптер UniMER-1M + UniMER-Test → data_raw/unimer/ в формате проекта.

Перепаковывает сырой HF-дамп в плоскую структуру `images/ + labels.json`,
идентичную data_raw/synthetic. После этого prepare_data.py обрабатывает
UniMER как обычный датасет (см. _list_unimer).

Источник (unimer_download/, как скачано huggingface-cli):
  UniMER-1M/
    images/NNNNNNN.png   — ~986k печатных формул
    train.txt            — формула per line; пустые строки = картинки нет
  UniMER-Test/
    {spe,cpe,sce,hwe}/NNNNNNN.png
    {spe,cpe,sce,hwe}.txt

Маппинг: картинка NNNNNNN.png ↔ строка с индексом N (0-indexed) в txt.

Назначение splits:
  train + validate  — стратифицированная выборка из UniMER-1M
  test              — UniMER-Test, категории spe/cpe/sce (печатные).
                      hwe (рукописные) по умолчанию НЕ берём — другой домен,
                      для stage 3. Включить через --test-categories.

Стратификация train/val — по длине формулы (в символах): все длинные формулы
(≥350 символов) берутся целиком, короткие добираются пропорционально до
--target-n. Это даёт тяжёлым конструкциям представительство выше
пропорционального — лечит «мало примеров сложных случаев».

Usage:
    python adapt_unimer.py
    python adapt_unimer.py --target-n 200000
    python adapt_unimer.py --test-categories spe cpe sce hwe
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


# Бакеты длины формулы (в СИМВОЛАХ — токенайзера на этом этапе ещё нет).
# Граница 350 делит "короткие/средние" и "длинные" — длинные берём целиком.
_LEN_BUCKETS = [(0, 100), (100, 200), (200, 350), (350, 500), (500, 10**9)]
_LONG_THRESHOLD = 200


def _bucket_of(char_len: int) -> tuple[int, int]:
    for lo, hi in _LEN_BUCKETS:
        if lo <= char_len < hi:
            return (lo, hi)
    return _LEN_BUCKETS[-1]


def _load_pairs(images_dir: Path, txt_path: Path) -> list[tuple[Path, str]]:
    """Возвращает [(image_path, formula)] для всех картинок у которых есть
    непустая строка-формула. Маппинг: image NNNNNNN.png → txt line index N."""
    if not images_dir.is_dir() or not txt_path.is_file():
        raise FileNotFoundError(f"Не найдено: {images_dir} или {txt_path}")

    lines = txt_path.read_text(encoding="utf-8").split("\n")

    pairs: list[tuple[Path, str]] = []
    n_skipped_empty = 0
    n_skipped_oob = 0
    for img_path in images_dir.glob("*.png"):
        try:
            idx = int(img_path.stem)
        except ValueError:
            continue   # имя не числовое — пропускаем
        if idx >= len(lines):
            n_skipped_oob += 1
            continue
        formula = lines[idx].strip()
        if not formula:
            n_skipped_empty += 1
            continue
        pairs.append((img_path, formula))

    if n_skipped_empty or n_skipped_oob:
        print(f"    [{images_dir.parent.name}/{images_dir.name}] "
              f"пропущено: {n_skipped_empty} пустых строк, "
              f"{n_skipped_oob} вне диапазона txt")
    return pairs


def _stratified_sample(pairs: list[tuple[Path, str]], target_n: int,
                       seed: int) -> list[tuple[Path, str]]:
    """Выборка target_n с over-representation длинных формул.

    Длинные (≥_LONG_THRESHOLD символов) берутся ЦЕЛИКОМ — они редкие и ценные.
    Остаток добирается из коротких/средних пропорционально (случайно).
    """
    if len(pairs) <= target_n:
        print(f"    Выборка не нужна: всего {len(pairs)} ≤ target {target_n}")
        return list(pairs)

    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list] = defaultdict(list)
    for p in pairs:
        buckets[_bucket_of(len(p[1]))].append(p)

    # Шаг 1: забираем все длинные формулы.
    sampled: list[tuple[Path, str]] = []
    for (lo, hi), items in buckets.items():
        if lo >= _LONG_THRESHOLD:
            sampled.extend(items)

    # Шаг 2: остаток добираем из коротких/средних.
    short = [p for (lo, hi), items in buckets.items()
             if lo < _LONG_THRESHOLD for p in items]
    rng.shuffle(short)
    remaining = max(0, target_n - len(sampled))
    sampled.extend(short[:remaining])

    rng.shuffle(sampled)

    # Диагностика распределения по бакетам.
    dist: dict[tuple[int, int], int] = defaultdict(int)
    for p in sampled:
        dist[_bucket_of(len(p[1]))] += 1
    print(f"    Выборка {len(sampled)} из {len(pairs)}. Распределение по длине (символы):")
    for lo, hi in _LEN_BUCKETS:
        cnt = dist[(lo, hi)]
        hi_label = "∞" if hi >= 10**9 else str(hi)
        print(f"      [{lo:>4}-{hi_label:>4}): {cnt:>7} ({100*cnt/len(sampled):.1f}%)")
    return sampled


def _collect_unimer1m(src: Path, target_n: int, val_ratio: float,
                      seed: int) -> list[tuple[Path, str, str, str]]:
    """train + validate из UniMER-1M. Возвращает [(path, formula, split, source)]."""
    print("  UniMER-1M (train + validate):")
    pairs = _load_pairs(src / "UniMER-1M" / "images",
                        src / "UniMER-1M" / "train.txt")
    print(f"    Всего валидных пар: {len(pairs)}")
    pairs = _stratified_sample(pairs, target_n, seed)

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    out: list[tuple[Path, str, str, str]] = []
    out += [(p, f, "validate", "unimer-1m") for p, f in pairs[:n_val]]
    out += [(p, f, "train", "unimer-1m") for p, f in pairs[n_val:]]
    print(f"    → train={len(pairs) - n_val}, validate={n_val}")
    return out


def _collect_unimer_test(src: Path,
                         categories: list[str]) -> list[tuple[Path, str, str, str]]:
    """test из UniMER-Test выбранных категорий."""
    print(f"  UniMER-Test (test, категории: {', '.join(categories)}):")
    out: list[tuple[Path, str, str, str]] = []
    for cat in categories:
        pairs = _load_pairs(src / "UniMER-Test" / cat,
                            src / "UniMER-Test" / f"{cat}.txt")
        out += [(p, f, "test", f"unimer-test-{cat}") for p, f in pairs]
        print(f"    {cat}: {len(pairs)}")
    print(f"    → test всего: {len(out)}")
    return out


def _copy_one(args: tuple[Path, Path]) -> None:
    src_path, dst_path = args
    shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="unimer_download",
                        help="Папка со скачанным UniMER (содержит UniMER-1M/, UniMER-Test/)")
    parser.add_argument("--dst", default="data_raw/unimer",
                        help="Куда писать адаптированный датасет")
    parser.add_argument("--target-n", type=int, default=500000,
                        help="Размер выборки из UniMER-1M (train+val). "
                             "200000 для быстрой итерации, 500000 для серьёзного прогона.")
    parser.add_argument("--val-ratio", type=float, default=0.03,
                        help="Доля validate от выборки UniMER-1M")
    parser.add_argument("--test-categories", nargs="+",
                        default=["spe", "cpe", "sce"],
                        choices=["spe", "cpe", "sce", "hwe"],
                        help="Категории UniMER-Test для test split. По умолчанию "
                             "печатные (spe/cpe/sce). hwe — рукописные, добавлять "
                             "только если оцениваешь handwritten-домен.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для воспроизводимой выборки и split'ов")
    parser.add_argument("--copy-workers", type=int, default=16,
                        help="Потоков для копирования PNG (I/O-bound)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise FileNotFoundError(f"Источник не найден: {src}")

    print(f"Источник: {src}")
    print(f"Назначение: {dst}\n")

    # 1. Собираем все (path, formula, split, source).
    samples = _collect_unimer1m(src, args.target_n, args.val_ratio, args.seed)
    samples += _collect_unimer_test(src, args.test_categories)

    # 2. Готовим папку назначения.
    images_dst = dst / "images"
    images_dst.mkdir(parents=True, exist_ok=True)

    # 3. Переименовываем последовательно (избегаем коллизий имён между
    # UniMER-1M и UniMER-Test — оба нумеруются с 0000000) и копируем.
    print(f"\nКопирование {len(samples)} картинок в {images_dst} "
          f"({args.copy_workers} потоков)...")
    labels: dict[str, dict] = {}
    copy_tasks: list[tuple[Path, Path]] = []
    for i, (src_path, formula, split, source) in enumerate(samples):
        new_name = f"{i:07d}.png"
        labels[new_name] = {"formula": formula, "split": split, "source": source}
        copy_tasks.append((src_path, images_dst / new_name))

    with ThreadPoolExecutor(max_workers=args.copy_workers) as pool:
        list(tqdm(pool.map(_copy_one, copy_tasks),
                  total=len(copy_tasks), desc="copy"))

    # 4. Пишем labels.json.
    labels_path = dst / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f"\nlabels.json → {labels_path}")

    # 5. Сводка.
    by_split: dict[str, int] = defaultdict(int)
    for entry in labels.values():
        by_split[entry["split"]] += 1
    print(f"\n{'=' * 50}")
    print(f"Готово. Всего {len(labels)} сэмплов:")
    for split in ("train", "validate", "test"):
        print(f"  {split:<10}: {by_split[split]}")
    print(f"{'=' * 50}")
    print(f"\nДальше: python prepare_data.py --datasets unimer")


if __name__ == "__main__":
    main()
