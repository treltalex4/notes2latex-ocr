"""Сравнение прогонов из checkpoints/runs/*.json.

Использование:
    python runs.py                         # таблица всех прогонов
    python runs.py --sort val_loss         # сортировка
    python runs.py --filter lr=1e-3        # фильтр по гиперпараметру
    python runs.py --details s1_lr1e-03... # подробности одного прогона
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RUNS_DIR = Path("checkpoints/runs")


def _load_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            data["_path"] = str(p)
            runs.append(data)
        except json.JSONDecodeError as e:
            print(f"[skip] не удалось распарсить {p.name}: {e}")
    return runs


def _filter_runs(runs: list[dict], filters: list[str]) -> list[dict]:
    """filters: ['lr=1e-3', 'wd=0.01']."""
    if not filters:
        return runs
    out = []
    for r in runs:
        h = r.get("hyperparams", {})
        match = True
        for f in filters:
            if "=" not in f:
                continue
            k, v = f.split("=", 1)
            actual = h.get(k)
            if actual is None:
                match = False
                break
            # Сравнение с tolerance для float'ов
            try:
                if abs(float(actual) - float(v)) > 1e-9:
                    match = False
                    break
            except (TypeError, ValueError):
                if str(actual) != v:
                    match = False
                    break
        if match:
            out.append(r)
    return out


def _print_table(runs: list[dict], sort_by: str) -> None:
    if not runs:
        print("Нет прогонов в", RUNS_DIR)
        return

    # Сбор строк
    rows = []
    for r in runs:
        h = r.get("hyperparams", {})
        b = r.get("best", {})
        rows.append({
            "name":       r.get("run_name", "?"),
            "stage":      r.get("stage", "?"),
            "lr":         h.get("lr", float("nan")),
            "wd":         h.get("weight_decay", float("nan")),
            "dropout":    h.get("dropout", float("nan")),
            "ls":         h.get("label_smoothing", float("nan")),
            "warmup":     h.get("warmup_steps", "?"),
            "lim_batch":  h.get("limit_batches", "all"),
            "epochs":     r.get("n_epochs_done", h.get("epochs", "?")),
            "best_loss":  b.get("val_loss", float("inf")),
            "best_acc":   b.get("val_acc", 0.0),
            "best_em":    b.get("val_em", 0.0),
            "best_epoch": (b.get("epoch", -1) + 1) if "epoch" in b else "-",
            "time_s":     r.get("total_time_seconds", 0),
            "seed":       h.get("seed", "-"),
            "interrupted": r.get("interrupted", False),
        })

    # Сортировка
    sort_key_map = {
        "val_loss":  lambda r: r["best_loss"],
        "val_acc":   lambda r: -r["best_acc"],
        "val_em":    lambda r: -r["best_em"],
        "name":      lambda r: r["name"],
        "lr":        lambda r: r["lr"],
        "time":      lambda r: r["time_s"],
    }
    rows.sort(key=sort_key_map.get(sort_by, sort_key_map["val_loss"]))

    # Печать
    print(f"\n{'#':<3} {'NAME':<48} {'LR':<8} {'WD':<6} {'DROP':<5} {'LS':<5} "
          f"{'EPOCHS':<7} {'BEST_LOSS':<10} {'ACC':<5} {'EM':<5} {'BEST_EP':<8} {'TIME':<8}")
    print("-" * 130)
    for i, r in enumerate(rows, 1):
        flag = " [!]" if r["interrupted"] else ""
        print(f"{i:<3} {r['name']:<48} "
              f"{r['lr']:<8.1e} {r['wd']:<6g} {r['dropout']:<5g} {r['ls']:<5g} "
              f"{str(r['epochs']):<7} "
              f"{r['best_loss']:<10.4f} {r['best_acc']:<5.3f} {r['best_em']:<5.3f} "
              f"{str(r['best_epoch']):<8} "
              f"{r['time_s']:<7.0f}s{flag}")
    print(f"\nTotal: {len(rows)} run(s).  Sort: {sort_by}")


def _print_details(run_name: str) -> None:
    candidates = list(RUNS_DIR.glob(f"{run_name}*.json"))
    if not candidates:
        # Fallback — поиск по подстроке
        candidates = [p for p in RUNS_DIR.glob("*.json") if run_name in p.stem]
    if not candidates:
        print(f"Не найдено: {run_name}")
        return
    if len(candidates) > 1:
        print(f"Найдено несколько совпадений ({len(candidates)}):")
        for p in candidates:
            print(f"  {p.stem}")
        print("Уточни имя.")
        return

    path = candidates[0]
    with path.open(encoding="utf-8") as f:
        run = json.load(f)

    print(f"\n=== {run['run_name']} ===")
    print(f"Stage: {run.get('stage')}  Started: {run.get('started_at', '?')}")
    print("\nHyperparams:")
    for k, v in run.get("hyperparams", {}).items():
        print(f"  {k:<18} {v}")
    print(f"\nEpochs ({len(run.get('epochs', []))}):")
    print(f"{'EP':<4} {'TRAIN_LOSS':<11} {'VAL_LOSS':<10} {'VAL_ACC':<8} "
          f"{'VAL_EM':<7} {'LR_END':<10} {'TIME':<6}")
    print("-" * 70)
    for e in run.get("epochs", []):
        print(f"{e['epoch']+1:<4} "
              f"{e['train_loss']:<11.4f} {e['val_loss']:<10.4f} "
              f"{e['val_acc']:<8.3f} {e['val_em']:<7.3f} "
              f"{e['lr_end']:<10.2e} {e['time_seconds']:<5.1f}s")

    if "best" in run:
        b = run["best"]
        print(f"\nBest: epoch {b['epoch']+1}  val_loss={b['val_loss']:.4f}  "
              f"val_acc={b['val_acc']:.3f}  val_em={b['val_em']:.3f}")
        print(f"Total time: {run.get('total_time_seconds', 0):.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Просмотр и сравнение прогонов train.py.")
    parser.add_argument("--sort", default="val_loss",
                        choices=["val_loss", "val_acc", "val_em", "name", "lr", "time"],
                        help="По какому полю сортировать (default: val_loss).")
    parser.add_argument("--filter", action="append", default=[],
                        help="Фильтр вида key=value, можно несколько. "
                             "Примеры: --filter lr=1e-3 --filter wd=0.01")
    parser.add_argument("--details", default=None,
                        help="Имя прогона (полное или подстрока) — покажет все его эпохи.")
    parser.add_argument("--all-details", action="store_true",
                        help="Подробности всех прогонов (с фильтром и сортировкой, как у table).")
    args = parser.parse_args()

    if args.details:
        _print_details(args.details)
        return

    runs = _load_runs()
    runs = _filter_runs(runs, args.filter)

    if args.all_details:
        # Сначала сводная таблица (для контекста), потом детали каждого прогона
        # в том же порядке.
        _print_table(runs, args.sort)
        sort_key_map = {
            "val_loss":  lambda r: r.get("best", {}).get("val_loss", float("inf")),
            "val_acc":   lambda r: -r.get("best", {}).get("val_acc", 0.0),
            "val_em":    lambda r: -r.get("best", {}).get("val_em", 0.0),
            "name":      lambda r: r.get("run_name", ""),
            "lr":        lambda r: r.get("hyperparams", {}).get("lr", 0),
            "time":      lambda r: r.get("total_time_seconds", 0),
        }
        sorted_runs = sorted(runs, key=sort_key_map.get(args.sort, sort_key_map["val_loss"]))
        for r in sorted_runs:
            _print_details(r["run_name"])
        return

    _print_table(runs, args.sort)


if __name__ == "__main__":
    main()
