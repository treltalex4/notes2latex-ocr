"""Compare all YOLO training runs in one table.

Usage:
    python -m yolo_training.compare_runs
    python -m yolo_training.compare_runs --sort map5095
    python -m yolo_training.compare_runs --top 5
    python -m yolo_training.compare_runs --detail        # show training params
    python -m yolo_training.compare_runs --detail --top 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

RUNS_DIR = Path(__file__).parent / "runs"

# Fields shown in --detail mode (from args.yaml saved by Ultralytics)
DETAIL_FIELDS = [
    ("model",         "model"),
    ("imgsz",         "imgsz"),
    ("batch",         "batch"),
    ("epochs",        "epochs"),
    ("optimizer",     "optimizer"),
    ("lr0",           "lr0"),
    ("lrf",           "lrf"),
    ("cos_lr",        "cos_lr"),
    ("freeze",        "freeze"),
    ("patience",      "patience"),
    ("close_mosaic",  "close_mosaic"),
    ("mosaic",        "mosaic"),
    ("mixup",         "mixup"),
    ("degrees",       "degrees"),
    ("translate",     "translate"),
    ("scale",         "scale"),
    ("perspective",   "perspective"),
    ("hsv_h",         "hsv_h"),
    ("hsv_s",         "hsv_s"),
    ("hsv_v",         "hsv_v"),
    ("flipud",        "flipud"),
    ("fliplr",        "fliplr"),
    ("label_smoothing", "label_smoothing"),
    ("box",           "box"),
    ("cls",           "cls"),
    ("dfl",           "dfl"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLO training runs.")
    p.add_argument("--sort", default="map50",
                   choices=["map50", "map5095", "name", "epochs"],
                   help="Column to sort by (default: map50).")
    p.add_argument("--top", type=int, default=None,
                   help="Show only top N runs.")
    p.add_argument("--detail", action="store_true",
                   help="Show full training params (model, imgsz, batch, "
                        "lr, augmentation, ...) for each run.")
    return p.parse_args()


def load_args_yaml(run_dir: Path) -> dict:
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        return {}
    try:
        with args_yaml.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def collect_run(csv_path: Path) -> dict | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    df.columns = [c.strip() for c in df.columns]

    map50_col = next((c for c in df.columns if "mAP50(B)" in c and "95" not in c), None)
    map5095_col = next((c for c in df.columns if "mAP50-95(B)" in c), None)
    if map50_col is None:
        return None

    best_idx = df[map50_col].idxmax()
    best_row = df.iloc[best_idx]

    # Total training duration = cumulative time at the LAST epoch (not best one).
    time_col = next((c for c in df.columns if c == "time"), None)
    duration = float(df[time_col].iloc[-1]) / 60 if time_col else None

    # Final epoch losses (proxy for overfitting check)
    train_box_col = next((c for c in df.columns if "train/box_loss" in c), None)
    val_box_col   = next((c for c in df.columns if "val/box_loss" in c), None)
    final_train_box = float(df[train_box_col].iloc[-1]) if train_box_col else None
    final_val_box   = float(df[val_box_col].iloc[-1])   if val_box_col   else None

    return {
        "run":        csv_path.parent.name,
        "run_dir":    csv_path.parent,
        "best_ep":    int(best_row.get("epoch", best_idx)) + 1,
        "total_ep":   len(df),
        "map50":      float(best_row[map50_col]),
        "map5095":    float(best_row[map5095_col]) if map5095_col else 0.0,
        "duration_m": duration,
        "final_train_box": final_train_box,
        "final_val_box":   final_val_box,
    }


def print_summary_table(runs: list[dict]) -> None:
    print(f"\n{'Run':<35} {'mAP50':>7} {'mAP50-95':>9} {'best@':>6} {'/total':>7} {'time':>8}")
    print("-" * 80)
    for r in runs:
        dur = f"{r['duration_m']:.0f}m" if r["duration_m"] else "?"
        print(f"{r['run']:<35} {r['map50']:>7.4f} {r['map5095']:>9.4f} "
              f"{r['best_ep']:>6} {r['total_ep']:>7} {dur:>8}")
    print()


def print_detail(runs: list[dict]) -> None:
    for i, r in enumerate(runs, 1):
        ya = load_args_yaml(r["run_dir"])
        dur = f"{r['duration_m']:.0f}m" if r["duration_m"] else "?"

        print(f"\n{'=' * 78}")
        print(f"[{i}] {r['run']}")
        print(f"{'=' * 78}")

        # Headline metrics
        overfit_gap = None
        if r["final_train_box"] is not None and r["final_val_box"] is not None:
            overfit_gap = r["final_val_box"] - r["final_train_box"]

        print(f"  mAP50       : {r['map50']:.4f}        "
              f"mAP50-95   : {r['map5095']:.4f}")
        print(f"  best epoch  : {r['best_ep']}/{r['total_ep']}        "
              f"duration   : {dur}")
        if overfit_gap is not None:
            warn = "  [!] overfit signal" if overfit_gap > 0.5 else ""
            print(f"  final losses: train_box={r['final_train_box']:.3f}  "
                  f"val_box={r['final_val_box']:.3f}  "
                  f"(gap={overfit_gap:+.3f}){warn}")

        if not ya:
            print("  (args.yaml not found - older run)")
            continue

        # Training params
        print(f"\n  -- Training -----------------------------------------------------")
        for label, key in DETAIL_FIELDS[:11]:
            if key in ya:
                print(f"    {label:<16}: {ya[key]}")

        print(f"\n  -- Augmentation -------------------------------------------------")
        for label, key in DETAIL_FIELDS[11:22]:
            if key in ya:
                print(f"    {label:<16}: {ya[key]}")

        print(f"\n  -- Loss weights -------------------------------------------------")
        for label, key in DETAIL_FIELDS[22:]:
            if key in ya:
                print(f"    {label:<16}: {ya[key]}")
    print()


def main() -> None:
    args = parse_args()

    if not RUNS_DIR.exists():
        print(f"Not found: {RUNS_DIR}")
        return

    runs = []
    for csv in sorted(RUNS_DIR.glob("*/results.csv")):
        info = collect_run(csv)
        if info:
            runs.append(info)

    if not runs:
        print("No completed runs found.")
        return

    sort_key = {"map50": "map50", "map5095": "map5095",
                "name": "run", "epochs": "total_ep"}[args.sort]
    runs.sort(key=lambda r: r[sort_key], reverse=(args.sort != "name"))

    if args.top:
        runs = runs[:args.top]

    if args.detail:
        print_summary_table(runs)
        print_detail(runs)
    else:
        print_summary_table(runs)


if __name__ == "__main__":
    main()
