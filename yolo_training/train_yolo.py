"""Train (or fine-tune) a YOLO line-detection model.

All parameters are in config_y.py — edit that file to change anything.
Weights are stored in yolo_training/weights/, runs in yolo_training/runs/.

Usage:
    python -m yolo_training.train_yolo                    # base training
    python -m yolo_training.train_yolo --name my_run
    python -m yolo_training.train_yolo --finetune         # fine-tune on f_dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
WEIGHTS_DIR = THIS_DIR / "weights"
RUNS_DIR = THIS_DIR / "runs"
DATA_YAML = THIS_DIR / "data.yaml"


def find_best_pt(model_stem: str) -> Path:
    """Most recently modified best.pt for the given model stem (excludes finetune runs)."""
    candidates = sorted(
        (p for p in RUNS_DIR.glob(f"{model_stem}*/weights/best.pt")
         if "_finetune" not in p.parts[-3]),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trained best.pt found under {RUNS_DIR}/{model_stem}*/weights/.\n"
            f"Train the base model first:  python -m yolo_training.train_yolo"
        )
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO for line detection.")
    p.add_argument("--name", default=None,
                   help="Run name inside runs/. Defaults to model stem.")
    p.add_argument("--finetune", action="store_true",
                   help="Fine-tune the latest best.pt on the dataset specified by "
                        "cfg.finetune_data_yaml (e.g. f_dataset/).")
    return p.parse_args()


def main() -> None:
    from yolo_training.config_y import cfg

    args = parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model_stem = Path(cfg.model).stem

    # ── Pick weights, dataset, hyperparams depending on mode ──────────────────
    if args.finetune:
        weight_path = find_best_pt(model_stem)
        data_yaml = THIS_DIR / cfg.finetune_data_yaml
        if not data_yaml.exists():
            print(f"Error: {data_yaml} not found. Create it pointing to f_dataset/.",
                  file=sys.stderr)
            sys.exit(1)
        run_name = args.name or f"{model_stem}_finetune"
        epochs = cfg.finetune_epochs
        lr0 = cfg.finetune_lr0
        freeze = cfg.finetune_freeze
        close_mosaic = cfg.finetune_close_mosaic
        print(f"[finetune] Starting from: {weight_path}")
        print(f"[finetune] Dataset:       {data_yaml}")
    else:
        weight_path = WEIGHTS_DIR / cfg.model
        data_yaml = DATA_YAML
        run_name = args.name or model_stem
        epochs = cfg.epochs
        lr0 = cfg.lr0
        freeze = cfg.freeze
        close_mosaic = cfg.close_mosaic

    os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS_DIR))

    from ultralytics import YOLO

    if weight_path.exists():
        model = YOLO(str(weight_path))
    else:
        print(f"[train] Downloading {cfg.model} → {weight_path}")
        model = YOLO(cfg.model)
        downloaded = Path(cfg.model)
        if downloaded.exists() and downloaded.resolve() != weight_path.resolve():
            downloaded.replace(weight_path)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        device=cfg.device,
        optimizer=cfg.optimizer,
        lr0=lr0,
        lrf=cfg.lrf,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        cos_lr=cfg.cos_lr,
        freeze=freeze,
        box=cfg.box,
        cls=cfg.cls,
        dfl=cfg.dfl,
        label_smoothing=cfg.label_smoothing,
        patience=cfg.patience,
        close_mosaic=close_mosaic,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
