"""Optuna-based hyperparameter search для stage 1 (im2latex pretrain).

Запускает N коротких трейлов, каждый = K эпох × M батчей на сабсете im2latex.
Минимизирует val_loss (или максимизирует val_em через --metric val_em).
Поддерживает MedianPruner для досрочного убийства плохих трейлов.

Использование:
    # Быстрая проверка
    python tune.py --n-trials 2 --epochs 1 --limit-batches 20

    # Реальный поиск
    python tune.py --n-trials 40 --epochs 3 --limit-batches 500

    # С resume через sqlite
    python tune.py --n-trials 100 --study-name lr_v1 --storage sqlite:///optuna.db

Результаты пишутся в --output (по умолчанию checkpoints/tune_results.json):
лучший трейл, топ-5, search space, готовая CLI-команда для train.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.trial import TrialState
from torch.optim import AdamW

from config import load_config
from data.dataset import build_multi_dataloaders
from data.tokenizer import LaTeXTokenizer, PAD_ID
from model.model import Notes2LaTeX, count_parameters
from train import (
    apply_curriculum, make_lr_scheduler, train_one_epoch, validate,
)


# Search space НАРРОВАННЫЙ — диапазоны вокруг валидированных вручную оптимумов.
# Что НЕ в search:
#   weight_decay: ручной свип (0, 0.01, 0.05, 0.1) на сабсете не дал сигнала.
#   warmup_steps: при коротких трейлах (~1000 optim steps) tune может сэмплить
#                 значения, при которых warmup не успевает завершиться, что
#                 даёт ложный сигнал. Дефолт 1000 уже валидирован на full pretrain.
SEARCH_SPACE = {
    "learning_rate":   {"low": 5e-4,  "high": 3e-3,  "log": True},     # вокруг 1e-3
    "dropout":         {"low": 0.05,  "high": 0.20,  "log": False},    # вокруг 0.1
    "label_smoothing": {"low": 0.0,   "high": 0.15,  "log": False},    # untested
}


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate":   trial.suggest_float("learning_rate",   5e-4, 3e-3, log=True),
        "dropout":         trial.suggest_float("dropout",         0.05, 0.20),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0,  0.15),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_objective(args, base_overrides: dict, tokenizer: LaTeXTokenizer, device):
    """Возвращает objective(trial) с захваченным контекстом.

    Каждый трейл строит свежую модель и loader'ы — гиперпараметры
    (особенно dropout) затрагивают архитектуру.
    """
    metric_name = args.metric  # "val_loss" | "val_em"

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        print(f"\n[trial {trial.number}] params: " +
              " ".join(f"{k}={v:.4g}" for k, v in params.items()))

        # Полный набор overrides: CLI + sampled.
        overrides = dict(base_overrides)
        overrides.update(params)
        config = load_config(args.profile, **overrides)

        model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
        if trial.number == 0:
            print(f"  Параметров модели: {count_parameters(model):,}")

        # AMP
        amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        use_amp = config.use_amp and device.type == "cuda"
        scaler_enabled = use_amp and amp_dtype == torch.float16
        scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)

        # Stage 1 loaders.
        train_loader, val_loader, _ = build_multi_dataloaders(config, tokenizer, stage=1)

        accum_steps = config.grad_accum_steps
        batches_per_epoch = len(train_loader)
        if args.limit_batches:
            batches_per_epoch = min(batches_per_epoch, args.limit_batches)
        steps_per_epoch = max(1, batches_per_epoch // accum_steps)
        total_steps = max(1, steps_per_epoch * args.epochs)

        # Warmup адаптируется под длину трейла (10% от total_steps, min 30).
        # Иначе config.warmup_steps=1000 может не успеть завершиться в коротком
        # трейле и trial выдаст ложный сигнал ("низкий lr — лучше" просто потому
        # что lr никогда не достигал base_lr).
        trial_warmup = max(30, int(total_steps * 0.10))
        if trial.number == 0:
            print(f"  Trial budget: total_steps={total_steps}  warmup={trial_warmup}")

        optimizer = AdamW(model.parameters(),
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay)
        scheduler = make_lr_scheduler(optimizer, trial_warmup, total_steps)
        criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_ID,
            label_smoothing=config.label_smoothing,
        )

        # Используем train.py-сигнатуру: SimpleNamespace с теми полями, что
        # читает train_one_epoch.
        train_args = SimpleNamespace(
            log_every=10**9,           # отключаем step-логи внутри трейла
            limit_batches=args.limit_batches,
        )

        best_metric = float("inf") if metric_name == "val_loss" else -float("inf")
        for epoch in range(args.epochs):
            t0 = time.time()
            apply_curriculum(config, train_loader, stage=1,
                             epoch=epoch, total_epochs=args.epochs)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                criterion, tokenizer, device,
                log_every=train_args.log_every,
                limit_batches=train_args.limit_batches,
                grad_clip_norm=config.grad_clip_norm,
                accum_steps=accum_steps,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            val_loss, val_acc, val_em = validate(
                model, val_loader, criterion, tokenizer, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
                n_em_batches=1,
                limit_batches=args.val_limit_batches,
            )
            dt = time.time() - t0

            # Метрика для Optuna: либо val_loss (min), либо -val_em (max → min).
            if metric_name == "val_loss":
                metric_value = val_loss
                best_metric = min(best_metric, val_loss)
            else:
                metric_value = -val_em
                best_metric = min(best_metric, -val_em)

            print(f"  ep {epoch+1}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_em={val_em:.3f} "
                  f"({dt:.1f}s)")

            trial.report(metric_value, epoch)
            if trial.should_prune():
                print(f"  [pruned at epoch {epoch+1}]")
                raise optuna.TrialPruned()

        return best_metric

    return objective


def _trial_to_dict(t) -> dict:
    return {
        "number": t.number,
        "value":  t.value,
        "params": t.params,
        "state":  t.state.name,
    }


def _save_results(study: optuna.Study, args, output_path: str) -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == TrialState.FAIL]

    # top-5 среди завершённых
    completed_sorted = sorted(completed, key=lambda t: t.value)
    top5 = [_trial_to_dict(t) for t in completed_sorted[:5]]

    if completed:
        best = study.best_trial
        # Готовая команда для train.py с лучшими гиперами. Только те, что
        # train.py CLI поддерживает напрямую — остальные пойдут через config.
        cmd = (
            f"python train.py --profile {args.profile} "
            f"--lr {best.params['learning_rate']:.6g} "
            f"--warmup-steps {best.params['warmup_steps']}"
        )
        best_block = {
            "number":      best.number,
            "value":       best.value,
            "params":      best.params,
            "train_cmd":   cmd,
            "note":        "weight_decay, dropout, label_smoothing — пропиши в config.py вручную или передай через load_config(**overrides)",
        }
    else:
        best_block = None

    out = {
        "study_name":         study.study_name,
        "n_trials_total":     len(study.trials),
        "n_trials_completed": len(completed),
        "n_trials_pruned":    len(pruned),
        "n_trials_failed":    len(failed),
        "metric":             args.metric,
        "best_trial":         best_block,
        "top_5":              top5,
        "search_space":       SEARCH_SPACE,
        "settings": {
            "profile":           args.profile,
            "epochs_per_trial":  args.epochs,
            "limit_batches":     args.limit_batches,
            "val_limit_batches": args.val_limit_batches,
            "seed":              args.seed,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="rtx4060_8gb")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Сколько трейлов. Optuna TPE с ~25-30 трейлов даёт "
                             "разумный signal на 3-параметрическом поиске.")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Эпох в одном трейле. 4 эпохи покрывают warmup и "
                             "несколько шагов после — разница между конфигами видна.")
    parser.add_argument("--limit-batches", type=int, default=300,
                        help="Сколько train-батчей в одном трейле. "
                             "Default 300 * batch_size = ~5k формул/эпоху.")
    parser.add_argument("--val-limit-batches", type=int, default=20,
                        help="Сколько val-батчей. Default 20 = ~320 формул, "
                             "ранг val_loss консервативен.")
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--study-name", default="latex-ocr-stage1")
    parser.add_argument("--storage", default=None,
                        help="URL для resume, напр. sqlite:///optuna.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/tune_results.json")
    parser.add_argument("--metric", default="val_loss", choices=["val_loss", "val_em"],
                        help="Что оптимизировать (val_em максимизируется через -val_em)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Максимум секунд на study (для CI/ночных прогонов)")

    # Standard overrides (как в train.py)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)

    args = parser.parse_args()

    _set_seed(args.seed)

    base_overrides: dict = {}
    if args.batch_size is not None:  base_overrides["batch_size"]  = args.batch_size
    if args.num_workers is not None: base_overrides["num_workers"] = args.num_workers
    if args.max_width is not None:   base_overrides["max_width"]   = args.max_width

    # Девайс и токенайзер строятся один раз — переиспользуются между трейлами.
    probe = load_config(args.profile, **base_overrides)
    device = torch.device(probe.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    )
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(args, base_overrides, tokenizer, device)

    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                       gc_after_trial=True, show_progress_bar=False)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] saving partial results...")
        _save_results(study, args, args.output)
        sys.exit(130)

    _save_results(study, args, args.output)

    # Финальный отчёт в stdout
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]
    print(f"\nDone. completed={len(completed)} pruned={len(pruned)} "
          f"total={len(study.trials)}")
    if completed:
        best = study.best_trial
        print(f"Best ({args.metric}={best.value:.4f}, trial {best.number}):")
        for k, v in best.params.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
