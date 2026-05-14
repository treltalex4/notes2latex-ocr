"""Optuna-based hyperparameter search для stage 1 (im2latex pretrain).

Запускает N трейлов на ПОЛНОМ датасете (без --limit-batches).
Минимизирует композитный objective (--metric composite, по умолчанию):
устойчивый хвост val_loss + штраф за откат (rebound) + штраф за разрыв
train/val − бонус за val_em. Подробности — в _compute_objective ниже.
MedianPruner + явный divergence-prune убивают плохие трейлы досрочно.

Использование:
    # Стандартный overnight поиск: 15 трейлов × 6 эпох на полном датасете.
    # Ожидаемо ~3-4 будут pruned досрочно → ~15-18 часов на RTX 4060.
    python tune.py

    # С указанием study name (для resume после прерывания)
    python tune.py --study-name lr_search_v2 --storage sqlite:///optuna.db

    # Быстрый smoke-тест (не для реального поиска)
    python tune.py --n-trials 2 --epochs 1 --limit-batches 50

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


# Search space расширен для полного датасета. Прошлые ручные прогоны делались
# на limit_batches и искажали верх диапазона LR — на полном датасете и bf16
# модель потенциально стабильна и при более высоких LR. Pruner отсечёт плохое.
# Что НЕ в search:
#   warmup_steps: адаптируется к длине трейла автоматически (см. ниже).
#   batch_size / grad_accum_steps: упираются в VRAM, инженерный параметр.
#   grad_clip_norm: safety mechanism, на bf16 в стабильной зоне не двигает loss.
SEARCH_SPACE = {
    "learning_rate":   {"low": 5e-5,  "high": 1e-3,  "log": True},
    "weight_decay":    {"low": 1e-4,  "high": 1e-1,  "log": True},
    "dropout":         {"low": 0.05,  "high": 0.30,  "log": False},
    "label_smoothing": {"low": 0.0,   "high": 0.15,  "log": False},
}


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate":   trial.suggest_float("learning_rate",   5e-5, 1e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",    1e-4, 1e-1, log=True),
        "dropout":         trial.suggest_float("dropout",         0.05, 0.30),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0,  0.15),
    }


# ===== Композитный objective =====
# Цель — не «самый низкий val_loss за всю траекторию» (это награждает
# случайный нырок у расходящихся трейлов: трейл [3,1.5,2.8,4.5,6.0]
# получал бы оценку 1.5 и обходил стабильный [3,2.5,2.2,2.0,1.9]).
# Вместо этого — «низкий и УСТОЙЧИВЫЙ хвост val_loss + хороший EM +
# малый разрыв train/val». Все слагаемые в шкале val_loss (~1-3),
# чтобы веса были интерпретируемы.
TAIL_EPOCHS       = 2      # сколько последних эпох усредняем в "хвост"
OBJ_REBOUND_VAL   = 1.0    # штраф: насколько val_loss откатился вверх от своего min
OBJ_REBOUND_TRAIN = 0.5    # штраф: то же для train_loss (нестабильность оптимизации)
OBJ_GAP           = 0.3    # штраф за разрыв train/val (переобучение)
OBJ_EM            = 1.5    # бонус за val_em (EM ∈ [0,1], выше — лучше → вычитается)

# Divergence-prune: независимо от MedianPruner убиваем трейл, который
# расходится САМ В СЕБЕ — val_loss поднялся выше ratio× своего минимума.
DIVERGENCE_RATIO         = 1.3
DIVERGENCE_WARMUP_EPOCHS = 2   # первые N эпох не проверяем (warmup, curriculum)


def _tail_mean(xs: list[float], k: int = TAIL_EPOCHS) -> float:
    tail = xs[-k:]
    return sum(tail) / len(tail)


def _rebound(xs: list[float]) -> float:
    """Насколько ряд откатился вверх от своего лучшего (минимального) значения."""
    return max(0.0, xs[-1] - min(xs))


def _compute_objective(metric: str, train_losses: list[float],
                       val_losses: list[float], val_ems: list[float]) -> float:
    """Метрика для Optuna (всегда минимизируется). Может считаться на
    частичной истории — для per-epoch trial.report() и pruning'а."""
    if metric == "val_loss":
        return _tail_mean(val_losses)
    if metric == "val_em":
        return -_tail_mean(val_ems)
    # composite
    tail_val   = _tail_mean(val_losses)
    tail_train = _tail_mean(train_losses)
    tail_em    = _tail_mean(val_ems)
    return (
        tail_val
        + OBJ_REBOUND_VAL   * _rebound(val_losses)
        + OBJ_REBOUND_TRAIN * _rebound(train_losses)
        + OBJ_GAP           * max(0.0, tail_val - tail_train)
        - OBJ_EM            * tail_em
    )


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
    metric_name = args.metric  # "composite" | "val_loss" | "val_em"

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial)
        print(f"\n[trial {trial.number}] params: " +
              " ".join(f"{k}={v:.4g}" for k, v in params.items()))

        # Возвращаем CUDA-пул системе перед сборкой новой модели: предыдущий
        # трейл (завершённый ИЛИ pruned) оставляет зарезервированный
        # фрагментированный пул. Без сброса фрагментация копится от трейла
        # к трейлу и поздние трейлы рискуют OOM. gc_after_trial чистит
        # только Python-объекты, но не отдаёт CUDA-память системе.
        if device.type == "cuda":
            torch.cuda.empty_cache()

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

        # Warmup: используем config.warmup_steps если он помещается в трейл
        # (< 50% total_steps). Иначе масштабируем до 15% — это гарантирует
        # что warmup завершится и модель успеет поработать при целевом LR.
        if config.warmup_steps <= total_steps * 0.5:
            trial_warmup = config.warmup_steps
        else:
            trial_warmup = max(30, int(total_steps * 0.15))
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
            log_every=args.trial_log_every or 10**9,
            limit_batches=args.limit_batches,
        )

        # Историю копим по эпохам — objective считается по траектории
        # целиком, а не по одиночному «лучшему» нырку.
        train_losses: list[float] = []
        val_losses:   list[float] = []
        val_ems:      list[float] = []
        val_accs:     list[float] = []

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
            # empty_cache до и после validate: у greedy decode (EM) другой
            # профиль аллокаций — без сброса фрагментированный пул повышает
            # риск OOM на длинных val-батчах. Зеркалит train.py.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_loss, val_acc, val_em = validate(
                model, val_loader, criterion, tokenizer, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
                n_em_batches=10,   # 10 батчей ~120 примеров — достаточно для EM-сигнала
                limit_batches=args.val_limit_batches,
                greedy_max_len=config.beam_max_len,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dt = time.time() - t0

            # NaN/inf — оптимизация развалилась, продолжать трейл бессмысленно.
            if not (val_loss == val_loss) or val_loss == float("inf"):
                print(f"  [diverged: val_loss={val_loss} на epoch {epoch+1}]")
                raise optuna.TrialPruned()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_ems.append(val_em)
            val_accs.append(val_acc)

            # Running objective — на частичной истории. MedianPruner
            # сравнивает именно его между трейлами на одной эпохе.
            metric_value = _compute_objective(metric_name, train_losses,
                                              val_losses, val_ems)

            print(f"  ep {epoch+1}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_em={val_em:.3f} "
                  f"| obj={metric_value:.4f} ({dt:.1f}s)")

            # Divergence-prune: трейл расходится сам в себе (откат от min).
            if epoch + 1 > DIVERGENCE_WARMUP_EPOCHS:
                min_val = min(val_losses[:-1])
                if val_losses[-1] > DIVERGENCE_RATIO * min_val:
                    print(f"  [diverged: val_loss {val_losses[-1]:.3f} > "
                          f"{DIVERGENCE_RATIO}× min {min_val:.3f}]")
                    raise optuna.TrialPruned()

            trial.report(metric_value, epoch)
            if trial.should_prune():
                print(f"  [pruned at epoch {epoch+1}]")
                raise optuna.TrialPruned()

        # Разбивка objective — в user_attrs, чтобы видеть её в JSON-отчёте.
        trial.set_user_attr("train_losses", [round(x, 4) for x in train_losses])
        trial.set_user_attr("val_losses",   [round(x, 4) for x in val_losses])
        trial.set_user_attr("val_ems",      [round(x, 4) for x in val_ems])
        trial.set_user_attr("val_accs",     [round(x, 4) for x in val_accs])
        trial.set_user_attr("tail_val_loss",  round(_tail_mean(val_losses), 4))
        trial.set_user_attr("tail_val_em",    round(_tail_mean(val_ems), 4))
        trial.set_user_attr("rebound_val",    round(_rebound(val_losses), 4))
        trial.set_user_attr("rebound_train",  round(_rebound(train_losses), 4))

        return _compute_objective(metric_name, train_losses, val_losses, val_ems)

    return objective


def _trial_to_dict(t) -> dict:
    return {
        "number":  t.number,
        "value":   t.value,
        "params":  t.params,
        "state":   t.state.name,
        "metrics": dict(t.user_attrs),
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
            f"--lr {best.params['learning_rate']:.6g}"
        )
        best_block = {
            "number":      best.number,
            "value":       best.value,
            "params":      best.params,
            "train_cmd":   cmd,
            "note":        "weight_decay, dropout, label_smoothing — пропиши в config.py вручную или передай в train.py через --weight-decay/--dropout/--label-smoothing",
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
    parser.add_argument("--n-trials", type=int, default=16,
                        help="Сколько трейлов")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Эпох в одном трейле на полном датасете. 6 эпох: "
                             "warmup завершается к концу epoch 1, далее 5 эпох "
                             "при рабочем LR — достаточно чтобы плохие конфиги "
                             "разошлись и были убиты pruner'ом.")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Ограничить train-батчи (только для smoke-тестов). "
                             "По умолчанию — полный датасет.")
    parser.add_argument("--val-limit-batches", type=int, default=None,
                        help="Ограничить val-батчи. По умолчанию — полная val.")
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--study-name", default="latex-ocr-stage1")
    parser.add_argument("--storage", default=None,
                        help="URL для resume, напр. sqlite:///optuna.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/tune_results.json")
    parser.add_argument("--metric", default="composite",
                        choices=["composite", "val_loss", "val_em"],
                        help="Что минимизировать. composite (по умолчанию): "
                             "устойчивый хвост val_loss + rebound-штрафы + "
                             "разрыв train/val − бонус за EM. val_loss: просто "
                             "среднее последних эпох. val_em: −среднее EM.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Максимум секунд на study (для CI/ночных прогонов)")
    parser.add_argument("--trial-log-every", type=int, default=1000,
                        help="Печатать step-лог внутри трейла раз в N батчей. "
                             "0 = отключить. По умолчанию 1000 (~7 строк/эпоху).")

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
        n_startup_trials=3,   # первые 3 трейла не прунятся — строим статистику
        n_warmup_steps=1,     # минимум 2 эпохи до прунинга (warmup_steps=1 → прун с epoch 2)
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
