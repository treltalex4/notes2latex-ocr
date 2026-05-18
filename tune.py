"""Optuna-based hyperparameter search для stage 1 (im2latex pretrain).

Подбирает 6 гиперпараметров (lr, wd, dropout, label_smoothing, warmup_steps,
grad_clip_norm) на ПОЛНОМ датасете. Минимизирует композитный objective
(--metric composite, по умолчанию): устойчивый хвост val_loss + штраф за
откат (rebound) + штраф за разрыв train/val − бонус за val_em. Подробности
— в _compute_objective ниже. MedianPruner + явный divergence-prune убивают
плохие трейлы досрочно.

Использование:
    # Полный большой tune (дефолт): 100 трейлов × 18 эпох, schedule_epochs=40.
    # Параметры найденные тут напрямую переносятся в 40-эпохный train.py:
    #  - lr подобран под cosine с горизонтом 40 эпох (schedule_epochs trick)
    #  - warmup_steps в АБСОЛЮТНЫХ шагах — одно значение в трейле и реале
    #  - curriculum в трейле привязан к schedule_epochs (см. apply_curriculum
    #    в objective) → переходы на тех же эпохах что и в реальном прогоне
    #  - остальные параметры horizon-independent
    python tune.py --schedule-epochs 40 --study-name big_tune_5090_v1 \\
                   --storage sqlite:///optuna_big.db

    # Resume после прерывания (SQLite сохраняет state)
    python tune.py --study-name big_tune_5090_v1 --storage sqlite:///optuna_big.db

    # Быстрый smoke-тест
    python tune.py --n-trials 2 --epochs 2 --limit-batches 50

Кросс-платформенность: tune использует --profile rtx4060_8gb (батч и архитектура
ноутбука) → найденные параметры подходят и для облака (RTX 5090) и для ноутбука.
На Linux включить torch.compile в config или через --use-compile в train.py.

Результаты пишутся в --output (по умолчанию checkpoints/tune_results.json):
лучший трейл с готовой CLI-командой для train.py (все 6 параметров).
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
    _compile_model,
)


# Search space для большого tune на полном датасете и длинных трейлах (15-20 эпох).
# Расширен по результатам ручных прогонов: warmup_steps и grad_clip_norm
# критически влияли на стабильность через curriculum-переходы (epoch 6, 12)
# — они теперь в search вместо ручного подбора.
# Что НЕ в search:
#   batch_size / grad_accum_steps: упираются в VRAM, инженерный параметр.
SEARCH_SPACE = {
    "learning_rate":   {"low": 3e-5,  "high": 1.5e-3, "log": True},
    "weight_decay":    {"low": 1e-5,  "high": 1e-1,   "log": True},
    "dropout":         {"low": 0.05,  "high": 0.40,   "log": False},
    "label_smoothing": {"low": 0.0,   "high": 0.15,   "log": False},
    "warmup_steps":    {"low": 1500,  "high": 20000,  "log": True},
    "grad_clip_norm":  {"low": 0.3,   "high": 2.0,    "log": True},
}


def _sample_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate":   trial.suggest_float("learning_rate",   3e-5, 1.5e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",    1e-5, 1e-1,   log=True),
        "dropout":         trial.suggest_float("dropout",         0.05, 0.40),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0,  0.15),
        "warmup_steps":    trial.suggest_int(  "warmup_steps",    1500, 20000,  log=True),
        "grad_clip_norm":  trial.suggest_float("grad_clip_norm",  0.3,  2.0,    log=True),
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
DIVERGENCE_WARMUP_EPOCHS = 3   # первые N эпох не проверяем (warmup может занимать ~2 эпохи)


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
            if config.use_compile and sys.platform == "win32":
                print(f"  WARNING: use_compile=True на Windows — Triton поддерживается плохо")

        if config.use_compile:
            model = _compile_model(model, config)

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

        # Scheduler total_steps РАСЦЕПЛЕНЫ от длины трейла.
        # Если args.schedule_epochs задан → scheduler считает что горизонт
        # такой (например 40), а трейл реально гоняет args.epochs (например 8).
        # Это даёт трейлу LR-режим первых args.epochs эпох реального прогона
        # вместо сжатого cosine, где LR быстро остывает.
        schedule_epochs = args.schedule_epochs or args.epochs
        scheduler_total_steps = max(1, steps_per_epoch * schedule_epochs)
        trial_total_steps     = max(1, steps_per_epoch * args.epochs)

        # Warmup: если задан schedule_epochs — используем config.warmup_steps
        # как есть (он рассчитан на real-horizon). Иначе fallback на старую
        # логику для backward-compat.
        if args.schedule_epochs:
            trial_warmup = config.warmup_steps
        elif config.warmup_steps <= trial_total_steps * 0.5:
            trial_warmup = config.warmup_steps
        else:
            trial_warmup = max(30, int(trial_total_steps * 0.15))

        if trial.number == 0:
            print(f"  Trial: epochs={args.epochs} steps={trial_total_steps}  "
                  f"Scheduler horizon: epochs={schedule_epochs} steps={scheduler_total_steps}  "
                  f"warmup={trial_warmup}")

        optimizer = AdamW(model.parameters(),
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay)
        scheduler = make_lr_scheduler(optimizer, trial_warmup, scheduler_total_steps)
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

        # КРИТИЧНО для трансфера параметров: curriculum привязан к schedule_epochs
        # (горизонту реального прогона), не к args.epochs трейла. Иначе трейл
        # видит curriculum-переходы в other absolute эпохах чем реальный прогон,
        # и параметры найденные на сжатом curriculum плохо переносятся на 40 эпох.
        # При schedule_epochs=40, epochs=18 — трейл проживает первые 18 эпох
        # реального прогона: max_length 200 (1-6) → 280 (7-11) → 350 (12-18).
        curriculum_total = schedule_epochs
        for epoch in range(args.epochs):
            t0 = time.time()
            apply_curriculum(config, train_loader, stage=1,
                             epoch=epoch, total_epochs=curriculum_total)

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
                n_em_batches=args.n_em_batches,
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
        # Готовая команда для train.py: ВСЕ 6 гиперпараметров через CLI.
        # train.py теперь принимает все шесть через флаги, не нужно править config.
        p = best.params
        cmd = (
            f"python train.py --profile {args.profile}"
            f" --lr {p['learning_rate']:.6g}"
            f" --weight-decay {p['weight_decay']:.6g}"
            f" --dropout {p['dropout']:.4g}"
            f" --label-smoothing {p['label_smoothing']:.4g}"
            f" --warmup-steps {int(p['warmup_steps'])}"
            f" --grad-clip-norm {p['grad_clip_norm']:.4g}"
        )
        best_block = {
            "number":      best.number,
            "value":       best.value,
            "params":      best.params,
            "train_cmd":   cmd,
            "note":        "На Linux/сервере добавь --use-compile --compile-mode max-autotune. "
                           "На Windows/ноутбуке — --no-use-compile (или дефолт config).",
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
            "schedule_epochs":   args.schedule_epochs,
            "n_em_batches":      args.n_em_batches,
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
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Сколько трейлов. 6 dims в search → 100 для уверенной "
                             "сходимости TPE и низкой вероятности что top-5 — везение. "
                             "На RTX 5090 с compile ~60-70ч с pruning.")
    parser.add_argument("--epochs", type=int, default=18,
                        help="Эпох в одном трейле. 18 эпох при schedule-epochs=40 "
                             "покрывают оба curriculum-перехода (200→280 на epoch 6, "
                             "280→350 на epoch 12) — критично для отсева LR, "
                             "которые ломаются на сложных данных. 15 минимум, "
                             "20 даёт чуть больше хвоста на max_length=350.")
    parser.add_argument("--schedule-epochs", type=int, default=40,
                        help="Горизонт LR-scheduler в эпохах")
    parser.add_argument("--n-em-batches", type=int, default=16,
                        help="Сколько val-батчей идёт в EM-метрику внутри трейла. "
                             "При val_batch_size=96 это ~1536 примеров — достаточно "
                             "для устойчивой оценки EM. Раньше было 60 при val_batch=24 "
                             "(те же 1440 примеров). Если увеличиваешь val_batch_size — "
                             "уменьшай n_em_batches пропорционально.")
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
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="override config.val_batch_size. Ускоряет EM (greedy decode) "
                             "в конце эпохи. 2-3× train-batch обычно безопасно (нет градиентов).")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=None,
                        help="Override config.use_compile. Дефолт — config (True). "
                             "Используй --no-use-compile на Windows.")

    args = parser.parse_args()

    _set_seed(args.seed)

    base_overrides: dict = {}
    if args.batch_size is not None:      base_overrides["batch_size"]     = args.batch_size
    if args.val_batch_size is not None:  base_overrides["val_batch_size"] = args.val_batch_size
    if args.num_workers is not None: base_overrides["num_workers"] = args.num_workers
    if args.max_width is not None:   base_overrides["max_width"]   = args.max_width
    if args.use_compile is not None: base_overrides["use_compile"] = args.use_compile

    # Девайс и токенайзер строятся один раз — переиспользуются между трейлами.
    probe = load_config(args.profile, **base_overrides)
    device = torch.device(probe.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,  # на 6-dim search нужно больше "якорей" для медианы
        n_warmup_steps=5,     # не прунить до epoch 6 — warmup может занимать ~2-3 эпохи
                              # при больших warmup_steps, плюс не убивать трейл до того
                              # как он увидит первый curriculum-переход (epoch 6 при schedule=40)
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
