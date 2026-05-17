import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset

from config import load_config
from data.dataset import build_multi_dataloaders
from data.tokenizer import LaTeXTokenizer, PAD_ID, SOS_ID, EOS_ID
from model.model import Notes2LaTeX, count_parameters
from utils.metrics import token_accuracy, exact_match
from utils.schedules import (
    get_augment_strength, get_elastic_params, get_max_length,
)


def _auto_run_name(args, config, base_lr: float, stage_name: str) -> str:
    """Генерирует имя прогона из ключевых гиперпараметров + timestamp.
    Пример: s1_lr1e-03_wd0.01_dr0.1_ls0.0_seed42_20260511-034500"""
    seed_str = f"_seed{args.seed}" if args.seed is not None else ""
    ts = time.strftime("%Y%m%d-%H%M%S")
    return (
        f"s{1 if stage_name == 'pretrain' else 2}_"
        f"lr{base_lr:.0e}_"
        f"wd{config.weight_decay:g}_"
        f"dr{config.dropout:g}_"
        f"ls{config.label_smoothing:g}"
        f"{seed_str}_{ts}"
    )


def _save_history(history: dict, path: str) -> None:
    """Сохраняет историю целиком, перезаписывая файл. Атомарность через
    write-to-tmp + rename: при крэше в момент записи не получишь обрезанный JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


_STAGE_NAMES = {1: "pretrain", 2: "mixed"}


def _unwrap_compiled(model):
    """Возвращает оригинальную модель если она обёрнута torch.compile.
    Нужно для save/load state_dict — чекпоинты должны быть без _orig_mod префикса,
    чтобы работать кроссплатформенно (Windows без compile / Linux с compile)."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _compile_model(model, config):
    """Обёртка torch.compile с настройками против recompile-thrashing.

    BucketBatchSampler даёт батчи разной ширины (~20-50 уникальных значений
    за прогон). Без явных настроек dynamo специализируется на первых N формах,
    при превышении cache_size_limit (default 8) сдаётся и падает в eager —
    отсюда катастрофическая просадка скорости. Лечим двумя ручками:
      - cache_size_limit=256: запас на все возможные размеры батча
      - dynamic=True (уже было): подсказка dynamo сразу строить
        дин-shape-graph вместо специализации.
    """
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
    if sys.platform == "win32":
        print(f"WARNING: use_compile=True на Windows — Triton поддерживается плохо, "
              f"возможны падения или отсутствие ускорения. Рекомендуется False.")
    print(f"torch.compile: mode={config.compile_mode} dynamic=True "
          f"cache_size_limit=256 (первая эпоха медленнее из-за компиляции)")
    return torch.compile(model, mode=config.compile_mode, dynamic=True)


def _make_checkpoint(stage, stage_name, epoch, model, optimizer, scheduler, scaler,
                     best_val_loss, epochs_no_improve, history_path, vocab_size,
                     val_loss, val_acc, val_em, interrupted=False) -> dict:
    """Полный снимок состояния для resume — сохраняется в best/last/interrupt.

    `epoch`: для last/best это завершённая эпоха (resume стартует с epoch+1),
    для interrupt — эпоха в процессе (resume перезапускает её целиком).
    """
    return {
        "stage": stage,
        "stage_name": stage_name,
        "epoch": epoch,
        "interrupted": interrupted,
        "model_state_dict": _unwrap_compiled(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "history_path": history_path,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_em": val_em,
        "vocab_size": vocab_size,
    }


def _resolve_resume_path(args, config) -> str | None:
    """Резолвит чекпоинт для resume: явный --resume-from приоритетнее, иначе
    --resume-mode подставляет checkpoints/<mode>_<stage>.pth для первой
    стадии из --stages."""
    if args.resume_from:
        return args.resume_from
    if args.resume_mode:
        stage_name = _STAGE_NAMES[args.stages[0]]
        return os.path.join(config.checkpoint_dir, f"{args.resume_mode}_{stage_name}.pth")
    return None


def _train_datasets(loader):
    """Возвращает список tail-датасетов вне зависимости от того, обернул
    DataLoader Im2LatexDataset напрямую или через ConcatDataset (stage 2/3)."""
    ds = loader.dataset
    if isinstance(ds, ConcatDataset):
        return list(ds.datasets)
    return [ds]


def apply_curriculum(config, train_loader, stage: int, epoch: int, total_epochs: int):
    """Обновляет augmentation/length параметры в train-датасетах перед эпохой.

    Возвращает dict с применёнными значениями — для логирования.
    """
    schedule_map = {
        1: config.elastic_schedule_stage1,
        2: config.elastic_schedule_stage2,
        3: config.elastic_schedule_stage3,
    }
    elastic_schedule = schedule_map[stage]

    p, alpha, sigma = get_elastic_params(epoch, total_epochs, elastic_schedule)
    strength = get_augment_strength(epoch, total_epochs, config.augment_strength_max)

    for ds in _train_datasets(train_loader):
        ds.elastic_p     = p
        ds.elastic_alpha = alpha
        ds.elastic_sigma = sigma
        ds.strength      = strength

    # Length curriculum — только для stage 1 (BucketBatchSampler).
    current_max = None
    if stage == 1:
        current_max = get_max_length(epoch, total_epochs, config.length_curriculum_stage1)
        train_loader.batch_sampler.current_max_length = current_max

    return {"elastic_p": p, "elastic_alpha": alpha, "elastic_sigma": sigma,
            "strength": strength, "max_length": current_max}


def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay to 0.

    Возвращает LambdaLR: lr_lambda(step) — множитель базового LR в [0, 1].
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def greedy_decode_batch(model, images, src_kpm, tokenizer, device, max_len=600):
    """Жадный декод с KV-кэшем в self-attn декодера.

    O(T) по compute и памяти на шаг вместо O(T²) — каждый шаг считает Q только
    для нового токена, K/V для предыдущих позиций берутся из кэша.
    """
    model.eval()
    B = images.shape[0]
    memory, memory_kpm = model.encoder(images, src_key_padding_mask=src_kpm)

    # Первый шаг: пропускаем SOS через декодер, инициализируем кэш.
    cur_token = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
    tokens: list[torch.Tensor] = [cur_token]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    kv_caches = None

    for step in range(max_len - 1):
        logits, kv_caches = model.decoder.forward_step(
            cur_token, memory,
            memory_key_padding_mask=memory_kpm,
            kv_caches=kv_caches,
            position_offset=step,
        )
        next_ids = logits[:, -1, :].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_ID), next_ids)
        cur_token = next_ids.unsqueeze(1)
        tokens.append(cur_token)
        finished = finished | (next_ids == EOS_ID)
        if finished.all():
            break

    generated = torch.cat(tokens, dim=1)
    return [tokenizer.decode(generated[i].tolist()) for i in range(B)]


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, tokenizer,
                    device, log_every, limit_batches, grad_clip_norm, accum_steps,
                    use_amp, amp_dtype):
    model.train()
    losses, accs = [], []

    # Профайлер data-loader bottleneck: data_wait — время, проведённое в
    # ожидании следующего батча от воркеров (если велико → loader не успевает
    # за GPU). compute — время forward+backward+step. Печатается в конце эпохи.
    data_wait, compute = 0.0, 0.0
    t_mark = time.time()

    optimizer.zero_grad()
    for batch_idx, (images, src_kpm, tgt_ids) in enumerate(loader):
        data_wait += time.time() - t_mark
        t_compute = time.time()

        images  = images.to(device)
        src_kpm = src_kpm.to(device)
        tgt_ids = tgt_ids.to(device)

        tgt_input  = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]

        # Forward — внутри autocast все операции автоматически в half precision.
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images, tgt_input, src_key_padding_mask=src_kpm)
            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                tgt_output.reshape(-1),
            )

        # scaler.scale: умножает loss на scale factor, чтобы при backward
        # маленькие градиенты не уходили в 0 при fp16. При enabled=False —
        # просто пропускает через себя.
        scaler.scale(loss / accum_steps).backward()

        # Эффективный шаг оптимизатора — раз в accum_steps батчей.
        if (batch_idx + 1) % accum_steps == 0:
            # unscale: вернуть градиенты в реальный масштаб ПЕРЕД clip_grad_norm,
            # иначе clipping будет на масштабированных значениях и обрежет неправильно.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            # scaler.step: если в градиентах обнаружены inf/nan (fp16 переполнение),
            # step пропускается и scale factor уменьшается. Иначе работает как обычный step.
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        accs.append(token_accuracy(logits, tgt_output, pad_idx=PAD_ID))

        if batch_idx % log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  step {batch_idx:4d} | loss={losses[-1]:.4f} | "
                  f"acc={accs[-1]:.3f} | lr={current_lr:.2e}")

        compute += time.time() - t_compute

        if limit_batches and batch_idx + 1 >= limit_batches:
            break

        t_mark = time.time()

    total = data_wait + compute
    if total > 0:
        print(f"  [profile] data_wait={data_wait:.1f}s compute={compute:.1f}s "
              f"→ {100 * data_wait / total:.0f}% waiting on data loader")

    return sum(losses) / len(losses), sum(accs) / len(accs)


@torch.no_grad()
def validate(model, loader, criterion, tokenizer, device,
             use_amp, amp_dtype,
             n_em_batches=2, limit_batches=None, greedy_max_len=600):
    model.eval()
    losses, accs = [], []
    em_predictions, em_references = [], []

    for batch_idx, (images, src_kpm, tgt_ids) in enumerate(loader):
        images  = images.to(device)
        src_kpm = src_kpm.to(device)
        tgt_ids = tgt_ids.to(device)

        tgt_input  = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images, tgt_input, src_key_padding_mask=src_kpm)
            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                tgt_output.reshape(-1),
            )
        losses.append(loss.item())
        accs.append(token_accuracy(logits, tgt_output, pad_idx=PAD_ID))

        if batch_idx < n_em_batches:
            predicted  = greedy_decode_batch(model, images, src_kpm, tokenizer, device, max_len=greedy_max_len)
            references = [tokenizer.decode(ids.tolist()) for ids in tgt_ids]
            em_predictions.extend(predicted)
            em_references.extend(references)

        if limit_batches and batch_idx + 1 >= limit_batches:
            break

    em = exact_match(em_predictions, em_references) if em_predictions else 0.0
    return sum(losses) / len(losses), sum(accs) / len(accs), em


def run_stage(model, config, args, tokenizer, device, scaler, use_amp, amp_dtype,
              stage: int, stage_name: str, epochs: int, lr_multiplier: float = 1.0,
              resume_state: dict | None = None):
    """Один этап обучения. Возвращает путь к лучшему чекпоинту.

    resume_state — снимок из _make_checkpoint для продолжения этой стадии:
    восстанавливает optimizer/scheduler/scaler, счётчики early stopping и
    дописывает в тот же history-файл.
    """
    print(f"\n{'=' * 60}")
    print(f"  STAGE {stage}: {stage_name}  ({epochs} epochs, lr×{lr_multiplier})")
    print(f"{'=' * 60}")

    train_loader, val_loader, _ = build_multi_dataloaders(config, tokenizer, stage=stage)
    print(f"Batches: train={len(train_loader)} val={len(val_loader)}")

    base_lr = config.learning_rate * lr_multiplier   # args.lr уже в config через overrides
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=config.label_smoothing,
    )

    accum_steps = config.grad_accum_steps
    batches_per_epoch = len(train_loader)
    if args.limit_batches:
        batches_per_epoch = min(batches_per_epoch, args.limit_batches)
    steps_per_epoch = batches_per_epoch // accum_steps
    # total_steps считается по ПОЛНОМУ числу эпох — при resume scheduler_state
    # восстанавливает позицию, а форма косинуса должна совпадать с оригиналом.
    # Поэтому --epochs-stage* при resume должен быть тем же, что в первом запуске.
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = config.warmup_steps   # args.warmup_steps уже в config через overrides
    scheduler = make_lr_scheduler(optimizer, warmup, total_steps)
    print(f"Scheduler: warmup={warmup} total_steps={total_steps} "
          f"(accum={accum_steps}, effective_batch={config.batch_size * accum_steps}, "
          f"lr_base={base_lr:.2e})")

    ckpt_path = os.path.join(config.checkpoint_dir, f"best_{stage_name}.pth")
    last_path = os.path.join(config.checkpoint_dir, f"last_{stage_name}.pth")

    # --- Resume: optimizer/scheduler/scaler + счётчики early stopping ---
    reset_schedule = getattr(args, "reset_schedule", False)
    full_resume = (resume_state is not None
                   and "optimizer_state_dict" in resume_state
                   and not reset_schedule)
    if reset_schedule and resume_state is not None:
        print("  --reset-schedule: optimizer/scheduler созданы заново под текущий config "
              f"(lr={base_lr:.2e}, warmup={warmup}); early stopping сброшен")
    if full_resume:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if resume_state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        best_val_loss     = resume_state.get("best_val_loss", float("inf"))
        epochs_no_improve = resume_state.get("epochs_no_improve", 0)
        # interrupt сохранён посреди эпохи → перезапускаем её целиком.
        # last/best сохранены на границе эпохи → стартуем со следующей.
        start_epoch = (resume_state["epoch"] if resume_state.get("interrupted")
                       else resume_state["epoch"] + 1)
        print(f"  RESUME: start_epoch={start_epoch + 1}/{epochs} "
              f"best_val_loss={best_val_loss:.4f} epochs_no_improve={epochs_no_improve}")
    else:
        if resume_state is not None:
            print("  WARNING: чекпоинт без optimizer_state — weights-only resume, "
                  "обучение стартует с epoch 1")
        best_val_loss     = float("inf")
        epochs_no_improve = 0
        start_epoch       = 0

    # --- History: дописываем существующий файл при resume, иначе создаём новый ---
    resume_history = resume_state.get("history_path") if full_resume else None
    if resume_history and os.path.exists(resume_history):
        history_path = resume_history
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        run_name = history["run_name"]
        # Отбрасываем записи эпох >= start_epoch (interrupt мог перезаписать эпоху).
        history["epochs"] = [e for e in history["epochs"] if e["epoch"] < start_epoch]
        history.pop("interrupted", None)
        history.setdefault("resumes", []).append(time.strftime("%Y-%m-%dT%H:%M:%S"))
    else:
        # Per-run JSON — отдельный файл на каждый эксперимент. Имя берётся из
        # args.run_name либо генерируется автоматически. Старые прогоны не
        # теряются — каждый пишется в свой файл в runs/.
        run_name = args.run_name or _auto_run_name(args, config, base_lr, stage_name)
        runs_dir = os.path.join(config.checkpoint_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        history_path = os.path.join(runs_dir, f"{run_name}.json")
        history = {
            "run_name":   run_name,
            "stage":      stage,
            "stage_name": stage_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hyperparams": {
                "lr":              base_lr,
                "weight_decay":    config.weight_decay,
                "dropout":         config.dropout,
                "label_smoothing": config.label_smoothing,
                "grad_clip_norm":  config.grad_clip_norm,
                "batch_size":      config.batch_size,
                "grad_accum":      config.grad_accum_steps,
                "warmup_steps":    warmup,
                "use_compile":     config.use_compile,
                "compile_mode":    config.compile_mode if config.use_compile else None,
                "limit_batches":   args.limit_batches,
                "epochs":          epochs,
                "seed":            args.seed,
            },
            "epochs": [],
        }
    print(f"Run name:  {run_name}")
    print(f"Logging:   {history_path}")

    epoch = start_epoch   # safety: если Ctrl+C попал до первой итерации

    try:
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            print(f"\n=== Epoch {epoch + 1}/{epochs} (stage {stage}: {stage_name}) ===")

            curriculum = apply_curriculum(config, train_loader, stage, epoch, epochs)
            print(f"  curriculum: elastic_p={curriculum['elastic_p']:.2f} "
                  f"alpha={curriculum['elastic_alpha']} sigma={curriculum['elastic_sigma']} "
                  f"strength={curriculum['strength']:.2f} "
                  f"max_length={curriculum['max_length']}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, criterion, tokenizer, device,
                log_every=args.log_every, limit_batches=args.limit_batches,
                grad_clip_norm=config.grad_clip_norm,
                accum_steps=accum_steps,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )

            # Возвращаем зарезервированную память пулом обратно в систему перед
            # validate — у greedy decode другой профиль аллокаций, и без сброса
            # PyTorch держит фрагментированный пул, что повышает риск OOM на
            # длинных val-батчах.
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
            print(f"\nEpoch {epoch + 1} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_em={val_em:.3f} | "
                  f"{dt:.1f}s")

            history["epochs"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "val_em":     val_em,
                "lr_end":     scheduler.get_last_lr()[0],
                "curriculum": curriculum,
                "time_seconds": dt,
            })
            _save_history(history, history_path)

            # Счётчик early stopping обновляем ДО сохранения чекпоинтов,
            # чтобы last_*.pth нёс актуальный epochs_no_improve.
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            ckpt = _make_checkpoint(
                stage, stage_name, epoch, model, optimizer, scheduler, scaler,
                best_val_loss, epochs_no_improve, history_path, tokenizer.vocab_size,
                val_loss, val_acc, val_em, interrupted=False,
            )
            # last_*.pth — каждую эпоху на границе. Чистая точка resume "продолжить".
            torch.save(ckpt, last_path)

            if is_best:
                torch.save(ckpt, ckpt_path)
                print(f"  → saved best to {ckpt_path}  (last → {last_path})")
            else:
                print(f"  saved last → {last_path}  | "
                      f"no improvement {epochs_no_improve}/{config.patience} "
                      f"(best val_loss={best_val_loss:.4f})")
                if epochs_no_improve >= config.patience:
                    print(f"\nEarly stopping: val_loss не улучшался {config.patience} эпох. "
                          f"Лучший val_loss={best_val_loss:.4f}")
                    break
    except KeyboardInterrupt:
        # Graceful shutdown: сохраняем текущее состояние (НЕ best_*.pth, чтобы
        # не затереть лучший чекпоинт) и пробрасываем исключение наверх.
        # interrupted=True → resume перезапустит эту эпоху целиком (scheduler
        # уйдёт чуть вперёд на доделанные шаги — допустимый дрейф для аварийной точки).
        # val_* = nan: Ctrl+C мог прийти до validate этой эпохи.
        interrupt_path = os.path.join(config.checkpoint_dir, f"interrupt_{stage_name}.pth")
        torch.save(
            _make_checkpoint(
                stage, stage_name, epoch, model, optimizer, scheduler, scaler,
                best_val_loss, epochs_no_improve, history_path, tokenizer.vocab_size,
                float("nan"), float("nan"), float("nan"), interrupted=True,
            ),
            interrupt_path,
        )
        print(f"\n[INTERRUPT] caught at epoch {epoch + 1}/{epochs}")
        print(f"  saved interrupt state -> {interrupt_path}")
        print(f"  best so far          -> {ckpt_path} (val_loss={best_val_loss:.4f})")
        print(f"  resume:  python train.py --resume-mode interrupt --stages {stage}")
        # Финализируем history даже при прерывании — данные не теряются.
        history["interrupted"] = True
        _finalize_history(history, history_path)
        raise

    _finalize_history(history, history_path)
    _print_run_summary(history)
    return ckpt_path


def _finalize_history(history: dict, path: str) -> None:
    """Добавляет финальный summary (best epoch, total time) и сохраняет."""
    epochs = history["epochs"]
    if epochs:
        best_idx = min(range(len(epochs)), key=lambda i: epochs[i]["val_loss"])
        history["best"] = {
            "epoch":    epochs[best_idx]["epoch"],
            "val_loss": epochs[best_idx]["val_loss"],
            "val_acc":  epochs[best_idx]["val_acc"],
            "val_em":   epochs[best_idx]["val_em"],
        }
        history["total_time_seconds"] = sum(e["time_seconds"] for e in epochs)
        history["n_epochs_done"] = len(epochs)
    _save_history(history, path)


def _print_run_summary(history: dict) -> None:
    """Финальная сводка по прогону — для удобного сравнения."""
    if "best" not in history:
        return
    b = history["best"]
    h = history["hyperparams"]
    print(f"\n{'─' * 60}")
    print(f"RUN SUMMARY: {history['run_name']}")
    print(f"{'─' * 60}")
    print(f"  lr={h['lr']:.2e}  wd={h['weight_decay']:g}  "
          f"dropout={h['dropout']:g}  label_smoothing={h['label_smoothing']:g}")
    print(f"  Best epoch {b['epoch'] + 1}: "
          f"val_loss={b['val_loss']:.4f}  val_acc={b['val_acc']:.3f}  val_em={b['val_em']:.3f}")
    print(f"  Total time: {history['total_time_seconds']:.0f}s "
          f"({history['n_epochs_done']} epochs)")
    print(f"  Compare:    python runs.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="rtx4060_8gb")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2],
                        choices=[1, 2],
                        help="Какие этапы обучения запустить (1=pretrain, 2=mixed)")
    parser.add_argument("--epochs-stage1", type=int, default=None,
                        help="override config.epochs_pretrain")
    parser.add_argument("--epochs-stage2", type=int, default=None,
                        help="override config.epochs_mixed")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="override config.weight_decay")
    parser.add_argument("--dropout", type=float, default=None,
                        help="override config.dropout (затронет архитектуру модели)")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="override config.label_smoothing")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--val-limit-batches", type=int, default=None,
                        help="Ограничить число val-батчей. Без флага = full val "
                             "(~600 батчей на im2latex). Учти: BucketBatchSampler "
                             "сортирует по длине, ограничение даст bias на короткие.")
    parser.add_argument("--n-em-batches", type=int, default=None,
                        help="Override config.n_em_batches (20 по дефолту). "
                             "Сколько val-батчей идёт в EM (greedy decode).")
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="override config.warmup_steps (useful for debug runs)")
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="override config.grad_clip_norm. Tune подбирает 0.3-2.0, "
                             "против дивергенции на curriculum-переходах.")
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=None,
                        help="override config.use_compile. Используй --use-compile на "
                             "Linux/сервере, --no-use-compile на Windows/ноутбуке.")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"],
                        default=None,
                        help="override config.compile_mode. max-autotune для долгих "
                             "прогонов (компиляция ~3-5мин, +5-10% к default).")
    parser.add_argument("--resume-from", default=None,
                        help="Путь к чекпоинту для полного resume (model + optimizer + "
                             "scheduler + scaler + счётчики early stopping). "
                             "Приоритетнее --resume-mode.")
    parser.add_argument("--resume-mode", choices=["last", "best", "interrupt"], default=None,
                        help="Shortcut вместо --resume-from: берёт "
                             "checkpoints/<mode>_<stage>.pth для первой стадии из --stages. "
                             "last=продолжить с границы эпохи; best=откатиться к лучшей "
                             "(обычно вместе со сменой гиперов, иначе воспроизведётся та же "
                             "траектория); interrupt=аварийная точка, перезапускает "
                             "прерванную эпоху.")
    parser.add_argument("--reset-schedule", action="store_true",
                        help="При resume: загрузить только веса модели, optimizer/scheduler "
                             "создать заново под текущий config (новый lr, warmup_steps). "
                             "Сбрасывает счётчик эпох и early stopping. Используй когда меняешь "
                             "lr/warmup и продолжаешь с best checkpoint.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config.seed. None в config = random.")
    parser.add_argument("--run-name", default=None,
                        help="Имя прогона для логирования в checkpoints/runs/<name>.json. "
                             "Если не указано — генерируется автоматически из гиперпараметров.")

    # для слабого железа
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="override config.grad_accum_steps. "
                             "Effective batch = batch_size * grad_accum_steps.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)

    args = parser.parse_args()

    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        overrides["grad_accum_steps"] = args.grad_accum_steps
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.max_width is not None:
        overrides["max_width"] = args.max_width
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.label_smoothing is not None:
        overrides["label_smoothing"] = args.label_smoothing
    # Унифицировано: lr/warmup/grad_clip тоже идут через overrides, чтобы config
    # объект отражал реально используемые значения (важно для history + run_name).
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.warmup_steps is not None:
        overrides["warmup_steps"] = args.warmup_steps
    if args.grad_clip_norm is not None:
        overrides["grad_clip_norm"] = args.grad_clip_norm
    if args.use_compile is not None:
        overrides["use_compile"] = args.use_compile
    if args.compile_mode is not None:
        overrides["compile_mode"] = args.compile_mode

    config = load_config(args.profile, **overrides)

    # Резолвим CLI ↔ config: CLI имеет приоритет, иначе берём из config.
    effective_seed = args.seed if args.seed is not None else config.seed
    if effective_seed is not None:
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
        print(f"Seed: {effective_seed}")
    args.seed = effective_seed  # для history записи

    if args.n_em_batches is None:
        args.n_em_batches = config.n_em_batches

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
    print(f"Параметров: {count_parameters(model):,}")

    if config.use_compile:
        model = _compile_model(model, config)

    resume_path = _resolve_resume_path(args, config)
    resume_state = None
    if resume_path is not None:
        if not os.path.exists(resume_path):
            print(f"ERROR: resume-чекпоинт не найден: {resume_path}")
            sys.exit(1)
        resume_state = torch.load(resume_path, map_location=device)
        _unwrap_compiled(model).load_state_dict(resume_state["model_state_dict"])
        kind = "interrupt (mid-epoch)" if resume_state.get("interrupted") else "epoch-boundary"
        bvl = resume_state.get("best_val_loss")
        print(f"Resume: {resume_path}")
        print(f"  stage={resume_state.get('stage')} epoch={resume_state.get('epoch')} "
              f"[{kind}]" + (f" best_val_loss={bvl:.4f}" if bvl is not None else ""))

    # AMP setup общий для всех стадий.
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    use_amp = config.use_amp and device.type == "cuda"
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    print(f"AMP: enabled={use_amp} dtype={config.amp_dtype} scaler={scaler_enabled}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Параметры каждой стадии: epochs и LR multiplier.
    # Stage 2 идёт с LR×0.5 — модель уже знает структуру, нужны мелкие подстройки.
    stage_configs = {
        1: {
            "name": "pretrain",
            "epochs": args.epochs_stage1 if args.epochs_stage1 is not None else config.epochs_pretrain,
            "lr_multiplier": 1.0,
        },
        2: {
            "name": "mixed",
            "epochs": args.epochs_stage2 if args.epochs_stage2 is not None else config.epochs_mixed,
            "lr_multiplier": 0.5,
        },
    }

    last_ckpt = None
    try:
        for stage in args.stages:
            cfg = stage_configs[stage]

            # Resume применяется только к стадии, на которой прервались.
            stage_resume = None
            if resume_state is not None:
                rs_stage = resume_state.get("stage")
                if rs_stage == stage:
                    stage_resume = resume_state
                elif rs_stage is not None and stage < rs_stage:
                    print(f"\nStage {stage} завершён до прерывания — пропускаем")
                    continue

            # Между стадиями: загружаем лучший чекпоинт предыдущей в модель,
            # чтобы стартовать с best, а не с последнего epoch state.
            # Пропускаем, если резюмим саму эту стадию (веса уже загружены).
            if last_ckpt is not None and stage_resume is None:
                state = torch.load(last_ckpt, map_location=device)
                _unwrap_compiled(model).load_state_dict(state["model_state_dict"])
                print(f"\nLoaded best from previous stage: {last_ckpt}")

            last_ckpt = run_stage(
                model, config, args, tokenizer, device,
                scaler, use_amp, amp_dtype,
                stage=stage, stage_name=cfg["name"],
                epochs=cfg["epochs"], lr_multiplier=cfg["lr_multiplier"],
                resume_state=stage_resume,
            )
            resume_state = None  # consumed

            # Между стадиями сбрасываем CUDA-пул: новый train_loader, новые
            # оптимизатор/scheduler — старые тензоры держать незачем.
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        # run_stage уже сохранил interrupt_*.pth и напечатал детали.
        # Здесь просто выходим с кодом 130 (стандарт SIGINT) без traceback.
        print("\nTraining stopped by user.")
        sys.exit(130)

    print(f"\nFinished. Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
