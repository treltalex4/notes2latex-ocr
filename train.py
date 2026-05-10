import argparse
import math
import os
import time

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
def greedy_decode_batch(model, images, src_kpm, tokenizer, device, max_len=128):
    # жадная декодировка батча → list[str].
    model.eval()
    B = images.shape[0]
    memory, memory_kpm = model.encoder(images, src_key_padding_mask=src_kpm)

    generated = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = model.decoder(generated, memory, memory_key_padding_mask=memory_kpm)
        next_ids = logits[:, -1, :].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_ID), next_ids)
        generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
        finished = finished | (next_ids == EOS_ID)
        if finished.all():
            break

    return [tokenizer.decode(generated[i].tolist()) for i in range(B)]


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, tokenizer,
                    device, log_every, limit_batches, grad_clip_norm, accum_steps,
                    use_amp, amp_dtype):
    model.train()
    losses, accs = [], []

    optimizer.zero_grad()
    for batch_idx, (images, src_kpm, tgt_ids) in enumerate(loader):
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

        if limit_batches and batch_idx + 1 >= limit_batches:
            break

    return sum(losses) / len(losses), sum(accs) / len(accs)


@torch.no_grad()
def validate(model, loader, criterion, tokenizer, device,
             use_amp, amp_dtype,
             n_em_batches=2, limit_batches=None):
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
            predicted  = greedy_decode_batch(model, images, src_kpm, tokenizer, device)
            references = [tokenizer.decode(ids.tolist()) for ids in tgt_ids]
            em_predictions.extend(predicted)
            em_references.extend(references)

        if limit_batches and batch_idx + 1 >= limit_batches:
            break

    em = exact_match(em_predictions, em_references) if em_predictions else 0.0
    return sum(losses) / len(losses), sum(accs) / len(accs), em


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="rtx4060_8gb")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--val-limit-batches", type=int, default=None)
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="override config.warmup_steps (useful for debug runs)")


    #for intel
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)

    args = parser.parse_args()

    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.max_width is not None:
        overrides["max_width"] = args.max_width

    config = load_config(args.profile, **overrides)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
    print(f"Параметров: {count_parameters(model):,}")

    train_loader, val_loader, _ = build_multi_dataloaders(config, tokenizer, stage=1)
    print(f"Batches: train={len(train_loader)} val={len(val_loader)}")

    lr = args.lr if args.lr is not None else config.learning_rate
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # AMP setup. На CPU autocast/scaler работают как no-op (enabled=False).
    # GradScaler нужен только для float16 — у bfloat16 диапазон как у fp32,
    # underflow градиентов не происходит.
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    use_amp = config.use_amp and device.type == "cuda"
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    print(f"AMP: enabled={use_amp} dtype={config.amp_dtype} scaler={scaler_enabled}")

    # LR scheduler: warmup + cosine annealing.
    # total_steps считается в эффективных шагах (после grad accumulation),
    # scheduler.step() вызывается раз в accum_steps батчей.
    accum_steps = config.grad_accum_steps
    batches_per_epoch = len(train_loader)
    if args.limit_batches:
        batches_per_epoch = min(batches_per_epoch, args.limit_batches)
    steps_per_epoch = batches_per_epoch // accum_steps
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = args.warmup_steps if args.warmup_steps is not None else config.warmup_steps
    scheduler = make_lr_scheduler(optimizer, warmup, total_steps)
    print(f"Scheduler: warmup={warmup} total_steps={total_steps} "
          f"(accum_steps={accum_steps}, effective_batch={config.batch_size * accum_steps})")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    stage = 1  # пока хардкод; шаг 3.7 сделает multi-stage

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        curriculum = apply_curriculum(config, train_loader, stage, epoch, args.epochs)
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

        val_loss, val_acc, val_em = validate(
            model, val_loader, criterion, tokenizer, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
            n_em_batches=2, limit_batches=args.val_limit_batches,
        )

        dt = time.time() - t0
        print(f"\nEpoch {epoch + 1} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_em={val_em:.3f} | "
              f"{dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(config.checkpoint_dir, "best_pretrain.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_em": val_em,
                "vocab_size": tokenizer.vocab_size,
            }, ckpt_path)
            print(f"  → saved best to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  no improvement for {epochs_no_improve}/{config.patience} epochs "
                  f"(best val_loss={best_val_loss:.4f})")
            if epochs_no_improve >= config.patience:
                print(f"\nEarly stopping: val_loss не улучшался {config.patience} эпох. "
                      f"Лучший val_loss={best_val_loss:.4f}")
                break


if __name__ == "__main__":
    main()
