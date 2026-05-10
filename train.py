import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import load_config
from data.dataset import build_multi_dataloaders
from data.tokenizer import LaTeXTokenizer, PAD_ID, SOS_ID, EOS_ID
from model.model import Notes2LaTeX, count_parameters
from utils.metrics import token_accuracy, exact_match


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


def train_one_epoch(model, loader, optimizer, criterion, tokenizer, device,
                    log_every, limit_batches):
    model.train()
    losses, accs = [], []

    for batch_idx, (images, src_kpm, tgt_ids) in enumerate(loader):
        images  = images.to(device)
        src_kpm = src_kpm.to(device)
        tgt_ids = tgt_ids.to(device)

        tgt_input  = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]

        logits = model(images, tgt_input, src_key_padding_mask=src_kpm)
        print(f"images={images.shape}  tgt_input={tgt_input.shape}  tgt_output={tgt_output.shape}  logits={logits.shape}  vocab={tokenizer.vocab_size}")

        loss = criterion(
            logits.reshape(-1, tokenizer.vocab_size),
            tgt_output.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(token_accuracy(logits, tgt_output, pad_idx=PAD_ID))

        if batch_idx % log_every == 0:
            print(f"  step {batch_idx:4d} | loss={losses[-1]:.4f} | acc={accs[-1]:.3f}")

        if limit_batches and batch_idx + 1 >= limit_batches:
            break

    return sum(losses) / len(losses), sum(accs) / len(accs)


@torch.no_grad()
def validate(model, loader, criterion, tokenizer, device,
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

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, tokenizer, device,
            log_every=args.log_every, limit_batches=args.limit_batches,
        )

        val_loss, val_acc, val_em = validate(
            model, val_loader, criterion, tokenizer, device,
            n_em_batches=2, limit_batches=args.val_limit_batches,
        )

        dt = time.time() - t0
        print(f"\nEpoch {epoch + 1} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_em={val_em:.3f} | "
              f"{dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(config.checkpoint_dir, "best_pretrain.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_em": val_em,
                "vocab_size": tokenizer.vocab_size,
            }, ckpt_path)
            print(f"  → saved best to {ckpt_path}")


if __name__ == "__main__":
    main()
