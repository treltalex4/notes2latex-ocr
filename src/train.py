"""Training script for notes2latex-ocr.

Usage
-----
    python -m src.train \\
        --train_manifest data/train.tsv \\
        --val_manifest   data/val.tsv   \\
        --vocab_file     data/vocab.txt \\
        --output_dir     checkpoints/   \\
        [--epochs 30] [--batch_size 32] [--lr 3e-4] ...

Manifest format (tab-separated)::

    /path/to/image.png\t\\frac{1}{2} + x^2

Vocab file: one token per line (characters / LaTeX tokens).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import LatexDataset, CharTokenizer, build_transforms
from src.model.model import Pix2Tex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_tokenizer(vocab_file: str | None, train_manifest: str) -> CharTokenizer:
    """Load a CharTokenizer from a vocab file, or build one from the manifest."""
    if vocab_file and os.path.isfile(vocab_file):
        with open(vocab_file, encoding="utf-8") as fh:
            vocab = [line.rstrip("\n") for line in fh if line.strip()]
        return CharTokenizer(vocab)

    # Build from training data
    texts: list[str] = []
    with open(train_manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                texts.append(parts[1].strip())
    return CharTokenizer.from_texts(texts)


def _collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    images = torch.stack([s["image"] for s in batch])
    tokens = torch.stack([s["tokens"] for s in batch])
    return {"image": images, "tokens": tokens}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Tokenizer ---
    tokenizer = _load_tokenizer(args.vocab_file, args.train_manifest)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # --- Datasets & loaders ---
    train_transform = build_transforms(args.image_height, args.image_width, augment=True)
    val_transform = build_transforms(args.image_height, args.image_width, augment=False)

    train_ds = LatexDataset(
        args.train_manifest, tokenizer, train_transform, args.max_seq_len
    )
    val_ds = LatexDataset(
        args.val_manifest, tokenizer, val_transform, args.max_seq_len
    ) if args.val_manifest else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate_fn,
            pin_memory=device.type == "cuda",
        )
        if val_ds
        else None
    )

    # --- Model ---
    model = Pix2Tex(
        vocab_size=tokenizer.vocab_size,
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        in_channels=1,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(images, tokens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()
        elapsed = time.time() - t0

        # ---- validate ----
        val_loss_str = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    tokens = batch["tokens"].to(device)
                    val_loss += model.compute_loss(images, tokens).item()
            val_loss /= len(val_loader)
            val_loss_str = f"  val_loss={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = output_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "vocab_size": tokenizer.vocab_size,
                    },
                    ckpt_path,
                )

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}{val_loss_str}  "
            f"time={elapsed:.1f}s"
        )

    # Always save final checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": tokenizer.vocab_size,
        },
        output_dir / "final_model.pt",
    )
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the notes2latex-ocr model")

    # Data
    p.add_argument("--train_manifest", required=True, help="Path to training manifest TSV")
    p.add_argument("--val_manifest", default=None, help="Path to validation manifest TSV")
    p.add_argument("--vocab_file", default=None, help="Optional vocab file (one token/char per line)")
    p.add_argument("--output_dir", default="checkpoints", help="Directory for saved checkpoints")

    # Image
    p.add_argument("--image_height", type=int, default=128)
    p.add_argument("--image_width", type=int, default=512)

    # Model architecture
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--decoder_depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=512)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
