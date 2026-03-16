"""Inference (prediction) script for notes2latex-ocr.

Usage
-----
    python -m src.predict \\
        --checkpoint checkpoints/best_model.pt \\
        --vocab_file  data/vocab.txt \\
        --image       path/to/image.png \\
        [--image_height 128] [--image_width 512]

The script prints the predicted LaTeX string to stdout.
"""

from __future__ import annotations

import argparse
import sys

import torch
from PIL import Image

from src.data.dataset import CharTokenizer, build_transforms
from src.model.model import Pix2Tex


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def predict(
    model: Pix2Tex,
    tokenizer: CharTokenizer,
    image_path: str,
    image_height: int = 128,
    image_width: int = 512,
    device: torch.device | None = None,
    max_new_tokens: int = 512,
) -> str:
    """Return the predicted LaTeX string for a single image.

    Parameters
    ----------
    model:
        A loaded :class:`~src.model.model.Pix2Tex` model in eval mode.
    tokenizer:
        The :class:`~src.data.dataset.CharTokenizer` used during training.
    image_path:
        Path to the input image file.
    image_height:
        Target image height (must match the model's configuration).
    image_width:
        Target image width (must match the model's configuration).
    device:
        Torch device.  Defaults to ``cuda`` when available, else ``cpu``.
    max_new_tokens:
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Predicted LaTeX string (special tokens stripped).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_transforms(image_height, image_width, augment=False)
    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        token_ids = model.generate(image_tensor, max_new_tokens=max_new_tokens)

    ids = token_ids[0].tolist()
    return tokenizer.decode(ids, skip_special=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict LaTeX from a handwritten image")
    p.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pt)")
    p.add_argument("--vocab_file", required=True, help="Vocab file used during training")
    p.add_argument("--image", required=True, help="Path to the input image")
    p.add_argument("--image_height", type=int, default=128)
    p.add_argument("--image_width", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--decoder_depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto-detected if omitted)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load vocab
    with open(args.vocab_file, encoding="utf-8") as fh:
        vocab = [line.rstrip("\n") for line in fh if line.strip()]
    tokenizer = CharTokenizer(vocab)

    # Load checkpoint
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    model = Pix2Tex(
        vocab_size=checkpoint.get("vocab_size", tokenizer.vocab_size),
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        max_seq_len=args.max_seq_len,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    result = predict(
        model,
        tokenizer,
        args.image,
        image_height=args.image_height,
        image_width=args.image_width,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    print(result)


if __name__ == "__main__":
    main()
