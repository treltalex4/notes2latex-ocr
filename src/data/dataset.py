"""Dataset and data-loading utilities for the notes2latex-ocr project.

Each sample consists of:
  - A grayscale image of a handwritten expression (or line of text).
  - The corresponding LaTeX string.

The dataset expects a plain-text manifest file where every line has the form::

    /path/to/image.png\t\\frac{1}{2} + x^2

Special tokens
--------------
PAD  : 0  – padding
BOS  : 1  – beginning of sequence
EOS  : 2  – end of sequence
UNK  : 3  – unknown token
(user-defined tokens start at 4)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Special token IDs (must be consistent with Pix2Tex defaults)
# ---------------------------------------------------------------------------

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def build_transforms(
    image_height: int = 128,
    image_width: int = 512,
    augment: bool = False,
) -> Callable:
    """Return a ``torchvision`` transform pipeline for input images.

    Parameters
    ----------
    image_height:
        Target image height in pixels.
    image_width:
        Target image width in pixels.
    augment:
        When ``True``, apply light random augmentation suitable for training.

    Returns
    -------
    Callable
        A composed transform that accepts a PIL image and returns a
        ``(1, H, W)`` float tensor normalised to ``[0, 1]``.
    """
    ops: list = [
        T.Grayscale(num_output_channels=1),
        T.Resize((image_height, image_width)),
    ]
    if augment:
        ops += [
            T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        T.ToTensor(),  # → float32 in [0, 1]
    ]
    return T.Compose(ops)


# ---------------------------------------------------------------------------
# Simple character-level tokeniser
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Minimal character-level tokeniser for LaTeX strings.

    Parameters
    ----------
    vocab:
        Ordered list of characters/tokens that form the vocabulary.  Special
        tokens are prepended automatically if not already present.
    """

    def __init__(self, vocab: list[str] | None = None) -> None:
        base = list(SPECIAL_TOKENS)
        if vocab:
            for tok in vocab:
                if tok not in base:
                    base.append(tok)
        self._tok2id: dict[str, int] = {tok: i for i, tok in enumerate(base)}
        self._id2tok: dict[int, str] = {i: tok for tok, i in self._tok2id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._tok2id)

    def encode(self, text: str) -> list[int]:
        """Encode a LaTeX string into a list of token IDs (no BOS/EOS added)."""
        return [self._tok2id.get(ch, UNK_TOKEN_ID) for ch in text]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode a list of token IDs back to a LaTeX string."""
        tokens = [self._id2tok.get(i, "<unk>") for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return "".join(tokens)

    def encode_with_special(self, text: str, max_len: int | None = None) -> list[int]:
        """Encode with BOS prepended and EOS appended, padded to *max_len*."""
        ids = [BOS_TOKEN_ID] + self.encode(text) + [EOS_TOKEN_ID]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len - 1] + [EOS_TOKEN_ID]
            else:
                ids += [PAD_TOKEN_ID] * (max_len - len(ids))
        return ids

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        """Build a tokeniser from a list of LaTeX strings."""
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        return cls(sorted(chars))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LatexDataset(Dataset):
    """Dataset of (image, LaTeX) pairs loaded from a manifest file.

    Manifest format (tab-separated, one sample per line)::

        /absolute/path/to/image.png\t\\frac{1}{2}

    Parameters
    ----------
    manifest_path:
        Path to the manifest ``.tsv`` file.
    tokenizer:
        A :class:`CharTokenizer` instance.
    transform:
        Image transform (see :func:`build_transforms`).
    max_seq_len:
        Maximum token-sequence length (longer sequences are truncated).
    """

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        tokenizer: CharTokenizer,
        transform: Callable | None = None,
        max_seq_len: int = 512,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.transform = transform or build_transforms()
        self.max_seq_len = max_seq_len

        self.samples: list[tuple[str, str]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        with open(self.manifest_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"{self.manifest_path}:{lineno}: expected "
                        f"<image_path>\\t<latex>, got: {line!r}"
                    )
                img_path, latex = parts
                self.samples.append((img_path.strip(), latex.strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, latex = self.samples[idx]

        image = Image.open(img_path).convert("L")
        image = self.transform(image)  # (1, H, W)

        token_ids = self.tokenizer.encode_with_special(latex, max_len=self.max_seq_len)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        return {"image": image, "tokens": tokens}
