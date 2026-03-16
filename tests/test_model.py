"""Tests for the combined Pix2Tex model and the data utilities."""

import io
import os
import tempfile

import torch
import pytest
from PIL import Image

from src.model.model import Pix2Tex
from src.data.dataset import (
    CharTokenizer,
    LatexDataset,
    build_transforms,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 60
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
PATCH_SIZE = 16
EMBED_DIM = 64
NUM_HEADS = 4


def make_model(**kwargs) -> Pix2Tex:
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        patch_size=PATCH_SIZE,
        in_channels=1,
        embed_dim=EMBED_DIM,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=NUM_HEADS,
        dropout=0.0,
        max_seq_len=32,
    )
    defaults.update(kwargs)
    return Pix2Tex(**defaults)


# ---------------------------------------------------------------------------
# Pix2Tex model tests
# ---------------------------------------------------------------------------


class TestPix2Tex:
    def test_forward_output_shape(self):
        model = make_model()
        images = torch.randn(2, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        tgt = torch.randint(1, VOCAB_SIZE, (2, 8))
        logits = model(images, tgt)
        assert logits.shape == (2, 8, VOCAB_SIZE)

    def test_compute_loss_scalar(self):
        model = make_model()
        images = torch.randn(2, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        # Full sequence: BOS + tokens + EOS, at least length 3
        tgt = torch.randint(1, VOCAB_SIZE, (2, 10))
        tgt[:, 0] = BOS_TOKEN_ID
        tgt[:, -1] = EOS_TOKEN_ID
        loss = model.compute_loss(images, tgt)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_loss_decreases_with_overfitting(self):
        """A small model should be able to overfit a single batch."""
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        images = torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        tgt = torch.tensor([[BOS_TOKEN_ID, 5, 6, 7, EOS_TOKEN_ID]])

        initial_loss = model.compute_loss(images, tgt).item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = model.compute_loss(images, tgt)
            loss.backward()
            optimizer.step()

        final_loss = model.compute_loss(images, tgt).item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_generate_output_shape(self):
        model = make_model()
        model.eval()
        images = torch.randn(2, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        with torch.no_grad():
            tokens = model.generate(images, max_new_tokens=10)
        assert tokens.shape[0] == 2
        assert tokens.shape[1] <= 10 + 1  # BOS counts

    def test_generate_starts_with_bos(self):
        model = make_model()
        model.eval()
        images = torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        with torch.no_grad():
            tokens = model.generate(images, max_new_tokens=5)
        assert tokens[0, 0].item() == BOS_TOKEN_ID

    def test_pad_positions_ignored_in_loss(self):
        """Padding tokens in the target should not contribute to the loss."""
        model = make_model()
        images = torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        # seq with no padding
        tgt_short = torch.tensor([[BOS_TOKEN_ID, 5, 6, EOS_TOKEN_ID]])
        # same seq padded to length 8
        tgt_padded = torch.tensor(
            [[BOS_TOKEN_ID, 5, 6, EOS_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]]
        )
        with torch.no_grad():
            loss_short = model.compute_loss(images, tgt_short)
            loss_padded = model.compute_loss(images, tgt_padded)
        # Losses should be equal since padding is ignored.
        assert torch.isclose(loss_short, loss_padded, atol=1e-5), (
            f"loss_short={loss_short.item():.6f}, loss_padded={loss_padded.item():.6f}"
        )


# ---------------------------------------------------------------------------
# CharTokenizer tests
# ---------------------------------------------------------------------------


class TestCharTokenizer:
    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer(["a", "b", "c"])
        text = "abc"
        ids = tok.encode(text)
        recovered = tok.decode(ids, skip_special=False)
        assert recovered == text

    def test_unknown_char(self):
        tok = CharTokenizer(["a"])
        ids = tok.encode("z")
        assert ids == [3]  # UNK_TOKEN_ID

    def test_encode_with_special_includes_bos_eos(self):
        tok = CharTokenizer(["a", "b"])
        ids = tok.encode_with_special("ab")
        assert ids[0] == BOS_TOKEN_ID
        assert ids[-1] == EOS_TOKEN_ID

    def test_encode_with_special_pads_to_max_len(self):
        tok = CharTokenizer(["a"])
        ids = tok.encode_with_special("a", max_len=10)
        assert len(ids) == 10
        assert ids[-1] == PAD_TOKEN_ID

    def test_encode_with_special_truncates(self):
        tok = CharTokenizer(list("abcdefgh"))
        ids = tok.encode_with_special("abcdefgh", max_len=5)
        assert len(ids) == 5
        assert ids[-1] == EOS_TOKEN_ID

    def test_vocab_size(self):
        tok = CharTokenizer(["a", "b"])
        # 4 special tokens + 2 user tokens
        assert tok.vocab_size == 6

    def test_from_texts(self):
        tok = CharTokenizer.from_texts(["x^2", "\\frac"])
        assert tok.vocab_size > 4  # at least special tokens + chars

    def test_decode_skip_special(self):
        tok = CharTokenizer(["a", "b"])
        ids = tok.encode_with_special("ab")
        text = tok.decode(ids, skip_special=True)
        assert text == "ab"


# ---------------------------------------------------------------------------
# LatexDataset tests
# ---------------------------------------------------------------------------


class TestLatexDataset:
    def _make_png(self, path: str, width: int = 32, height: int = 16) -> None:
        img = Image.new("L", (width, height), color=200)
        img.save(path)

    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.png")
            self._make_png(img_path)
            manifest = os.path.join(tmpdir, "manifest.tsv")
            with open(manifest, "w") as f:
                f.write(f"{img_path}\t\\frac{{1}}{{2}}\n")

            tok = CharTokenizer.from_texts(["\\frac{12}"])
            ds = LatexDataset(manifest, tok, build_transforms(32, 64), max_seq_len=16)
            assert len(ds) == 1

    def test_sample_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.png")
            self._make_png(img_path, width=64, height=32)
            manifest = os.path.join(tmpdir, "manifest.tsv")
            with open(manifest, "w") as f:
                f.write(f"{img_path}\tx^2\n")

            tok = CharTokenizer.from_texts(["x^2"])
            ds = LatexDataset(
                manifest, tok, build_transforms(32, 64), max_seq_len=10
            )
            sample = ds[0]
            assert sample["image"].shape == (1, 32, 64)
            assert sample["tokens"].shape == (10,)
            assert sample["tokens"].dtype == torch.long

    def test_invalid_manifest_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "bad.tsv")
            with open(manifest, "w") as f:
                f.write("no_tab_here\n")
            tok = CharTokenizer()
            with pytest.raises(ValueError):
                LatexDataset(manifest, tok)

    def test_comment_lines_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "img.png")
            self._make_png(img_path)
            manifest = os.path.join(tmpdir, "manifest.tsv")
            with open(manifest, "w") as f:
                f.write("# this is a comment\n")
                f.write(f"{img_path}\tx\n")
            tok = CharTokenizer(["x"])
            ds = LatexDataset(manifest, tok, build_transforms(16, 32), max_seq_len=5)
            assert len(ds) == 1


# ---------------------------------------------------------------------------
# build_transforms tests
# ---------------------------------------------------------------------------


class TestBuildTransforms:
    def test_output_tensor_shape(self):
        transform = build_transforms(64, 128, augment=False)
        img = Image.new("RGB", (200, 100))
        tensor = transform(img)
        assert tensor.shape == (1, 64, 128)

    def test_values_in_unit_range(self):
        transform = build_transforms(32, 64)
        img = Image.new("L", (100, 50))
        tensor = transform(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
