"""Tests for the ViT encoder."""

import torch
import pytest
from src.model.encoder import PatchEmbedding, ViTBlock, ViTEncoder


class TestPatchEmbedding:
    def test_output_shape(self):
        embed = PatchEmbedding(
            image_height=64, image_width=128, patch_size=16, in_channels=1, embed_dim=64
        )
        x = torch.randn(2, 1, 64, 128)
        out = embed(x)
        # num_patches = (64/16) * (128/16) = 4 * 8 = 32
        assert out.shape == (2, 32, 64)

    def test_num_patches(self):
        embed = PatchEmbedding(
            image_height=128, image_width=512, patch_size=16, in_channels=1, embed_dim=256
        )
        assert embed.num_patches == (128 // 16) * (512 // 16)

    def test_invalid_height_raises(self):
        with pytest.raises(AssertionError):
            PatchEmbedding(
                image_height=65, image_width=128, patch_size=16, in_channels=1, embed_dim=64
            )

    def test_invalid_width_raises(self):
        with pytest.raises(AssertionError):
            PatchEmbedding(
                image_height=64, image_width=130, patch_size=16, in_channels=1, embed_dim=64
            )

    def test_rgb_input(self):
        embed = PatchEmbedding(
            image_height=32, image_width=32, patch_size=16, in_channels=3, embed_dim=64
        )
        x = torch.randn(1, 3, 32, 32)
        out = embed(x)
        assert out.shape == (1, 4, 64)


class TestViTBlock:
    def test_output_shape(self):
        block = ViTBlock(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (weights are random, not identity)."""
        block = ViTBlock(embed_dim=64, num_heads=4)
        x = torch.randn(1, 5, 64)
        out = block(x)
        assert not torch.allclose(out, x)


class TestViTEncoder:
    def _make_encoder(self, **kwargs):
        defaults = dict(
            image_height=64,
            image_width=128,
            patch_size=16,
            in_channels=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return ViTEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_encoder()
        x = torch.randn(2, 1, 64, 128)
        out = enc(x)
        num_patches = (64 // 16) * (128 // 16)
        assert out.shape == (2, num_patches, 64)

    def test_batch_independence(self):
        """Two independent forward passes should give the same result for the same image."""
        enc = self._make_encoder()
        enc.eval()
        x = torch.randn(1, 1, 64, 128)
        with torch.no_grad():
            out1 = enc(x)
            out2 = enc(x)
        assert torch.allclose(out1, out2)

    def test_pos_embed_shape(self):
        enc = self._make_encoder()
        num_patches = (64 // 16) * (128 // 16)
        assert enc.pos_embed.shape == (1, num_patches, 64)

    def test_gradient_flow(self):
        enc = self._make_encoder()
        x = torch.randn(1, 1, 64, 128)
        out = enc(x).mean()
        out.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
