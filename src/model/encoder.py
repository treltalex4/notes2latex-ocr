"""ViT Encoder: splits an image into fixed-size patches and encodes them with a
Transformer, following the Vision Transformer (ViT) architecture described in
"An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).

Pipeline
--------
image  →  patch embedding  →  + positional embedding  →  Transformer blocks  →  encoded sequence
"""

import math

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Project image patches to the model dimension.

    Parameters
    ----------
    image_height:
        Expected height of the input image (pixels).
    image_width:
        Expected width of the input image (pixels).
    patch_size:
        Side length of each square patch in pixels.
    in_channels:
        Number of input image channels (1 for grayscale, 3 for RGB).
    embed_dim:
        Dimension of the patch embedding vector.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        assert image_height % patch_size == 0, (
            f"Image height {image_height} must be divisible by patch_size {patch_size}"
        )
        assert image_width % patch_size == 0, (
            f"Image width {image_width} must be divisible by patch_size {patch_size}"
        )

        self.patch_size = patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # A single convolution replaces the explicit reshape + linear projection.
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape ``(B, num_patches, embed_dim)``.
        """
        # (B, embed_dim, H/P, W/P) → (B, num_patches, embed_dim)
        x = self.projection(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class ViTBlock(nn.Module):
    """Single Transformer encoder block (multi-head self-attention + FFN).

    Parameters
    ----------
    embed_dim:
        Dimensionality of the token embeddings.
    num_heads:
        Number of attention heads.
    mlp_ratio:
        Hidden dimension of the FFN relative to *embed_dim*.
    dropout:
        Dropout probability applied inside attention and the FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Token sequence of shape ``(B, N, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Updated token sequence, same shape as input.
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer (ViT) encoder for handwritten-text images.

    The image is divided into non-overlapping square patches.  Each patch is
    linearly projected into an embedding vector.  A learnable 1-D sinusoidal
    positional embedding is added before feeding the sequence into a stack of
    standard Transformer encoder blocks.

    Parameters
    ----------
    image_height:
        Input image height in pixels.
    image_width:
        Input image width in pixels.
    patch_size:
        Side length of each square patch (default: 16).
    in_channels:
        Number of image channels (default: 1 for grayscale).
    embed_dim:
        Dimensionality of patch embeddings (default: 256).
    depth:
        Number of Transformer encoder blocks (default: 6).
    num_heads:
        Number of attention heads per block (default: 8).
    mlp_ratio:
        FFN hidden-dim expansion factor (default: 4.0).
    dropout:
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 512,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_height, image_width, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Learnable positional embedding — one vector per patch position.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self._init_pos_embed(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_pos_embed(self, num_patches: int, embed_dim: int) -> None:
        """Fill ``pos_embed`` with 1-D sinusoidal values as a warm start."""
        pe = torch.zeros(num_patches, embed_dim)
        position = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embed_dim // 2])
        with torch.no_grad():
            self.pos_embed.copy_(pe.unsqueeze(0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into a sequence of patch embeddings.

        Parameters
        ----------
        x:
            Image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Encoded sequence of shape ``(B, num_patches, embed_dim)``.
        """
        x = self.patch_embed(x)            # (B, N, D)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
