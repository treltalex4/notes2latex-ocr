"""Combined Pix2Tex model: ViT encoder + Transformer decoder.

Given an image of handwritten text the model produces a LaTeX token sequence
autoregressively, attending over the encoded image patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder import TransformerDecoder


class Pix2Tex(nn.Module):
    """End-to-end image-to-LaTeX model.

    The encoder splits the input image into patches and encodes them with a
    Vision Transformer.  The decoder attends over these patch representations
    and generates a LaTeX token sequence one token at a time.

    Parameters
    ----------
    vocab_size:
        Size of the LaTeX token vocabulary.
    image_height:
        Expected input image height (pixels).
    image_width:
        Expected input image width (pixels).
    patch_size:
        Square patch side length (pixels).
    in_channels:
        Number of image channels (1 = grayscale, 3 = RGB).
    embed_dim:
        Shared embedding dimension for encoder and decoder.
    encoder_depth:
        Number of ViT encoder blocks.
    decoder_depth:
        Number of Transformer decoder blocks.
    num_heads:
        Number of attention heads (used in both encoder and decoder).
    mlp_ratio:
        FFN hidden-dim expansion factor.
    dropout:
        Dropout probability.
    max_seq_len:
        Maximum generated sequence length in tokens.
    pad_token_id:
        Padding token ID (0 by default).
    bos_token_id:
        Beginning-of-sequence token ID (1 by default).
    eos_token_id:
        End-of-sequence token ID (2 by default).
    """

    def __init__(
        self,
        vocab_size: int,
        image_height: int = 128,
        image_width: int = 512,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> None:
        super().__init__()

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.encoder = ViTEncoder(
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

    # ------------------------------------------------------------------
    # Forward (teacher-forced, used during training)
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy logits for a batch (teacher forcing).

        Parameters
        ----------
        images:
            Image batch ``(B, C, H, W)``.
        tgt_tokens:
            Target token IDs ``(B, T)`` including the leading BOS token.

        Returns
        -------
        torch.Tensor
            Logits ``(B, T, vocab_size)``.
        """
        memory = self.encoder(images)                   # (B, N, D)
        logits = self.decoder(tgt_tokens, memory)       # (B, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the teacher-forced cross-entropy loss.

        The loss is computed by shifting *tgt_tokens* by one position:
        the decoder input is ``tgt_tokens[:, :-1]`` (BOS … last-1) and the
        target labels are ``tgt_tokens[:, 1:]`` (first+1 … EOS).  Padding
        positions are ignored.

        Parameters
        ----------
        images:
            Image batch ``(B, C, H, W)``.
        tgt_tokens:
            Full target sequences ``(B, T)`` with BOS prepended and EOS appended.

        Returns
        -------
        torch.Tensor
            Scalar cross-entropy loss.
        """
        decoder_input = tgt_tokens[:, :-1]   # (B, T-1)
        labels = tgt_tokens[:, 1:]           # (B, T-1)

        logits = self.forward(images, decoder_input)  # (B, T-1, vocab_size)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Generate LaTeX token sequences for a batch of images.

        Parameters
        ----------
        images:
            Image batch ``(B, C, H, W)``.
        max_new_tokens:
            Maximum number of tokens to generate per image.

        Returns
        -------
        torch.Tensor
            Token IDs ``(B, T_out)``.
        """
        memory = self.encoder(images)
        return self.decoder.generate(
            memory,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
