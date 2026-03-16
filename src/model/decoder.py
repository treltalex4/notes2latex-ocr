"""Transformer Decoder: autoregressively generates a LaTeX token sequence by
attending over the encoded patch representations produced by the ViT encoder.

Architecture follows the standard Transformer decoder:
    - Token embedding + positional embedding
    - N × (masked self-attention → cross-attention over encoder output → FFN)
    - Linear projection to vocabulary logits
"""

import math

import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """Single Transformer decoder block.

    Parameters
    ----------
    embed_dim:
        Dimensionality of the token embeddings.
    num_heads:
        Number of attention heads.
    mlp_ratio:
        FFN hidden-dim expansion factor.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # 1. Masked causal self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 2. Cross-attention over encoder memory
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 3. Position-wise FFN
        self.norm3 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Decoder token sequence ``(B, T, embed_dim)``.
        memory:
            Encoder output ``(B, N, embed_dim)``.
        tgt_mask:
            Causal additive mask ``(T, T)`` — typically upper-triangular ``-inf``.
        tgt_key_padding_mask:
            Boolean padding mask for decoder tokens ``(B, T)``.
        memory_key_padding_mask:
            Boolean padding mask for encoder tokens ``(B, N)``.

        Returns
        -------
        torch.Tensor
            Updated decoder sequence ``(B, T, embed_dim)``.
        """
        # Masked self-attention
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(
            normed,
            normed,
            normed,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + sa_out

        # Cross-attention
        normed = self.norm2(x)
        ca_out, _ = self.cross_attn(
            normed,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + ca_out

        # FFN
        x = x + self.mlp(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    """Autoregressive Transformer decoder that generates LaTeX token sequences.

    Parameters
    ----------
    vocab_size:
        Total number of tokens in the LaTeX vocabulary (including special tokens).
    embed_dim:
        Dimensionality of the token embeddings (must match the encoder output dim).
    depth:
        Number of decoder blocks.
    num_heads:
        Number of attention heads per block.
    max_seq_len:
        Maximum output sequence length (in tokens).
    mlp_ratio:
        FFN hidden-dim expansion factor.
    dropout:
        Dropout probability.
    pad_token_id:
        Token ID used for padding (masked out in attention).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Return an additive upper-triangular causal mask of shape ``(T, T)``."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    # ------------------------------------------------------------------
    # Forward (teacher-forced, used during training)
    # ------------------------------------------------------------------

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits for the full target sequence (teacher forcing).

        Parameters
        ----------
        tgt_tokens:
            Target token IDs ``(B, T)``.
        memory:
            Encoder output ``(B, N, embed_dim)``.
        memory_key_padding_mask:
            Boolean mask ``(B, N)``; ``True`` where encoder positions are padding.

        Returns
        -------
        torch.Tensor
            Logits ``(B, T, vocab_size)``.
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.embed_drop(self.token_embed(tgt_tokens) + self.pos_embed(positions))

        tgt_mask = self._causal_mask(T, device)

        # Use a float additive key-padding mask to match the type of tgt_mask.
        bool_pad = tgt_tokens == self.pad_token_id  # (B, T)
        tgt_pad_mask_additive = torch.zeros_like(bool_pad, dtype=torch.float)
        tgt_pad_mask_additive.masked_fill_(bool_pad, float("-inf"))

        for block in self.blocks:
            x = block(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask_additive,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        x = self.norm(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Greedy decode (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding.

        Parameters
        ----------
        memory:
            Encoder output ``(B, N, embed_dim)``.
        bos_token_id:
            Token ID for the beginning-of-sequence token.
        eos_token_id:
            Token ID for the end-of-sequence token.
        max_new_tokens:
            Maximum number of tokens to generate (defaults to ``max_seq_len``).
        memory_key_padding_mask:
            Boolean mask ``(B, N)`` for encoder padding.

        Returns
        -------
        torch.Tensor
            Generated token IDs ``(B, T_out)`` (including the BOS token).
        """
        B = memory.size(0)
        device = memory.device
        max_new = max_new_tokens or self.max_seq_len

        tokens = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new):
            logits = self.forward(tokens, memory, memory_key_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished |= next_token == eos_token_id
            if finished.all():
                break

        return tokens
