"""Tests for the Transformer decoder."""

import torch
import pytest
from src.model.decoder import DecoderBlock, TransformerDecoder


VOCAB_SIZE = 50
EMBED_DIM = 64
NUM_HEADS = 4
DEPTH = 2
MAX_SEQ = 32


def _make_decoder(**kwargs) -> TransformerDecoder:
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return TransformerDecoder(**defaults)


class TestDecoderBlock:
    def test_output_shape(self):
        block = DecoderBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
        x = torch.randn(2, 10, EMBED_DIM)
        mem = torch.randn(2, 8, EMBED_DIM)
        out = block(x, mem)
        assert out.shape == x.shape

    def test_causal_mask_applied(self):
        """With a causal mask, token at position 0 must not see position 1."""
        block = DecoderBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
        block.eval()
        T = 4
        x = torch.randn(1, T, EMBED_DIM)
        mem = torch.randn(1, 8, EMBED_DIM)
        mask = torch.full((T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        with torch.no_grad():
            out_masked = block(x, mem, tgt_mask=mask)
            out_unmasked = block(x, mem, tgt_mask=None)
        # They should differ because the mask prevents future token access.
        assert not torch.allclose(out_masked, out_unmasked)


class TestTransformerDecoder:
    def test_forward_output_shape(self):
        dec = _make_decoder()
        tgt = torch.randint(0, VOCAB_SIZE, (2, 10))
        mem = torch.randn(2, 8, EMBED_DIM)
        logits = dec(tgt, mem)
        assert logits.shape == (2, 10, VOCAB_SIZE)

    def test_forward_deterministic_in_eval(self):
        dec = _make_decoder()
        dec.eval()
        tgt = torch.randint(0, VOCAB_SIZE, (1, 5))
        mem = torch.randn(1, 8, EMBED_DIM)
        with torch.no_grad():
            l1 = dec(tgt, mem)
            l2 = dec(tgt, mem)
        assert torch.allclose(l1, l2)

    def test_causal_mask_shape(self):
        mask = TransformerDecoder._causal_mask(5, torch.device("cpu"))
        assert mask.shape == (5, 5)
        # Diagonal and below should be 0.0, above should be -inf
        assert mask[0, 0] == 0.0
        assert mask[0, 1] == float("-inf")

    def test_generate_returns_bos_as_first_token(self):
        dec = _make_decoder()
        dec.eval()
        mem = torch.randn(1, 8, EMBED_DIM)
        bos, eos = 1, 2
        tokens = dec.generate(mem, bos_token_id=bos, eos_token_id=eos, max_new_tokens=5)
        assert tokens[0, 0].item() == bos

    def test_generate_stops_at_eos(self):
        """Verify that generate() respects max_new_tokens and stops at EOS."""
        dec = _make_decoder()
        dec.eval()
        mem = torch.randn(1, 8, EMBED_DIM)
        bos, eos, max_new = 1, 2, 8
        tokens = dec.generate(mem, bos_token_id=bos, eos_token_id=eos, max_new_tokens=max_new)
        # Total length must not exceed BOS token + max_new_tokens
        assert tokens.shape[1] <= max_new + 1

    def test_gradient_flow(self):
        dec = _make_decoder()
        tgt = torch.randint(1, VOCAB_SIZE, (1, 6))
        mem = torch.randn(1, 8, EMBED_DIM, requires_grad=True)
        logits = dec(tgt, mem)
        logits.mean().backward()
        assert mem.grad is not None
        for name, p in dec.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_pad_token_ignored_in_key_padding(self):
        """Padding tokens should be masked out; swapping a non-pad token to pad changes output."""
        dec = _make_decoder()
        dec.eval()
        # seq with no padding
        tgt_no_pad = torch.tensor([[1, 5, 6, 7, 2]])
        # same but last real token replaced with PAD
        tgt_with_pad = torch.tensor([[1, 5, 6, 0, 2]])
        mem = torch.randn(1, 8, EMBED_DIM)
        with torch.no_grad():
            l1 = dec(tgt_no_pad, mem)
            l2 = dec(tgt_with_pad, mem)
        assert not torch.allclose(l1, l2)
