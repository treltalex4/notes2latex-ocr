import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rope import RotaryEmbedding

class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        # параметры размерностей
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model должен делиться на nhead"

        # линейные слои Query, Key, Value
        # self.W_q == nn.Linear(d_model, d_model)
        # self.W_k == nn.Linear(d_model, d_model)
        # self.W_v == nn.Linear(d_model, d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # выходная проекция
        self.W_o = nn.Linear(d_model, d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        # проекция Query Key Value
        # [B, seq_len, 3, nhead, head_dim]
        B, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv =  qkv.view(B, seq_len, 3, self.nhead, self.head_dim)

        # разбиение и перестановка осей
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, nhead, seq_len, head_dim]

        q, k, v = qkv[0], qkv[1], qkv[2]

        #RoPE
        q, k = self.rope(q, k)

        # Attention

        # [B, nhead, seq_len, head_dim]
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True, 
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        out = out.transpose(1, 2)
        out = out.contiguous()
        out = out.view(B, seq_len, self.d_model)
        out = self.W_o(out)
        out = self.dropout(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # подслои selfattention, cross attention, feed forward
        self.self_attn = RoPESelfAttention(d_model, nhead, dropout)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory): 
        # x: [B, tgt_len, d_model]
        # memory: [b, src_len, d_model]

        # selfatt
        x = x + self.dropout(self.self_attn(self.norm1(x)))

        # cross
        normed = self.norm2(x)
        attn_out, _ = self.cross_attn(normed, memory, memory)
        x = x + self.dropout(attn_out)

        # ffn
        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x
    

class LaTeXDecoder(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            config.d_model,
            padding_idx=0,
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead, 
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
            for _ in range(config.num_decoder_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, vocab_size)

    def forward(self, tgt_ids, memory):
        # tgt_ids: [B, tgt_len]
        # memory:  [B, src_len, d_model]

        x = self.embedding(tgt_ids) # [B, tgt_len, d_model]

        for layer in self.layers:
            x = layer(x, memory) # [B, tgt_len, d_model]

        x = self.final_norm(x)
        logits = self.output_proj(x) # [B, tgt_len, vocab_size]
        return logits


if __name__ == "__main__":
    from config import load_config
    config = load_config()

    vocab_size = 1000
    decoder = LaTeXDecoder(config, vocab_size)

    tgt_ids = torch.randint(0, vocab_size, (2, 30))      # [B=2, tgt_len=30]
    memory = torch.randn(2, 100, config.d_model)          # [B=2, src_len=100, 256]

    logits = decoder(tgt_ids, memory)
    print("Shape:", logits.shape)            # torch.Size([2, 30, 1000])

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Параметров: {n_params:,}")       # ~4.7M
