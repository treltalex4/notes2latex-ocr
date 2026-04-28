import torch 
import torch.nn as nn
import torch.nn.functional as F

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
    

if __name__ == "__main__":
    attn = RoPESelfAttention(d_model=256, nhead=8, dropout=0.1)
    x = torch.randn(2, 50, 256)        # batch=2, seq=50
    out = attn(x)
    print("Shape:", out.shape)          # torch.Size([2, 50, 256])

    n_params = sum(p.numel() for p in attn.parameters())
    print(f"Параметров: {n_params:,}")  # ~263k (4 Linear по 256×256 + bias)
