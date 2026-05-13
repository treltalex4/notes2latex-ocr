import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from data.tokenizer import PAD_ID
from model.rope import RotaryEmbedding


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model должен делиться на nhead"

        # qkv = concat(W_q, W_k, W_v) — одна матмул-операция вместо трёх
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        position_offset: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # x: [B, seq_len, d_model]
        # key_padding_mask: [B, seq_len], True = маскировать
        # kv_cache: (K_past, V_past) с [B, nhead, T_past, head_dim] для авторегрессивного декода.
        # use_cache: вернуть обновлённый (K_full, V_full) для следующего шага.
        B, seq_len, _ = x.shape
        qkv = self.qkv(x).view(B, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, nhead, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k, offset=position_offset)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # SDPA не позволяет одновременно is_causal=True и attn_mask, поэтому при
        # наличии padding-маски собираем единую булеву маску [B, 1, S, S].
        if kv_cache is not None:
            # Декод с кэшем: Q — только новые токены, K/V содержат всю историю.
            # Causal-маска не нужна — будущих токенов в K/V просто нет.
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        elif key_padding_mask is not None:
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
            # True = заблокировать. Паддинг блокируем по столбцам (keys).
            pad = key_padding_mask[:, None, None, :]  # [B, 1, 1, seq_len]
            attn_mask = causal[None, None, :, :] | pad  # broadcast → [B, 1, S, S]
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=~attn_mask,  # SDPA: True = разрешить attend (с bool-маской)
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0,
            )

        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        out = self.W_o(out)
        # Dropout не применяется здесь — внешний residual dropout в DecoderLayer
        # покрывает этот путь (иначе получается двойной dropout, p_eff = 1-(1-p)²).
        return out, new_cache


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
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

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        position_offset: int = 0,
        self_attn_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # selfatt
        sa_out, new_cache = self.self_attn(
            self.norm1(x),
            key_padding_mask=tgt_key_padding_mask,
            position_offset=position_offset,
            kv_cache=self_attn_cache,
            use_cache=use_cache,
        )
        x = x + self.dropout(sa_out)

        # cross
        normed = self.norm2(x)
        attn_out, _ = self.cross_attn(
            normed, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # ffn
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x, new_cache


class LaTeXDecoder(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            config.d_model,
            padding_idx=PAD_ID,
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

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        # tgt_ids: [B, tgt_len]
        # memory:  [B, src_len, d_model]
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt_ids == PAD_ID)

        x = self.embedding(tgt_ids)

        for layer in self.layers:
            x, _ = layer(
                x, memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                position_offset=position_offset,
            )

        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits

    def forward_step(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Один шаг авторегрессивного декода с KV-кэшем.

        tgt_ids: [B, 1] — новый токен (или [B, T] при первом вызове).
        kv_caches: list длины num_layers, каждый элемент — (K, V) с
          [B, nhead, T_past, head_dim]. None при первом шаге.
        position_offset: позиция первого токена tgt_ids в полной
          последовательности (= размер кэша к моменту вызова).

        Возвращает (logits, new_caches):
          logits — [B, T_new, vocab], где T_new = tgt_ids.shape[1].
          new_caches — обновлённые кэши для следующего шага.
        """
        x = self.embedding(tgt_ids)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, memory,
                tgt_key_padding_mask=None,   # авторегрессивно — паддинга в кэше нет
                memory_key_padding_mask=memory_key_padding_mask,
                position_offset=position_offset,
                self_attn_cache=layer_cache,
                use_cache=True,
            )
            new_caches.append(new_cache)

        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits, new_caches


if __name__ == "__main__":
    from config import load_config
    config = load_config()

    vocab_size = 1000
    decoder = LaTeXDecoder(config, vocab_size)
    decoder.eval()

    # Базовый тест
    tgt_ids = torch.randint(1, vocab_size, (2, 30))   # без PAD'ов
    memory = torch.randn(2, 100, config.d_model)
    logits = decoder(tgt_ids, memory)
    print("Shape:", logits.shape)            # torch.Size([2, 30, 1000])

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Параметров: {n_params:,}")

    # Проверка маски: маскированные позиции PAD не должны менять логиты остальных
    tgt_pad = tgt_ids.clone()
    tgt_pad[:, 25:] = PAD_ID
    logits_pad = decoder(tgt_pad, memory)
    same = torch.allclose(logits[:, :25, :], logits_pad[:, :25, :], atol=1e-5)
    print(f"PAD-маска не влияет на реальные позиции: {same}")

    # Проверка memory-маски: shape сохраняется
    mem_pad = torch.zeros(2, 100, dtype=torch.bool)
    mem_pad[:, 80:] = True
    logits_mem = decoder(tgt_ids, memory, memory_key_padding_mask=mem_pad)
    print(f"С memory-маской shape: {logits_mem.shape}")

    # Проверка position_offset: один токен в позиции 5 ≈ logits[5] full-prefix
    # (точного равенства не будет — cross-attn агрегирует всю memory одинаково,
    # но self-attn не вырождается).
    logits_one = decoder(tgt_ids[:, 5:6], memory, position_offset=5)
    print(f"position_offset works, shape: {logits_one.shape}")
