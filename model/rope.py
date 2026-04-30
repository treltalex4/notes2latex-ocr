import torch
import torch.nn as nn


# меняем половинки местами 
def rotate_half(x):
    half = x.shape[-1] // 2 
    x1 = x[..., :half] 
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__() # инициализируем nn.module

        # Q_i = 1 / (10000 ^ (2i / dim))
        i = torch.arange(0, dim, 2).float()
        exponents = i / dim
        freqs = base ** exponents
        inv_freq = 1.0 / freqs
        self.register_buffer("inv_freq", inv_freq)

    # поворачиваем вектора
    def forward(self, q, k, offset: int = 0):
        # матрица углов. offset позволяет сдвинуть стартовую позицию —
        # нужно для авторегрессивного декодинга с KV-кэшем, когда в forward
        # подаётся один новый токен, но его реальная позиция не нулевая.
        seq_len = q.shape[2]
        t = torch.arange(seq_len, device=self.inv_freq.device) + offset
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, None, :, :] # первые два места для осей batch и head
        sin = emb.sin()[None, None, :, :]

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot


# проверка
if __name__ == "__main__":
    # форма inv_freq
    rope = RotaryEmbedding(dim=32)
    print(rope.inv_freq.shape)   # torch.Size([16])

    # форма выходных тензоров не изменилась
    B, nhead, seq_len, head_dim = 2, 4, 10, 32
    q = torch.randn(B, nhead, seq_len, head_dim)
    k = torch.randn(B, nhead, seq_len, head_dim)
    q_rot, k_rot = rope(q, k)
    print(q_rot.shape)   # torch.Size([2, 4, 10, 32])
    print(k_rot.shape)   # torch.Size([2, 4, 10, 32])

    # вращение сохраняет длину вектора (норма не меняется)
    print(torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5))  # True

    # offset: один токен в позиции 5 должен совпасть с пятым токеном full-prefix
    full_q, _ = rope(q, k)
    one_q, _ = rope(q[:, :, 5:6, :], k[:, :, 5:6, :], offset=5)
    print(torch.allclose(full_q[:, :, 5:6, :], one_q, atol=1e-5))  # True

