import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

# pos encoding
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        #PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        #PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

        i = torch.arange(0, d_model, 2).float()
        exponents = i / d_model
        freqs = 10000 ** exponents
        inv_freq = 1.0 / freqs
        
        angles = position * inv_freq
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe)
    def forward(self, x):
        # x : [B, seq_len, d_model]
        seq_len = x.shape[1]

        return x + self.pe[:seq_len].unsqueeze(0)



class _HeightMean(nn.Module):
    """Эквивалент nn.AdaptiveAvgPool2d((1, None)) — среднее по H, ширина
    сохраняется. Используем mean вместо AdaptiveAvgPool2d, потому что
    последний с символической шириной ломает torch.compile (Inductor
    lowering пытается решить window_size > 25 на symbolic expressions
    и падает с TypeError на Relational)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2, keepdim=True)


class HybridEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        

        # CNN
        def _make_block(in_ch, out_ch, pool_kernel):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel),
            )
        

        channels = (1,) + tuple(config.cnn_channels) + (config.d_model,)

        pool_kernels = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]

        blocks = []
        for i in range(5):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            pool = pool_kernels[i]
            blocks.append(_make_block(in_ch, out_ch, pool))
        blocks.append(_HeightMean())

        self.cnn = nn.Sequential(*blocks)
        self.pe = SinusoidalPE(config.d_model, max_len=config.max_seq_len)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder (
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: [B, 1, H, W]
        # src_key_padding_mask: [B, W] на пиксельной ширине, True = паддинг.
        # Возвращает (memory, memory_key_padding_mask) — маска уже понижена под
        # ширину memory (CNN суммарно уменьшает W в WIDTH_STRIDE раз).
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # [B, mem_len, d_model]
        x = self.pe(x)

        memory_kpm: torch.Tensor | None = None
        if src_key_padding_mask is not None:
            mem_len = x.shape[1]
            # Понижаем пиксельную маску до memory-разрешения. Используем
            # max_pool1d (=any): окно считается паддингом, если хоть один
            # пиксель в нём — паддинг. Адаптивная стратегия: вычисляем
            # фактический stride из соотношения размеров, чтобы не зависеть
            # от хардкода даже при изменении pool_kernels.
            pad_f = src_key_padding_mask.float().unsqueeze(1)  # [B, 1, W]
            stride = pad_f.shape[-1] // mem_len
            pooled = F.max_pool1d(pad_f, kernel_size=stride, stride=stride)
            memory_kpm = pooled.squeeze(1)[:, :mem_len].bool()

        x = self.transformer(x, src_key_padding_mask=memory_kpm)
        return x, memory_kpm

        

        
if __name__ == "__main__":
    from config import load_config
    config = load_config()
    encoder = HybridEncoder(config)

    x = torch.randn(2, 1, 128, 400)
    out, kpm = encoder(x)
    print("Output shape:", out.shape, "  mask:", kpm)   # torch.Size([2, 100, 256])

    # С маской: последние 80 пикселей (=20 memory-токенов) паддинг
    src_kpm = torch.zeros(2, 400, dtype=torch.bool)
    src_kpm[:, 320:] = True
    out2, kpm2 = encoder(x, src_key_padding_mask=src_kpm)
    print("With mask:", out2.shape, kpm2.shape, "  pad-tokens:", kpm2.sum(dim=1).tolist())

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Параметров: {n_params:,}")        # ~4.1M
