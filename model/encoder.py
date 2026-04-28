import os
import sys
import torch
import torch.nn as nn

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
        blocks.append(nn.AdaptiveAvgPool2d((1, None)))

        self.cnn = nn.Sequential(*blocks)

        self.pe = SinusoidalPE(config.d_model, max_len=config.max_seq_len)

        

    # transformer encoder
        
if __name__ == "__main__":
    from config import load_config
    config = load_config()

    # SinusoidalPE 
    pe = SinusoidalPE(d_model=256, max_len=512)
    print("PE shape:", pe.pe.shape)              # torch.Size([512, 256])
    print("PE[0]:", pe.pe[0, :4])                 # [0, 1, 0, 1] 

    # Forward 
    x = torch.randn(2, 50, 256)
    out = pe(x)
    print("Forward shape:", out.shape)            # torch.Size([2, 50, 256])

    # Encoder + CNN + PE 
    encoder = HybridEncoder(config)
    x = torch.randn(2, 1, 128, 400)
    out = encoder.cnn(x)
    print("CNN shape:", out.shape)                # torch.Size([2, 256, 1, 100])

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Параметров: {n_params:,}")           

