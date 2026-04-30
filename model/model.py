import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

from model.encoder import HybridEncoder
from model.decoder import LaTeXDecoder

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Notes2LaTeX(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.encoder = HybridEncoder(config)
        self.decoder = LaTeXDecoder(config, vocab_size)

    def forward(
        self,
        images: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, memory_kpm = self.encoder(images, src_key_padding_mask=src_key_padding_mask)
        logits = self.decoder(
            tgt_ids, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_kpm,
        )
        return logits

if __name__ == "__main__":
    from config import load_config
    config = load_config()

    vocab_size = 1000
    model = Notes2LaTeX(config, vocab_size)

    images = torch.randn(2, 1, 128, 400)
    tgt_ids = torch.randint(0, vocab_size, (2, 30))

    logits = model(images, tgt_ids)
    print("Shape:", logits.shape)              # torch.Size([2, 30, 1000])

    print(f"Параметров: {count_parameters(model):,}")
    # ~8.8M (4.1M encoder + 4.7M decoder)

    # Проверка что градиент течёт (forward + dummy backward)
    loss = logits.sum()
    loss.backward()
    print("Backward OK")
