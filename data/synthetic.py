import json
import os

import torch
from torch.utils.data import Dataset

from data.preprocess import preprocess_image


class SyntheticDataset(Dataset):
    def __init__(self, synthetic_dir: str, target_h: int = 128, max_w: int = 1024):
        self.synthetic_dir = synthetic_dir
        self.target_h = target_h
        self.max_w = max_w

        labels_path = os.path.join(synthetic_dir, "labels.json")
        with open(labels_path, encoding="utf-8") as f:
            labels: dict[str, str] = json.load(f)

        self.samples: list[tuple[str, str]] = [
            (os.path.join(synthetic_dir, "images", fname), latex)
            for fname, latex in labels.items()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path, formula = self.samples[idx]
        image = preprocess_image(image_path, self.target_h, self.max_w, augment=True)
        if image is None:
            image = torch.zeros(1, self.target_h, 32)
        return image, formula
