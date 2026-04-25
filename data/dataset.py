import os
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from data.preprocess import preprocess_image
from data.tokenizer import LaTeXTokenizer, PAD_ID


class Im2LatexDataset(Dataset):
    """PyTorch Dataset для im2latex-100k."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        target_h: int = 128,
        target_w: int = 1024,
    ):
        self.data_dir = data_dir
        self.target_h = target_h
        self.target_w = target_w
        self.is_train = (split == "train")

        formulas_path = os.path.join(data_dir, "im2latex_formulas.lst")
        with open(formulas_path, encoding="latin-1", newline='\n') as f:
            # Убираем пробелы и случайные переносы каретки (\r) внутри строк
            self.formulas = [line.replace('\r', '').strip() for line in f]

        split_path = os.path.join(data_dir, f"im2latex_{split}.lst")
        self.samples: list[tuple[str, int]] = []

        with open(split_path, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                formula_idx = int(parts[0])
                image_name = parts[1]
                image_path = os.path.join(data_dir, "formula_images", image_name + ".png")

                if formula_idx < len(self.formulas) and self.formulas[formula_idx]:
                    self.samples.append((image_path, formula_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path, formula_idx = self.samples[idx]
        formula = self.formulas[formula_idx]

        # Изображение может быть разной ширины: [1, target_h, W]. Аугментируем только на трейне.
        image = preprocess_image(image_path, self.target_h, self.target_w, augment=self.is_train)

        if image is None:
            image = torch.zeros(1, self.target_h, 32) # Заглушка

        return image, formula


class BucketBatchSampler(Sampler):
    """Группирует формулы. Динамически уменьшает размер батча для гигантских строк (VRAM protection)."""
    
    def __init__(self, dataset: Im2LatexDataset, base_batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.shuffle = shuffle
        
        # Считаем длину формул
        self.lengths = [len(dataset.formulas[idx]) for _, idx in dataset.samples]

    def _get_dynamic_batch_size(self, max_len: int) -> int:
        """Если строчки огромные, берем меньше картинок в батч."""
        if max_len > 800:
            return max(1, self.base_batch_size // 8)
        elif max_len > 500:
            return max(1, self.base_batch_size // 4)
        elif max_len > 300:
            return max(1, self.base_batch_size // 2)
        else:
            return self.base_batch_size

    def _generate_batches(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            noisy_lengths = [l + random.uniform(-20, 20) for l in self.lengths]
            indices.sort(key=lambda i: noisy_lengths[i])
        else:
            indices.sort(key=lambda i: self.lengths[i])
            
        batches = []
        current_batch = []
        current_max_len = 0
        
        for idx in indices:
            current_max_len = max(current_max_len, self.lengths[idx])
            current_batch.append(idx)
            
            # Проверяем лимит
            dynamic_bs = self._get_dynamic_batch_size(current_max_len)
            if len(current_batch) >= dynamic_bs:
                batches.append(current_batch)
                current_batch = []
                current_max_len = 0
                
        if current_batch:
            batches.append(current_batch)
            
        return batches

    def __iter__(self):
        batches = self._generate_batches()
        
        if self.shuffle:
            random.shuffle(batches)
            
        for batch in batches:
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        # Отключаем шум, чтобы вернуть точное количество батчей
        old_shuffle = self.shuffle
        self.shuffle = False
        length = len(self._generate_batches())
        self.shuffle = old_shuffle
        return length


class CollateFunction:    
    def __init__(self, tokenizer: LaTeXTokenizer, max_len: int = 500):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(
        self, batch: list[tuple[torch.Tensor, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        encoded_formulas = []

        # Находим максимальные размеры картинок в этом конкретном батче
        max_h = max(img.shape[1] for img, _ in batch)
        max_w = max(img.shape[2] for img, _ in batch)

        #Находим макс длину формулы в этом батче
        batch_max_tok = 0
        for _, formula in batch:
            toks = self.tokenizer.tokenize(formula)
            batch_max_tok = max(batch_max_tok, len(toks))
            
        real_max_len = min(self.max_len, batch_max_tok + 2) # +2 для SOS и EOS

        for image, formula in batch:
            # Динамический паддинг картинок белым фоном (value=1.0)
            _, h, w = image.shape
            pad_w = max_w - w
            pad_h = max_h - h
            
            if pad_w > 0 or pad_h > 0:
                image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=1.0)
            images.append(image)

            # Динамическая токенезация
            encoded = self.tokenizer.encode(formula, max_len=real_max_len)
            encoded_formulas.append(encoded)

        images = torch.stack(images, dim=0)
        labels = torch.tensor(encoded_formulas, dtype=torch.long)

        return images, labels


def build_dataloaders(
    data_dir: str,
    tokenizer: LaTeXTokenizer,
    batch_size: int = 16, # Уменьшено с 32, так как картинки теперь 128px
    max_len: int = 500,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    collate_fn = CollateFunction(tokenizer, max_len=max_len)

    train_dataset = Im2LatexDataset(data_dir, split="train")
    val_dataset = Im2LatexDataset(data_dir, split="validate")
    test_dataset = Im2LatexDataset(data_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketBatchSampler(train_dataset, batch_size, shuffle=True),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=BucketBatchSampler(val_dataset, batch_size, shuffle=False),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_sampler=BucketBatchSampler(test_dataset, batch_size, shuffle=False),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
