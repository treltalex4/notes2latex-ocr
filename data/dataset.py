import random

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler,
)

from config import Config
from data.cache import load_manifest as _load_manifest
from data.preprocess import apply_augmentations, to_tensor
from data.tokenizer import LaTeXTokenizer, PAD_ID


class _CachedDataset(Dataset):
    """Base class: reads preprocessed uint8 .npy arrays from data_cache/."""
    dataset_type: str = ""

    def __init__(self, entries: list[dict]) -> None:
        self.samples: list[tuple[str, str]] = [(e["npy_path"], e["formula"]) for e in entries]
        self.lengths: list[int] = [e["length"] for e in entries]
        # updated by train.py before each epoch
        self.elastic_p: float = 0.0
        self.elastic_alpha: int = 0
        self.elastic_sigma: int = 0
        self.strength: float = 0.0
        self.max_length: int = 512

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        npy_path, formula = self.samples[idx]
        img = np.load(npy_path)
        img = apply_augmentations(
            img, self.dataset_type,
            self.elastic_p, self.elastic_alpha, self.elastic_sigma, self.strength,
        )
        return to_tensor(img), formula


class Im2LatexDataset(_CachedDataset):
    dataset_type = "im2latex"

    def __init__(self, cache_dir: str, split: str = "train", max_length: int = 512) -> None:
        entries = _load_manifest(cache_dir, "im2latex")
        entries = [e for e in entries if e.get("split", "train") == split]
        super().__init__(entries)
        self.max_length = max_length


class SyntheticDataset(_CachedDataset):
    dataset_type = "synthetic"

    def __init__(self, cache_dir: str) -> None:
        entries = _load_manifest(cache_dir, "synthetic")
        super().__init__(entries)


class HandwrittenDataset(_CachedDataset):
    dataset_type = "handwritten"

    def __init__(self, cache_dir: str, split: str = "train") -> None:
        entries = _load_manifest(cache_dir, "handwritten")
        entries = sorted(entries, key=lambda e: e["npy_path"])  # deterministic order
        n_train = int(len(entries) * 0.8)
        entries = entries[:n_train] if split == "train" else entries[n_train:]
        super().__init__(entries)


class BucketBatchSampler(Sampler):
    """Groups samples by formula length. Shrinks batch size for very long sequences."""

    def __init__(self, dataset: _CachedDataset, base_batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.shuffle = shuffle
        self.lengths = dataset.lengths
        self.current_max_length: int = dataset.max_length  # updated by train.py each epoch

    def _get_dynamic_batch_size(self, max_len: int) -> int:
        if max_len > 800:
            return max(1, self.base_batch_size // 8)
        elif max_len > 500:
            return max(1, self.base_batch_size // 4)
        elif max_len > 300:
            return max(1, self.base_batch_size // 2)
        return self.base_batch_size

    def _generate_batches(self) -> list[list[int]]:
        indices = [i for i in range(len(self.dataset)) if self.lengths[i] <= self.current_max_length]

        if self.shuffle:
            noisy = {i: self.lengths[i] + random.uniform(-20, 20) for i in indices}
            indices.sort(key=lambda i: noisy[i])
        else:
            indices.sort(key=lambda i: self.lengths[i])

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_len = 0

        for idx in indices:
            current_max_len = max(current_max_len, self.lengths[idx])
            current_batch.append(idx)
            if len(current_batch) >= self._get_dynamic_batch_size(current_max_len):
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
                random.shuffle(batch)
                yield batch
        else:
            yield from batches

    def __len__(self) -> int:
        # batch count is independent of shuffle order, so just generate without side-effects
        return len(self._generate_batches())


class CollateFunction:
    def __init__(self, tokenizer: LaTeXTokenizer, max_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_h = max(img.shape[1] for img, _ in batch)
        max_w = max(img.shape[2] for img, _ in batch)

        batch_max_tok = max(len(self.tokenizer.tokenize(f)) for _, f in batch)
        real_max_len = min(self.max_len, batch_max_tok + 2)

        images, encoded_formulas = [], []
        for image, formula in batch:
            _, h, w = image.shape
            if max_w - w > 0 or max_h - h > 0:
                image = torch.nn.functional.pad(image, (0, max_w - w, 0, max_h - h), value=1.0)
            images.append(image)
            encoded_formulas.append(self.tokenizer.encode(formula, max_len=real_max_len))

        return torch.stack(images), torch.tensor(encoded_formulas, dtype=torch.long)


def _compute_sample_weights(
    named_datasets: list[tuple[str, Dataset]],
    target_ratios: dict[str, float],
) -> list[float]:
    weights: list[float] = []
    for name, ds in named_datasets:
        ratio = target_ratios.get(name, 1.0)
        n = len(ds)
        per_sample = ratio / n if n > 0 else 0.0
        weights.extend([per_sample] * n)
    return weights


def _make_loader_kwargs(config: Config, collate_fn: CollateFunction) -> dict:
    kwargs: dict = {"collate_fn": collate_fn, "num_workers": config.num_workers, "pin_memory": True}
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["persistent_workers"] = True  # avoid worker respawn overhead between epochs
    return kwargs


def build_multi_dataloaders(
    config: Config,
    tokenizer: LaTeXTokenizer,
    stage: int,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    stage=1: im2latex only (pretrain). BucketBatchSampler + length curriculum.
    stage=2: im2latex + synthetic. WeightedRandomSampler, val on im2latex validate split.
    stage=3: handwritten + synthetic replay. WeightedRandomSampler, val on handwritten val split.

    Returns (train_loader, val_loader, test_loader). test_loader is None for stages 2 and 3.
    """
    collate_fn = CollateFunction(tokenizer, max_len=config.tokenizer_max_len)
    kw = _make_loader_kwargs(config, collate_fn)

    if stage == 1:
        train_ds = Im2LatexDataset(config.cache_dir, split="train", max_length=config.tokenizer_max_len)
        val_ds   = Im2LatexDataset(config.cache_dir, split="validate")
        test_ds  = Im2LatexDataset(config.cache_dir, split="test")

        train_loader = DataLoader(train_ds, batch_sampler=BucketBatchSampler(train_ds, config.batch_size, shuffle=True),  **kw)
        val_loader   = DataLoader(val_ds,   batch_sampler=BucketBatchSampler(val_ds,   config.batch_size, shuffle=False), **kw)
        test_loader  = DataLoader(test_ds,  batch_sampler=BucketBatchSampler(test_ds,  config.batch_size, shuffle=False), **kw)
        return train_loader, val_loader, test_loader

    if stage == 2:
        im2latex_ds  = Im2LatexDataset(config.cache_dir, split="train")
        synthetic_ds = SyntheticDataset(config.cache_dir)
        val_ds       = Im2LatexDataset(config.cache_dir, split="validate")

        named = [("im2latex", im2latex_ds), ("synthetic", synthetic_ds)]
        combined = ConcatDataset([ds for _, ds in named])
        weights = _compute_sample_weights(named, config.dataset_weights_stage2)
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)

        train_loader = DataLoader(combined, sampler=sampler, batch_size=config.batch_size,
                                  drop_last=True, **kw)
        val_loader   = DataLoader(val_ds, batch_sampler=BucketBatchSampler(val_ds, config.batch_size, shuffle=False), **kw)
        return train_loader, val_loader, None

    if stage == 3:
        hw_train_ds  = HandwrittenDataset(config.cache_dir, split="train")
        hw_val_ds    = HandwrittenDataset(config.cache_dir, split="val")
        synthetic_ds = SyntheticDataset(config.cache_dir)

        named = [("handwritten", hw_train_ds), ("synthetic", synthetic_ds)]
        combined = ConcatDataset([ds for _, ds in named])
        weights = _compute_sample_weights(named, config.dataset_weights_stage3)
        # epoch size: enough to see each handwritten sample ~once
        hw_ratio = config.dataset_weights_stage3.get("handwritten", 0.82)
        num_samples = max(len(combined), int(len(hw_train_ds) / hw_ratio))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

        train_loader = DataLoader(combined, sampler=sampler, batch_size=config.batch_size,
                                  drop_last=True, **kw)
        val_loader   = DataLoader(hw_val_ds, batch_sampler=BucketBatchSampler(hw_val_ds, config.batch_size, shuffle=False), **kw)
        return train_loader, val_loader, None

    raise ValueError(f"stage должен быть 1, 2 или 3, получено: {stage}")
