import json
import os
import random

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler, random_split,
)

from data.preprocess import apply_augmentations, preprocess_image, to_tensor
from data.tokenizer import LaTeXTokenizer, PAD_ID


# ─── Raw dataset (for test_pipeline.py before cache is built) ────────────────

class RawIm2LatexDataset(Dataset):
    """Reads raw PNGs from data_raw/. For pipeline testing only."""

    def __init__(self, data_dir: str, split: str = "train", target_h: int = 128, target_w: int = 1024):
        self.data_dir = data_dir
        self.target_h = target_h
        self.target_w = target_w
        self.is_train = (split == "train")

        formulas_path = os.path.join(data_dir, "im2latex_formulas.lst")
        with open(formulas_path, encoding="latin-1", newline="\n") as f:
            self.formulas = [line.replace("\r", "").strip() for line in f]

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
        image = preprocess_image(image_path, self.target_h, self.target_w, augment=self.is_train)
        if image is None:
            image = torch.zeros(1, self.target_h, 32)
        return image, formula


# ─── Cached datasets (production) ────────────────────────────────────────────

def _load_manifest(cache_dir: str, dataset_name: str, split: str | None = None) -> list[dict]:
    fname = f"manifest_{split}.json" if split else "manifest.json"
    path = os.path.join(cache_dir, dataset_name, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run: python prepare_data.py --datasets {dataset_name}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _CachedDataset(Dataset):
    """Base class for all cache-backed datasets."""
    dataset_type: str = ""

    def __init__(self, entries: list[dict]):
        self.samples = [(e["path"], e["formula"]) for e in entries]
        self.lengths = [e["length"] for e in entries]
        # Updated by train.py before each epoch via curriculum schedule
        self.elastic_p: float = 0.0
        self.elastic_alpha: int = 0
        self.elastic_sigma: int = 0
        self.strength: float = 0.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path, formula = self.samples[idx]
        img = np.load(path)
        if self.elastic_p > 0 or self.strength > 0:
            img = apply_augmentations(
                img, self.dataset_type,
                self.elastic_p, self.elastic_alpha, self.elastic_sigma, self.strength,
            )
        return to_tensor(img), formula


class Im2LatexDataset(_CachedDataset):
    dataset_type = "im2latex"

    def __init__(self, config, split: str = "train"):
        super().__init__(_load_manifest(config.cache_dir, "im2latex", split))


class SyntheticDataset(_CachedDataset):
    dataset_type = "synthetic"

    def __init__(self, config):
        super().__init__(_load_manifest(config.cache_dir, "synthetic"))


class HandwrittenDataset(_CachedDataset):
    dataset_type = "handwritten"

    def __init__(self, config):
        super().__init__(_load_manifest(config.cache_dir, "handwritten"))


# ─── Samplers ─────────────────────────────────────────────────────────────────

class BucketBatchSampler(Sampler):
    """Groups samples by length for efficient batching.
    current_max_length: set by train.py for length curriculum filtering.
    """

    def __init__(self, lengths: list[int], base_batch_size: int, shuffle: bool = True):
        self.lengths = lengths
        self.base_batch_size = base_batch_size
        self.shuffle = shuffle
        self.current_max_length: int | None = None

    def _get_dynamic_batch_size(self, max_len: int) -> int:
        if max_len > 800:
            return max(1, self.base_batch_size // 8)
        elif max_len > 500:
            return max(1, self.base_batch_size // 4)
        elif max_len > 300:
            return max(1, self.base_batch_size // 2)
        return self.base_batch_size

    def _generate_batches(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))

        if self.current_max_length is not None:
            indices = [i for i in indices if self.lengths[i] <= self.current_max_length]

        if self.shuffle:
            noisy = {i: self.lengths[i] + random.uniform(-20, 20) for i in indices}
            indices.sort(key=lambda i: noisy[i])
        else:
            indices.sort(key=lambda i: self.lengths[i])

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max = 0

        for idx in indices:
            current_max = max(current_max, self.lengths[idx])
            current_batch.append(idx)
            if len(current_batch) >= self._get_dynamic_batch_size(current_max):
                batches.append(current_batch)
                current_batch = []
                current_max = 0

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
        old = self.shuffle
        self.shuffle = False
        n = len(self._generate_batches())
        self.shuffle = old
        return n


# ─── Collate ──────────────────────────────────────────────────────────────────

class CollateFunction:
    def __init__(self, tokenizer: LaTeXTokenizer, max_len: int = 500):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_h = max(img.shape[1] for img, _ in batch)
        max_w = max(img.shape[2] for img, _ in batch)

        batch_max_tok = max(len(self.tokenizer.tokenize(f)) for _, f in batch)
        real_max_len = min(self.max_len, batch_max_tok + 2)  # +2 for SOS and EOS

        images = []
        encoded_formulas = []
        for image, formula in batch:
            _, h, w = image.shape
            pad_w = max_w - w
            pad_h = max_h - h
            if pad_w > 0 or pad_h > 0:
                image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=1.0)
            images.append(image)
            encoded_formulas.append(self.tokenizer.encode(formula, max_len=real_max_len))

        return torch.stack(images), torch.tensor(encoded_formulas, dtype=torch.long)


# ─── Dataset builders ─────────────────────────────────────────────────────────

def _make_weighted_loader(
    datasets: list[Dataset],
    ratios: list[float],
    collate_fn: CollateFunction,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    combined = ConcatDataset(datasets)
    weights: list[float] = []
    for ds, ratio in zip(datasets, ratios):
        n = len(ds)
        w = ratio / n if n > 0 else 0.0
        weights.extend([w] * n)
    sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
    return DataLoader(
        combined,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_multi_dataloaders(
    config,
    tokenizer: LaTeXTokenizer,
    stage: int,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Returns (train_loader, val_loader, test_loader).
    stage=1: im2latex only, BucketBatchSampler + length curriculum
    stage=2: im2latex + synthetic, WeightedRandomSampler
    stage=3: handwritten (80%) + synthetic replay (20%), WeightedRandomSampler
             test_loader is None (no separate test split for handwritten)
    """
    collate_fn = CollateFunction(tokenizer, max_len=config.tokenizer_max_len)
    n_workers = config.num_workers
    bs = config.batch_size

    if stage == 1:
        train_ds = Im2LatexDataset(config, split="train")
        val_ds = Im2LatexDataset(config, split="validate")
        test_ds = Im2LatexDataset(config, split="test")

        train_sampler = BucketBatchSampler(train_ds.lengths, bs, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=BucketBatchSampler(val_ds.lengths, bs, shuffle=False),
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_sampler=BucketBatchSampler(test_ds.lengths, bs, shuffle=False),
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    if stage == 2:
        im2latex_ds = Im2LatexDataset(config, split="train")
        synthetic_ds = SyntheticDataset(config)
        val_ds = Im2LatexDataset(config, split="validate")
        test_ds = Im2LatexDataset(config, split="test")

        w = config.dataset_weights_stage2
        train_loader = _make_weighted_loader(
            [im2latex_ds, synthetic_ds],
            [w["im2latex"], w["synthetic"]],
            collate_fn, bs, n_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=BucketBatchSampler(val_ds.lengths, bs, shuffle=False),
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_sampler=BucketBatchSampler(test_ds.lengths, bs, shuffle=False),
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    if stage == 3:
        hw_ds = HandwrittenDataset(config)
        synthetic_ds = SyntheticDataset(config)

        n_val = max(1, int(len(hw_ds) * 0.2))
        n_train = len(hw_ds) - n_val
        hw_train, hw_val = random_split(
            hw_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        w = config.dataset_weights_stage3
        train_loader = _make_weighted_loader(
            [hw_train, synthetic_ds],
            [w["handwritten"], w["synthetic"]],
            collate_fn, bs, n_workers,
        )
        val_loader = DataLoader(
            hw_val,
            batch_size=bs,
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=True,
            shuffle=False,
        )
        return train_loader, val_loader, None

    raise ValueError(f"Unknown training stage: {stage}. Expected 1, 2, or 3.")


def build_dataloaders(
    data_dir: str,
    tokenizer: LaTeXTokenizer,
    batch_size: int = 16,
    max_len: int = 500,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Legacy builder for test_pipeline.py. Reads raw PNGs, no cache required."""
    collate_fn = CollateFunction(tokenizer, max_len=max_len)

    train_ds = RawIm2LatexDataset(data_dir, split="train")
    val_ds = RawIm2LatexDataset(data_dir, split="validate")
    test_ds = RawIm2LatexDataset(data_dir, split="test")

    def lengths_of(ds: RawIm2LatexDataset) -> list[int]:
        return [len(ds.formulas[idx]) for _, idx in ds.samples]

    return (
        DataLoader(train_ds, batch_sampler=BucketBatchSampler(lengths_of(train_ds), batch_size, True), collate_fn=collate_fn, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_sampler=BucketBatchSampler(lengths_of(val_ds),   batch_size, False), collate_fn=collate_fn, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_sampler=BucketBatchSampler(lengths_of(test_ds),  batch_size, False), collate_fn=collate_fn, num_workers=num_workers, pin_memory=True),
    )
