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
        # Ширина препроцессенной картинки — для вторичной сортировки в
        # BucketBatchSampler (батчи однородны по ширине → encoder не считает
        # белый паддинг). 0 если ключа нет — graceful degradation.
        self.widths: list[int] = [e.get("width", 0) for e in entries]
        # updated by train.py before each epoch
        self.elastic_p: float = 0.0
        self.elastic_alpha: int = 0
        self.elastic_sigma: int = 0
        self.strength: float = 0.0
        self.max_length: int = 512
        # Per-dataset elastic multiplier. Устанавливается build_multi_dataloaders
        # из config.elastic_factor_<dataset_type>.
        self.elastic_factor: float = 1.0
        # Per-dataset grid_aug. effective_prob = grid_aug_prob × grid_aug_factor.
        self.grid_aug_prob: float = 0.0
        self.grid_aug_factor: float = 1.0
        # Все остальные aug-параметры (probabilities + magnitudes из config).
        # Заполняется в build_multi_dataloaders. Дефолт — пустой dict (= захардкоженные defaults).
        self.aug_kwargs: dict = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        npy_path, formula = self.samples[idx]
        img = np.load(npy_path)
        img = apply_augmentations(
            img, self.dataset_type,
            self.elastic_p, self.elastic_alpha, self.elastic_sigma, self.strength,
            elastic_factor=self.elastic_factor,
            grid_aug_prob=self.grid_aug_prob * self.grid_aug_factor,
            **self.aug_kwargs,
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
        self.widths = dataset.widths
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

        # Двухуровневая сортировка:
        #   primary   — длина формулы, квантованная в бакеты по LEN_BUCKET
        #               токенов (внутри бакета длины "одинаковы" для целей
        #               паддинга токенов — сохраняет старое поведение).
        #   secondary — ширина картинки: батч получается однородным по ширине,
        #               collate паддит до близкого max_w, encoder/CNN не тратят
        #               compute на белый паддинг.
        # Шум на обоих уровнях сохраняет перемешивание батчей между эпохами.
        LEN_BUCKET = 32
        if self.shuffle:
            keyed = {
                i: (
                    int((self.lengths[i] + random.uniform(-16, 16)) // LEN_BUCKET),
                    self.widths[i] + random.uniform(-30, 30),
                )
                for i in indices
            }
            indices.sort(key=lambda i: keyed[i])
        else:
            # Детерминированно: длина, затем ширина.
            indices.sort(key=lambda i: (self.lengths[i], self.widths[i]))

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
            # Fixed-seed shuffle of batch ORDER so val sampling (n_em_batches,
            # val_limit_batches) sees a representative length mix instead of
            # only the shortest sequences. Within-batch order kept.
            rng = random.Random(0)
            rng.shuffle(batches)
            yield from batches

    def __len__(self) -> int:
        # Restore global RNG state after the call so __len__ has no side effects.
        saved = random.getstate()
        n = len(self._generate_batches())
        random.setstate(saved)
        return n


class CollateFunction:
    # Bucket-padding для совместимости с torch.compile:
    # без бакетов BucketBatchSampler даёт ~30-50 уникальных ширин и длин →
    # каждая форма рекомпилируется (cache_size_limit ловится). С бакетами —
    # ровно len(WIDTH_BUCKETS) × len(TGT_BUCKETS) уникальных пар форм.
    # Бакеты подобраны под распределение im2latex (median width≈400, p95≈1200)
    # с лёгкими «лестничками» в часто используемом диапазоне.
    WIDTH_BUCKETS: tuple[int, ...] = (256, 384, 512, 768, 1024, 1536, 2048, 2800)
    TGT_BUCKETS:   tuple[int, ...] = (64, 96, 128, 192, 256, 384, 512)

    def __init__(self, tokenizer: LaTeXTokenizer, max_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    @staticmethod
    def _bucket_up(value: int, buckets: tuple[int, ...]) -> int:
        """Округляет вверх до ближайшего бакета (последний бакет — потолок)."""
        for b in buckets:
            if value <= b:
                return b
        return buckets[-1]

    def __call__(
        self, batch: list[tuple[torch.Tensor, str]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает (images, src_key_padding_mask, tgt_ids).

        - images: [B, 1, max_h, max_w] — паддинг справа значением 1.0 (белый).
          max_w округлён вверх до WIDTH_BUCKETS для стабильности torch.compile.
        - src_key_padding_mask: [B, max_w] — True там, где пиксельный паддинг
          (нужно энкодеру, чтобы cross-attn декодера не attend'ил к шуму).
          Маска по высоте не делается: после CNN высота схлопывается в 1
          через AdaptiveAvgPool2d, поэтому вертикальный паддинг безвреден.
        - tgt_ids: [B, real_max_len] — токены с PAD'ами в конце.
          real_max_len округлён вверх до TGT_BUCKETS.
        """
        max_h = max(img.shape[1] for img, _ in batch)
        natural_w = max(img.shape[2] for img, _ in batch)
        max_w = self._bucket_up(natural_w, self.WIDTH_BUCKETS)

        batch_max_tok = max(len(self.tokenizer.tokenize(f)) for _, f in batch)
        natural_len = min(self.max_len, batch_max_tok + 2)
        real_max_len = min(self.max_len,
                           self._bucket_up(natural_len, self.TGT_BUCKETS))

        B = len(batch)
        src_kpm = torch.zeros(B, max_w, dtype=torch.bool)

        images, encoded_formulas = [], []
        for i, (image, formula) in enumerate(batch):
            _, h, w = image.shape
            if w < max_w:
                src_kpm[i, w:] = True
            if max_w - w > 0 or max_h - h > 0:
                image = torch.nn.functional.pad(image, (0, max_w - w, 0, max_h - h), value=1.0)
            images.append(image)
            encoded_formulas.append(self.tokenizer.encode(formula, max_len=real_max_len))

        tgt_ids = torch.tensor(encoded_formulas, dtype=torch.long)
        return torch.stack(images), src_kpm, tgt_ids


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


def _apply_dataset_factors(config: Config, dataset: "_CachedDataset", is_train: bool = True) -> None:
    """Установить per-dataset факторы и aug-параметры из конфига.

    is_train=False для val/test: grid_aug_prob=0, чтобы метрики были
    воспроизводимы между эпохами и измеряли target distribution без
    синтетических искажений.
    """
    elastic_map = {
        "im2latex":    config.elastic_factor_im2latex,
        "synthetic":   config.elastic_factor_synthetic,
        "handwritten": config.elastic_factor_handwritten,
    }
    grid_map = {
        "im2latex":    config.grid_aug_factor_im2latex,
        "synthetic":   config.grid_aug_factor_synthetic,
        "handwritten": config.grid_aug_factor_handwritten,
    }
    dataset.elastic_factor = elastic_map.get(dataset.dataset_type, 1.0)
    dataset.grid_aug_factor = grid_map.get(dataset.dataset_type, 1.0)
    dataset.grid_aug_prob = config.grid_aug_prob if is_train else 0.0

    # Остальные aug-параметры — общие для всех датасетов.
    dataset.aug_kwargs = {
        "dilate_erode_prob":    config.aug_dilate_erode_prob,
        "noise_prob":           config.aug_noise_prob,
        "noise_sigma":          config.aug_noise_sigma,
        "blur_prob":            config.aug_blur_prob,
        "brightness_prob":      config.aug_brightness_prob,
        "brightness_limit":     config.aug_brightness_limit,
        "contrast_limit":       config.aug_contrast_limit,
        "affine_prob":          config.aug_affine_prob,
        "affine_rotate_deg":    config.aug_affine_rotate_deg,
        "affine_scale_pct":     config.aug_affine_scale_pct,
        "grid_cell_range":      (config.grid_cell_min, config.grid_cell_max),
        "grid_intensity_range": (config.grid_intensity_min, config.grid_intensity_max),
        "grid_jitter":          config.grid_line_jitter,
        "grid_line_noise":      config.grid_line_noise,
    }


# Backward-compat alias (если где-то осталось старое имя)
_apply_elastic_factor = _apply_dataset_factors


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy per-worker. PyTorch seeds random/torch automatically but not numpy."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


def _make_loader_kwargs(config: Config, collate_fn: CollateFunction) -> dict:
    # persistent_workers НЕ используем: воркеры форкаются на первой итерации и
    # держат snapshot датасета — curriculum-апдейты elastic_p/strength из main
    # никогда не доходят до них. Респавн каждую эпоху подхватывает актуальные
    # значения и заодно ресидит numpy через worker_init_fn.
    kwargs: dict = {
        "collate_fn": collate_fn,
        "num_workers": config.num_workers,
        "pin_memory": True,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["worker_init_fn"] = _worker_init_fn
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
    val_bs = config.val_batch_size or config.batch_size

    if stage == 1:
        train_ds = Im2LatexDataset(config.cache_dir, split="train", max_length=config.tokenizer_max_len)
        val_ds   = Im2LatexDataset(config.cache_dir, split="validate")
        test_ds  = Im2LatexDataset(config.cache_dir, split="test")
        _apply_elastic_factor(config, train_ds, is_train=True)
        _apply_elastic_factor(config, val_ds,   is_train=False)
        _apply_elastic_factor(config, test_ds,  is_train=False)

        train_loader = DataLoader(train_ds, batch_sampler=BucketBatchSampler(train_ds, config.batch_size, shuffle=True),  **kw)
        val_loader   = DataLoader(val_ds,   batch_sampler=BucketBatchSampler(val_ds,   val_bs, shuffle=False), **kw)
        test_loader  = DataLoader(test_ds,  batch_sampler=BucketBatchSampler(test_ds,  val_bs, shuffle=False), **kw)
        return train_loader, val_loader, test_loader

    if stage == 2:
        im2latex_ds  = Im2LatexDataset(config.cache_dir, split="train")
        synthetic_ds = SyntheticDataset(config.cache_dir)
        val_ds       = Im2LatexDataset(config.cache_dir, split="validate")
        _apply_elastic_factor(config, im2latex_ds,  is_train=True)
        _apply_elastic_factor(config, synthetic_ds, is_train=True)
        _apply_elastic_factor(config, val_ds,       is_train=False)

        named = [("im2latex", im2latex_ds), ("synthetic", synthetic_ds)]
        combined = ConcatDataset([ds for _, ds in named])
        weights = _compute_sample_weights(named, config.dataset_weights_stage2)
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)

        train_loader = DataLoader(combined, sampler=sampler, batch_size=config.batch_size,
                                  drop_last=True, **kw)
        val_loader   = DataLoader(val_ds, batch_sampler=BucketBatchSampler(val_ds, val_bs, shuffle=False), **kw)
        return train_loader, val_loader, None

    if stage == 3:
        hw_train_ds  = HandwrittenDataset(config.cache_dir, split="train")
        hw_val_ds    = HandwrittenDataset(config.cache_dir, split="val")
        synthetic_ds = SyntheticDataset(config.cache_dir)
        _apply_elastic_factor(config, hw_train_ds,  is_train=True)
        _apply_elastic_factor(config, hw_val_ds,    is_train=False)
        _apply_elastic_factor(config, synthetic_ds, is_train=True)

        named = [("handwritten", hw_train_ds), ("synthetic", synthetic_ds)]
        combined = ConcatDataset([ds for _, ds in named])
        weights = _compute_sample_weights(named, config.dataset_weights_stage3)
        # epoch size: enough to see each handwritten sample ~once
        hw_ratio = config.dataset_weights_stage3.get("handwritten", 0.82)
        num_samples = max(len(combined), int(len(hw_train_ds) / hw_ratio))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

        train_loader = DataLoader(combined, sampler=sampler, batch_size=config.batch_size,
                                  drop_last=True, **kw)
        val_loader   = DataLoader(hw_val_ds, batch_sampler=BucketBatchSampler(hw_val_ds, val_bs, shuffle=False), **kw)
        return train_loader, val_loader, None

    raise ValueError(f"stage должен быть 1, 2 или 3, получено: {stage}")
