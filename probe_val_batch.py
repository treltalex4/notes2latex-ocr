"""
Probe максимальный val_batch_size: пробует ряд размеров, запускает несколько
батчей с full forward + greedy_decode, печатает пиковую VRAM и время.

Usage:
    python probe_val_batch.py
    python probe_val_batch.py --sizes 32 64 96 128 192 256
    python probe_val_batch.py --profile rtx5090_32gb
"""

import argparse
import sys
import time

import torch

from config import load_config
from data.dataset import build_multi_dataloaders
from data.tokenizer import LaTeXTokenizer, PAD_ID
from model.model import Notes2LaTeX
from train import (
    _compile_model, _mark_dynamic_inputs, greedy_decode_batch,
)


def probe_size(config, tokenizer, device, val_bs, n_batches=3, use_compile=False):
    """Возвращает (peak_mb, sec_per_batch) или None если OOM."""
    # Patch config с новым val_batch_size
    config.val_batch_size = val_bs

    model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
    if use_compile:
        model = _compile_model(model, config)

    amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    use_amp = config.use_amp and device.type == "cuda"

    try:
        _, val_loader, _ = build_multi_dataloaders(config, tokenizer, stage=1)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # n_batches forward + greedy_decode
        t_total = 0.0
        n_done = 0
        with torch.no_grad():
            for batch_idx, (images, src_kpm, tgt_ids) in enumerate(val_loader):
                images  = images.to(device)
                src_kpm = src_kpm.to(device)
                tgt_ids = tgt_ids.to(device)

                tgt_input = tgt_ids[:, :-1]
                _mark_dynamic_inputs(images, tgt_input, src_kpm)

                t0 = time.time()
                # Full forward (val_loss path)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _ = model(images, tgt_input, src_key_padding_mask=src_kpm)
                # Greedy decode (EM path) — самый тяжёлый по VRAM из-за kv_cache
                _ = greedy_decode_batch(model, images, src_kpm, tokenizer, device,
                                        max_len=config.beam_max_len)
                torch.cuda.synchronize()
                t_total += time.time() - t0
                n_done += 1
                if n_done >= n_batches:
                    break

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        sec_per_batch = t_total / max(1, n_done)
        return peak_mb, sec_per_batch

    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="rtx4060_8gb")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[24, 48, 64, 96, 128, 192, 256, 384])
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--n-batches", type=int, default=3,
                        help="Сколько батчей прогнать на каждом размере для замера")
    parser.add_argument("--use-compile", action="store_true",
                        help="Включить torch.compile (memory profile отличается от eager)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA недоступен")
        sys.exit(1)

    config = load_config(args.profile)
    device = torch.device("cuda")
    tokenizer = LaTeXTokenizer.load(args.tokenizer)

    total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {total_mb:.0f} MB")
    print(f"Profile: {args.profile} (batch_size={config.batch_size})")
    print(f"compile: {args.use_compile}\n")

    results = []
    for bs in args.sizes:
        print(f"  Probing val_batch_size={bs} ...", flush=True)
        r = probe_size(config, tokenizer, device, bs, args.n_batches, args.use_compile)
        if r is None:
            print(f"    OOM")
            break
        peak_mb, sec = r
        pct = 100 * peak_mb / total_mb
        results.append((bs, peak_mb, pct, sec))
        print(f"    peak={peak_mb:.0f} MB ({pct:.0f}% VRAM)  time/batch={sec:.2f}s")

    print(f"\n{'='*60}")
    print(f"{'val_batch':<12}{'peak_MB':<12}{'%VRAM':<10}{'sec/batch':<12}{'sec/sample':<12}")
    print(f"{'-'*60}")
    for bs, mb, pct, sec in results:
        print(f"{bs:<12}{mb:<12.0f}{pct:<10.0f}{sec:<12.2f}{sec/bs:<12.4f}")

    if results:
        best = min(results, key=lambda r: r[3] / r[0])  # лучший sec/sample
        print(f"\nЛучший по throughput: val_batch_size={best[0]} "
              f"({best[3]/best[0]*1000:.1f} ms/sample, peak {best[1]:.0f} MB)")
        # Рекомендация: 80% потолка по VRAM
        safe = [r for r in results if r[2] <= 80]
        if safe:
            rec = max(safe, key=lambda r: r[0])
            print(f"Безопасно (≤80% VRAM):  val_batch_size={rec[0]} "
                  f"({rec[2]:.0f}% VRAM, {rec[3]/rec[0]*1000:.1f} ms/sample)")


if __name__ == "__main__":
    main()
