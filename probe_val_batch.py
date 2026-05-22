"""
Probe максимальный val_batch_size: пробует ряд размеров, печатает пиковую
VRAM и throughput. Три режима, соответствующих use case'у:

  --mode tune  (default): для подбора val_batch_size в tune.py.
      Каждый размер замеряется на ~одинаковой выборке (--n-samples примеров),
      forward+greedy на каждом батче (моделирует EM-метрику tune.py).
      n_batches = ceil(--n-samples / val_bs), не меньше --min-batches.
      Цель — честное сравнение throughput разных размеров.

  --mode train: для подбора val_batch_size в train.py.
      Каждый размер замеряется на ПОЛНОМ val_loader (все 9311 im2latex
      примеров). Первые --n-em-batches батчей идут с greedy_decode
      (моделирует val_em), остальные — только forward (моделирует
      val_loss/val_acc). Точно отражает время одного epoch'а validate().
      Цель — найти оптимум для реального validate() в train.

  --mode evaluate: для подбора val_batch_size в evaluate.py.
      ПОЛНЫЙ val_loader, на КАЖДОМ батче beam_search_batch (а не forward).
      Опционально + greedy_decode_batch если --compare-greedy (моделирует
      `evaluate.py --compare-greedy`). У beam VRAM ≈ B×K (K=beam_size),
      поэтому потолок val_bs значимо ниже чем в tune/train.
      Цель — найти максимальный val_bs для evaluate без OOM.

Usage:
    python probe_val_batch.py --mode tune --sizes 24 48 64 96
    python probe_val_batch.py --mode train --sizes 32 64 96 --n-em-samples 960
    python probe_val_batch.py --mode evaluate --sizes 8 16 24 32
    python probe_val_batch.py --mode evaluate --sizes 16 24 --compare-greedy
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
from utils.beam_search import beam_search_batch


def _is_oom_error(exc: BaseException) -> bool:
    """OOM в PyTorch разносится через несколько классов в зависимости от версии:
    torch.cuda.OutOfMemoryError (старое API), torch.AcceleratorError (новое),
    либо обычный RuntimeError с 'out of memory' в сообщении."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if hasattr(torch, "AcceleratorError") and isinstance(exc, torch.AcceleratorError):
        return True
    return "out of memory" in str(exc).lower()


def probe_size(config, tokenizer, device, val_bs, n_batches=None,
               run_forward=True, greedy_limit=0, beam_limit=0,
               use_compile=False):
    """Замер одного val_batch_size — гибкий под все три режима.

    n_batches: общее число батчей. None = весь val_loader.
    run_forward: teacher-forced forward на каждом батче (val_loss/val_acc path).
                 True для tune/train, False для evaluate (там forward не нужен).
    greedy_limit: greedy_decode_batch на первых N батчах. 0 = skip полностью.
    beam_limit:   beam_search_batch на первых N батчах. 0 = skip полностью.
                  beam требует тензоров B*K, поэтому VRAM-профиль другой.

    Возвращает (peak_mb, sec_per_batch, n_done) или None если OOM.
    """
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

        total_target = n_batches if n_batches is not None else len(val_loader)

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
                # 1. Forward (val_loss/val_acc path) — в tune/train, не в evaluate.
                if run_forward:
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        _ = model(images, tgt_input, src_key_padding_mask=src_kpm)
                # 2. Greedy decode — EM в tune/train, опциональный для evaluate (compare).
                if batch_idx < greedy_limit:
                    _ = greedy_decode_batch(model, images, src_kpm, tokenizer, device,
                                            max_len=config.beam_max_len)
                # 3. Beam search — основной decoder в evaluate. VRAM ~B*K раз больше
                # за счёт параллельных beam'ов и их KV-кэшей.
                if batch_idx < beam_limit:
                    _ = beam_search_batch(model, images, src_kpm, tokenizer, config)
                torch.cuda.synchronize()
                t_total += time.time() - t0
                n_done += 1
                if n_done >= total_target:
                    break

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        sec_per_batch = t_total / max(1, n_done)
        return peak_mb, sec_per_batch, n_done

    except Exception as e:
        if _is_oom_error(e):
            return None
        raise
    finally:
        del model
        # empty_cache сам может упасть после OOM (контекст в плохом состоянии).
        # Игнорируем — следующая итерация всё равно создаст новую модель и
        # CUDA-аллокатор переинициализируется.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "train", "evaluate"], default="tune",
                        help="tune: замер на ~одинаковой выборке (--n-samples), "
                             "greedy на каждом батче — соответствует EM в tune.py. "
                             "train: ПОЛНЫЙ val_loader, greedy только на первых "
                             "--n-em-batches батчах — соответствует validate() в train.py. "
                             "evaluate: ПОЛНЫЙ val_loader, beam_search_batch на каждом "
                             "батче (+ greedy если --compare-greedy) — соответствует "
                             "evaluate.py. VRAM выше из-за B×K beam-тензоров.")
    parser.add_argument("--profile", default="rtx4060_8gb")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[24, 48, 64, 96, 128, 192, 256, 384])
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    # --- tune mode args ---
    parser.add_argument("--n-samples", type=int, default=1500,
                        help="[tune mode] Целевое количество примеров для замера "
                             "на каждом размере. n_batches = ceil(n_samples / val_bs), "
                             "не меньше --min-batches. 1500 ≈ типичный объём "
                             "EM-метрики из config.")
    parser.add_argument("--min-batches", type=int, default=5,
                        help="[tune mode] Минимум батчей даже для больших val_bs.")
    parser.add_argument("--n-batches", type=int, default=None,
                        help="[tune mode] Override: фиксированное число батчей "
                             "вместо --n-samples. Только для debug.")
    parser.add_argument("--n-em-samples", type=int, default=960,
                        help="[оба режима] Зафиксировать число примеров под "
                             "greedy_decode (EM). em_batches = ceil(N / val_bs) → "
                             "одинаковая EM-выборка для всех bs (apples-to-apples). "
                             "В tune mode: None = greedy на каждом батче (conservative "
                             "VRAM). В train mode имеет приоритет над --n-em-batches.")
    # --- train mode args ---
    parser.add_argument("--n-em-batches", type=int, default=20,
                        help="[train mode] Override: фиксированное число первых "
                             "батчей с greedy_decode. Используется ТОЛЬКО если "
                             "--n-em-samples не задан (None). Дефолт 20 = "
                             "config.n_em_batches. Чаще используй --n-em-samples "
                             "для честного сравнения разных bs.")
    # --- evaluate mode args ---
    parser.add_argument("--compare-greedy", action="store_true",
                        help="[evaluate mode] Дополнительно запускать greedy_decode_batch "
                             "на каждом батче параллельно с beam — моделирует "
                             "`evaluate.py --compare-greedy`. ~2× медленнее, VRAM по "
                             "максимуму из двух decoder'ов.")
    # --- common ---
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
    print(f"Mode:    {args.mode}")
    print(f"compile: {args.use_compile}")
    if args.mode == "tune":
        if args.n_batches is None:
            print(f"Target samples per size: {args.n_samples} "
                  f"(min batches: {args.min_batches})")
        else:
            print(f"Fixed n_batches: {args.n_batches} (override)")
        em_msg = (f"first ceil({args.n_em_samples} / val_bs) batches"
                  if args.n_em_samples is not None
                  else "every batch (conservative VRAM, non-comparable throughput)")
        print(f"Greedy decode (EM): {em_msg}\n")
    elif args.mode == "train":
        # --n-em-samples приоритетнее --n-em-batches
        if args.n_em_samples is not None:
            em_msg = (f"first ceil({args.n_em_samples} / val_bs) batches "
                      f"(~{args.n_em_samples} EM samples for all bs)")
        else:
            em_msg = f"first {args.n_em_batches} batches (fixed batch count)"
        print(f"Full val_loader per size; forward on all + greedy on {em_msg}\n")
    else:
        # evaluate mode: full val_loader, beam_search_batch на каждом батче
        decoders = "beam + greedy (compare)" if args.compare_greedy else "beam only"
        print(f"Full val_loader per size; NO forward (как evaluate.py); "
              f"decoders: {decoders} (beam_size={config.beam_size})\n")

    results = []
    for bs in args.sizes:
        # Дефолты — будут переопределены под режим.
        run_forward = True
        greedy_limit = 0
        beam_limit = 0

        if args.mode == "tune":
            # Динамический подбор n_batches под целевую выборку — честное
            # сравнение разных bs (одинаковое число примеров).
            n_batches_target = (args.n_batches if args.n_batches is not None
                                else max(args.min_batches,
                                         (args.n_samples + bs - 1) // bs))
            # EM: либо greedy на всех (default, conservative VRAM) либо
            # ceil(--n-em-samples / bs) — apples-to-apples по greedy выборке.
            if args.n_em_samples is None:
                greedy_limit = n_batches_target   # все батчи
                em_label = "greedy=all"
            else:
                greedy_limit = max(1, (args.n_em_samples + bs - 1) // bs)
                em_label = f"greedy={greedy_limit} ({greedy_limit * bs} samples)"
            label = (f"n_batches={n_batches_target} "
                     f"(~{n_batches_target * bs} samples)  {em_label}")

        elif args.mode == "train":
            # Полный val_loader, forward+greedy на первых N батчах.
            n_batches_target = None
            if args.n_em_samples is not None:
                greedy_limit = max(1, (args.n_em_samples + bs - 1) // bs)
                label = (f"full val_loader  greedy={greedy_limit} batches "
                         f"({greedy_limit * bs} EM samples)")
            else:
                greedy_limit = args.n_em_batches
                label = (f"full val_loader  greedy={greedy_limit} batches "
                         f"({greedy_limit * bs} EM samples, scales with bs)")

        else:
            # evaluate: полный val_loader, beam на каждом батче, без forward.
            # Огромное число для greedy_limit/beam_limit = "на всех батчах".
            n_batches_target = None
            run_forward = False
            beam_limit = 10**9
            if args.compare_greedy:
                greedy_limit = 10**9
                label = "full val_loader  beam+greedy on every batch"
            else:
                label = "full val_loader  beam on every batch"

        print(f"  Probing val_batch_size={bs}  {label} ...", flush=True)
        r = probe_size(config, tokenizer, device, bs,
                       n_batches=n_batches_target,
                       run_forward=run_forward,
                       greedy_limit=greedy_limit,
                       beam_limit=beam_limit,
                       use_compile=args.use_compile)
        if r is None:
            print(f"    OOM")
            continue   # пробуем следующий размер (может быть меньше → влезет)
        peak_mb, sec, n_done = r
        pct = 100 * peak_mb / total_mb
        total_time = sec * n_done
        results.append((bs, peak_mb, pct, sec, n_done, total_time))
        print(f"    peak={peak_mb:.0f} MB ({pct:.0f}% VRAM)  "
              f"time/batch={sec:.2f}s  total={total_time:.1f}s  "
              f"({n_done} batches × {bs} = {n_done * bs} samples)")

    print(f"\n{'='*72}")
    print(f"{'val_batch':<11}{'n_batch':<9}{'peak_MB':<11}{'%VRAM':<9}"
          f"{'sec/batch':<11}{'ms/sample':<11}{'total_s':<8}")
    print(f"{'-'*72}")
    for bs, mb, pct, sec, nb, total_s in results:
        print(f"{bs:<11}{nb:<9}{mb:<11.0f}{pct:<9.0f}{sec:<11.2f}"
              f"{sec/bs*1000:<11.1f}{total_s:<8.1f}")

    if results:
        if args.mode in ("train", "evaluate"):
            # total_s == реальное время полного val-pass — главная метрика
            # (меньше = быстрее эпоха в train / быстрее оценка в evaluate).
            best = min(results, key=lambda r: r[5])
            label = "полному val-pass" if args.mode == "train" else "полной evaluate"
            print(f"\nЛучший по {label}: val_batch_size={best[0]} "
                  f"(total={best[5]:.1f}s, peak {best[1]:.0f} MB / {best[2]:.0f}% VRAM)")
        else:
            # tune mode: throughput на одинаковой выборке.
            best = min(results, key=lambda r: r[3] / r[0])
            print(f"\nЛучший по throughput: val_batch_size={best[0]} "
                  f"({best[3]/best[0]*1000:.1f} ms/sample, peak {best[1]:.0f} MB)")
        # Безопасный по VRAM (≤80%) — независимо от режима. Особенно важно для
        # evaluate где B×K расход растёт быстро.
        safe = [r for r in results if r[2] <= 80]
        if safe:
            rec = max(safe, key=lambda r: r[0])
            print(f"Безопасно (≤80% VRAM):  val_batch_size={rec[0]} "
                  f"({rec[2]:.0f}% VRAM, {rec[3]/rec[0]*1000:.1f} ms/sample)")


if __name__ == "__main__":
    main()
