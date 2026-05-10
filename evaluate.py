"""Оценка обученной модели на test split.

Использование:
    python evaluate.py --checkpoint checkpoints/best_pretrain.pth --stage 1
    python evaluate.py --checkpoint checkpoints/best_mixed.pth --stage 2 --greedy
    python evaluate.py --checkpoint ... --save-predictions preds.json --plot-history
"""
import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from config import load_config
from data.dataset import build_multi_dataloaders
from data.tokenizer import LaTeXTokenizer
from model.model import Notes2LaTeX, count_parameters
from utils.beam_search import beam_search
from utils.metrics import (
    bleu_score, character_error_rate, edit_distance_score, exact_match,
)
from utils.visualization import plot_learning_curves, show_predictions


@torch.no_grad()
def greedy_decode_batch(model, images, src_kpm, tokenizer, device, max_len=200):
    """Жадная декодировка батча — копия из train.py для независимости evaluate."""
    from data.tokenizer import EOS_ID, PAD_ID, SOS_ID
    model.eval()
    B = images.shape[0]
    memory, memory_kpm = model.encoder(images, src_key_padding_mask=src_kpm)

    generated = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = model.decoder(generated, memory, memory_key_padding_mask=memory_kpm)
        next_ids = logits[:, -1, :].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_ID), next_ids)
        generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
        finished = finished | (next_ids == EOS_ID)
        if finished.all():
            break

    return [tokenizer.decode(generated[i].tolist()) for i in range(B)]


@torch.no_grad()
def run_evaluation(model, loader, tokenizer, config, device,
                   limit_batches=None, use_greedy=False):
    model.eval()
    all_predictions: list[str] = []
    all_references:  list[str] = []

    iterator = enumerate(loader)
    pbar = tqdm(iterator, total=limit_batches or len(loader), desc="eval")

    for batch_idx, (images, src_kpm, tgt_ids) in pbar:
        images  = images.to(device)
        src_kpm = src_kpm.to(device)
        tgt_ids = tgt_ids.to(device)

        # Эталоны
        references = [tokenizer.decode(ids.tolist()) for ids in tgt_ids]
        all_references.extend(references)

        # Предсказания
        if use_greedy:
            predictions = greedy_decode_batch(model, images, src_kpm, tokenizer, device)
        else:
            # Beam search по одному изображению — медленно, но корректно.
            predictions = []
            for i in range(images.shape[0]):
                pred = beam_search(
                    model, images[i:i+1], tokenizer, config,
                    src_key_padding_mask=src_kpm[i:i+1],
                )
                predictions.append(pred)

        all_predictions.extend(predictions)

        if limit_batches and batch_idx + 1 >= limit_batches:
            break

    return all_predictions, all_references


def compute_all_metrics(predictions: list[str], references: list[str]) -> dict:
    return {
        "n_samples":            len(predictions),
        "exact_match":          exact_match(predictions, references),
        "character_error_rate": character_error_rate(predictions, references),
        "edit_distance_score":  edit_distance_score(predictions, references),
        "bleu_score":           bleu_score(predictions, references),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="rtx4060_8gb")
    parser.add_argument("--checkpoint", required=True,
                        help="путь к чекпоинту (.pth)")
    parser.add_argument("--tokenizer", default="data_cache/tokenizer.json")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="на каком datasets-наборе оценивать (1=im2latex test)")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--greedy", action="store_true",
                        help="greedy decode вместо beam search (быстрее)")
    parser.add_argument("--save-predictions", default=None,
                        help="JSON-файл куда сохранить пары pred/ref")
    parser.add_argument("--plot-history", action="store_true",
                        help="нарисовать графики обучения из history_*.json")
    parser.add_argument("--show-predictions", type=int, default=0,
                        help="нарисовать N примеров с GT/PRED в PNG (0=skip)")

    # для слабого железа
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.max_width is not None:
        overrides["max_width"] = args.max_width

    config = load_config(args.profile, **overrides)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = LaTeXTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = Notes2LaTeX(config, tokenizer.vocab_size).to(device)
    print(f"Параметров: {count_parameters(model):,}")

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  stage={state.get('stage_name', '?')} epoch={state.get('epoch', '?')}")
    if "val_loss" in state:
        print(f"  train val_loss={state['val_loss']:.4f} val_acc={state.get('val_acc', 0):.3f}")

    # Test loader. Для stage 1 build_multi_dataloaders возвращает test_loader,
    # для stage 2/3 — None, тогда оцениваем на val.
    train_loader, val_loader, test_loader = build_multi_dataloaders(
        config, tokenizer, stage=args.stage,
    )
    loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "val"
    print(f"Evaluating on {split_name} split: {len(loader)} batches")

    # Decode mode
    mode = "greedy" if args.greedy else f"beam_size={config.beam_size}"
    print(f"Decode: {mode}")

    t0 = time.time()
    predictions, references = run_evaluation(
        model, loader, tokenizer, config, device,
        limit_batches=args.limit_batches, use_greedy=args.greedy,
    )
    dt = time.time() - t0

    metrics = compute_all_metrics(predictions, references)

    print(f"\n=== Results ({metrics['n_samples']} samples in {dt:.1f}s) ===")
    print(f"  Exact Match:           {metrics['exact_match']:.4f}")
    print(f"  Character Error Rate:  {metrics['character_error_rate']:.4f}")
    print(f"  Edit Distance Score:   {metrics['edit_distance_score']:.4f}  (1=identical)")
    print(f"  BLEU-4:                {metrics['bleu_score']:.4f}")

    # --- Save predictions JSON ---
    if args.save_predictions:
        out = {
            "checkpoint": args.checkpoint,
            "stage": args.stage,
            "split": split_name,
            "decode_mode": mode,
            "metrics": metrics,
            "samples": [
                {"prediction": p, "reference": r}
                for p, r in zip(predictions, references)
            ],
        }
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to {args.save_predictions}")

    # --- Plot history ---
    if args.plot_history:
        stage_name = state.get("stage_name", "pretrain")
        history_path = os.path.join(config.checkpoint_dir, f"history_{stage_name}.json")
        if os.path.exists(history_path):
            plot_learning_curves(history_path, config.plots_dir)
        else:
            print(f"history not found: {history_path}")

    # --- Show predictions PNG ---
    if args.show_predictions > 0:
        # Используем val_loader.dataset (или test) как источник примеров
        dataset = loader.dataset
        out_path = os.path.join(config.plots_dir, f"predictions_{split_name}.png")
        os.makedirs(config.plots_dir, exist_ok=True)
        show_predictions(model, dataset, tokenizer, device,
                         n=args.show_predictions, save_path=out_path)


if __name__ == "__main__":
    main()
