"""Build the tokenizer vocabulary from cached manifests and save to JSON.

Run ONCE after prepare_data.py. Both train.py and evaluate need to open the
same vocab file — otherwise the model's embedding layer won't match what it
was trained on.

Usage:
    python build_tokenizer.py                                  # im2latex + synthetic
    python build_tokenizer.py --datasets im2latex
    python build_tokenizer.py --datasets im2latex synthetic handwritten
    python build_tokenizer.py --output data_cache/tokenizer.json
"""

from __future__ import annotations

import argparse
import os

from config import load_config
from data.cache import load_manifest, cache_exists
from data.tokenizer import LaTeXTokenizer


def _collect_formulas(cache_dir: str, datasets: list[str]) -> list[str]:
    formulas: list[str] = []
    for name in datasets:
        if not cache_exists(cache_dir, name):
            print(f"  [skip] cache for '{name}' not found — skipping")
            continue
        entries = load_manifest(cache_dir, name)
        train_entries = [e for e in entries if e.get("split", "train") == "train"]
        formulas.extend(e["formula"] for e in train_entries)
        print(f"  [{name}] +{len(train_entries)} formulas (train split)")
    return formulas


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LaTeXTokenizer vocabulary.")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["im2latex", "synthetic", "handwritten"],
        default=["im2latex", "synthetic"],
        help="Datasets to include in the vocabulary.",
    )
    parser.add_argument(
        "--profile", default="rtx4060_8gb",
        choices=["rtx4060_8gb", "rtx5090_32gb"],
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Defaults to <cache_dir>/tokenizer.json.",
    )
    args = parser.parse_args()

    config = load_config(args.profile)
    out_path = args.output or os.path.join(config.cache_dir, "tokenizer.json")

    print(f"Profile:    {args.profile}")
    print(f"Cache:      {os.path.abspath(config.cache_dir)}")
    print(f"Min freq:   {config.min_token_freq}")
    print(f"Datasets:   {args.datasets}\n")

    formulas = _collect_formulas(config.cache_dir, args.datasets)
    if not formulas:
        raise RuntimeError(
            "No formulas collected. "
            "Run prepare_data.py for the required datasets first."
        )
    print(f"\nTotal formulas: {len(formulas)}")

    tok = LaTeXTokenizer()
    tok.build_vocab(formulas, min_freq=config.min_token_freq)
    print(f"Vocabulary:     {tok.vocab_size} tokens")

    tok.save(out_path)
    print(f"Saved to:       {os.path.abspath(out_path)}")

    # Roundtrip check: reload and verify equality
    reloaded = LaTeXTokenizer.load(out_path)
    assert reloaded.vocab_size == tok.vocab_size, "Roundtrip: vocab sizes differ"
    assert reloaded.token2id == tok.token2id, "Roundtrip: token2id dicts differ"
    print("Roundtrip OK")


if __name__ == "__main__":
    main()
