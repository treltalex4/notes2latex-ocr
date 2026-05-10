import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rapidfuzz.distance import Levenshtein

from data.tokenizer import LaTeXTokenizer


def token_accuracy(logits, targets, pad_idx: int = 0) -> float:
    predictions = logits.argmax(dim=-1)        # [B, T]
    mask = targets != pad_idx                  # True там где НЕ паддинг
    n_real = mask.sum()
    if n_real == 0:
        return 0.0
    correct = (predictions == targets) & mask
    return (correct.sum() / n_real).item()


def exact_match(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    """CER = edit_distance(pred, ref) / len(ref), усреднено по парам.

    Пустой эталон пропускается. CER=0 идеально, CER>1 возможен если pred много длиннее.
    """
    if not predictions:
        return 0.0
    total = 0.0
    count = 0
    for pred, ref in zip(predictions, references):
        if not ref:
            continue
        total += Levenshtein.distance(pred, ref) / len(ref)
        count += 1
    return total / count if count else 0.0


def edit_distance_score(predictions: list[str], references: list[str]) -> float:
    """Нормированный Levenshtein как схожесть: 1 = идентично, 0 = всё разное.

    Формула: 1 - dist / max(len(pred), len(ref)). Удобнее CER для отображения
    "положительной" метрики качества.
    """
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        maxlen = max(len(pred), len(ref))
        if maxlen == 0:
            scores.append(1.0)   # обе пустые → совпадение
            continue
        scores.append(1.0 - Levenshtein.distance(pred, ref) / maxlen)
    return sum(scores) / len(scores)


def bleu_score(predictions: list[str], references: list[str]) -> float:
    """BLEU-4 через nltk. Токенизация — через LaTeXTokenizer для честного сравнения
    LaTeX-команд (\\frac как один токен, не как 5 символов).
    """
    if not predictions:
        return 0.0
    smoothing = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = LaTeXTokenizer.tokenize(pred)
        ref_tokens  = LaTeXTokenizer.tokenize(ref)
        if not pred_tokens or not ref_tokens:
            continue
        scores.append(sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),   # BLEU-4 (одинаковые веса 1..4-gram)
            smoothing_function=smoothing,
        ))
    return sum(scores) / len(scores) if scores else 0.0
