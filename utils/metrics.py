import torch

def token_accuracy(logits, targets, pad_idx: int = 0) -> float:

    predictions = logits.argmax(dim=-1) # [B, T]
    mask = targets != pad_idx # правда там, где не падинг
    n_real = mask.sum()
    if n_real == 0:
        return 0.0
    correct = (predictions == targets)& mask
    return (correct.sum() / n_real).item()

def exact_match(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)



def bleu_score(predictions: list[str], references: list[str]) -> float:
    raise NotImplementedError



def character_error_rate(predictions: list[str], references: list[str]) -> float:
    raise NotImplementedError


def edit_distance_score(predictions: list[str], references: list[str]) -> float:
    raise NotImplementedError
