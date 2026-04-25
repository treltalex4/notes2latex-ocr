def bleu_score(predictions: list[str], references: list[str]) -> float:
    raise NotImplementedError


def exact_match(predictions: list[str], references: list[str]) -> float:
    raise NotImplementedError


def token_accuracy(logits, targets, pad_idx: int = 0) -> float:
    raise NotImplementedError


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    raise NotImplementedError


def edit_distance_score(predictions: list[str], references: list[str]) -> float:
    raise NotImplementedError
