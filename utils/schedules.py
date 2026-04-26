def get_elastic_params(epoch: int, total_epochs: int, schedule: list) -> tuple:
    """
    schedule = [(threshold_ratio, p, alpha, sigma), ...]
    Returns (p, alpha, sigma) for the current epoch.
    """
    ratio = epoch / max(total_epochs, 1)
    for threshold, p, alpha, sigma in schedule:
        if ratio <= threshold:
            return p, alpha, sigma
    return schedule[-1][1], schedule[-1][2], schedule[-1][3]


def get_max_length(epoch: int, total_epochs: int, schedule: list) -> int:
    """Returns the current token length limit from length_curriculum."""
    ratio = epoch / max(total_epochs, 1)
    for threshold, max_len in schedule:
        if ratio <= threshold:
            return max_len
    return schedule[-1][1]


def get_augment_strength(epoch: int, total_epochs: int, max_strength: float) -> float:
    """Linear ramp-up of non-elastic augmentation intensity."""
    return min(max_strength, max_strength * (epoch / max(total_epochs, 1)))
