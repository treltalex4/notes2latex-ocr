def get_elastic_params(epoch: int, total_epochs: int, schedule: list) -> tuple:
    raise NotImplementedError


def get_max_length(epoch: int, total_epochs: int, schedule: list) -> int:
    raise NotImplementedError


def get_augment_strength(epoch: int, total_epochs: int, max_strength: float) -> float:
    raise NotImplementedError
