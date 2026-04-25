"""
Работа с кэшем предобработанных изображений (data_cache/).

"""


def load_manifest(cache_dir: str, dataset_name: str) -> list[dict]:
    raise NotImplementedError


def write_manifest_entry(manifest_path: str, entry: dict) -> None:
    raise NotImplementedError


def cache_exists(cache_dir: str, dataset_name: str) -> bool:
    raise NotImplementedError
