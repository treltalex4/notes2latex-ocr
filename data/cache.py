import json
import os


def load_manifest(cache_dir: str, dataset_name: str) -> list[dict]:
    path = os.path.join(cache_dir, dataset_name, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Кэш не найден для '{dataset_name}'. "
            f"Запустите: python prepare_data.py --datasets {dataset_name}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_stats(cache_dir: str, dataset_name: str) -> dict:
    path = os.path.join(cache_dir, dataset_name, "stats.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cache_exists(cache_dir: str, dataset_name: str) -> bool:
    path = os.path.join(cache_dir, dataset_name, "manifest.json")
    return os.path.exists(path)
