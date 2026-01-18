"""Dataset loaders for benchmarking."""

from .base import Request, DatasetLoader
from .random_dataset import RandomDataset
from .sharegpt import ShareGPTDataset

DATASETS = {
    "random": RandomDataset,
    "sharegpt": ShareGPTDataset,
}

def get_dataset(name: str, **kwargs) -> DatasetLoader:
    """Get a dataset loader by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name](**kwargs)

__all__ = ["Request", "DatasetLoader", "get_dataset", "DATASETS"]
