import torch
from dataclasses import dataclass
from typing import Callable, Any

from pie_backend.loader import Schema, Source, WeightStore


@dataclass
class MockConfig:
    device: torch.device = torch.device("cpu")
    world_size: int = 1
    rank: int = 0
    quantization: Any = None


def test_gather_transform_fix():
    """Test that Source.gather().transform() works without raising ValueError."""
    print("Testing gather().transform() fix...")
    mock_tensors = {
        "a": torch.tensor([1.0]),
        "b": torch.tensor([2.0]),
        "c": torch.tensor([3.0]),
    }

    def reader(name):
        return mock_tensors[name]

    def transform_fn(tensors, kwargs):
        # Just sum them up
        return torch.sum(torch.stack(tensors))

    schema = Schema("test").define(
        "fused", Source.gather(["a", "b", "c"]).transform(transform_fn)
    )

    config = MockConfig()
    store = schema.load(reader, config, num_layers=0)

    assert "fused" in store
    assert store.get("fused").item() == 6.0
    print("SUCCESS: gather().transform() worked as expected.")


def test_gather_without_transform_raises():
    """Test that Source.gather() without .transform() still raises ValueError."""
    print("Testing gather() without transform still raises...")
    mock_tensors = {"a": torch.tensor([1.0])}

    def reader(name):
        return mock_tensors[name]

    schema = Schema("test").define("gathered", Source.gather(["a"]))

    config = MockConfig()
    try:
        schema.load(reader, config, num_layers=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Source.gather() should be used with .transform()" in str(e)
    print("SUCCESS: gather() without transform raised ValueError.")


if __name__ == "__main__":
    test_gather_transform_fix()
    test_gather_without_transform_raises()
    print("\nAll tests passed successfully!")
