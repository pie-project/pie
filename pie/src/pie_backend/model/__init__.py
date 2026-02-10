from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RuntimeConfig


# Model architecture specification (abstract base class)
@dataclass
class ModelConfig(ABC):
    """
    Abstract base class for model architecture specifications.

    Each model (e.g., llama3) should define its own ModelConfig that inherits
    from this class and specifies architecture-specific parameters.
    """

    num_vocabs: int

    @staticmethod
    @abstractmethod
    def from_dict(spec: dict) -> "ModelConfig":
        """Construct a ModelConfig object from a configuration dictionary."""
        pass

    @abstractmethod
    def eval_max_num_kv_pages(self, runtime_config: "RuntimeConfig") -> int:
        """Evaluate the maximum number of KV pages based on available memory."""
        pass


@dataclass
class Buffer(ABC):
    """Abstract base class for model buffers (e.g., KV cache)."""

    model_config: ModelConfig
    runtime_config: "RuntimeConfig"

    @staticmethod
    @abstractmethod
    def from_config(
        model_config: ModelConfig,
        runtime_config: "RuntimeConfig",
    ) -> "Buffer":
        """Create a Buffer object from model and runtime configurations."""
        pass
