from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RuntimeConfig


# =============================================================================
# Abstract Base Classes
# =============================================================================


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


# =============================================================================
# Model Registry
# =============================================================================


REGISTRY: dict[str, ModuleType] = {}
"""Architecture name → module with (ModelConfig, ForwardPass, create_kv_cache, create_adapter_cache)."""

HF_TO_PIE_ARCH: dict[str, str] = {}
"""HuggingFace model_type → PIE architecture name."""


def register(
    name: str,
    module: ModuleType,
    *,
    aliases: tuple[str, ...] = (),
    hf_model_types: tuple[str, ...] = (),
):
    """Register a model architecture.

    Args:
        name: Primary architecture name (e.g., "llama3")
        module: Module containing ModelConfig, ForwardPass, create_kv_cache, create_adapter_cache
        aliases: Additional names that map to the same module
        hf_model_types: HuggingFace model_type strings that map to this architecture
    """
    REGISTRY[name] = module
    for alias in aliases:
        REGISTRY[alias] = module
    for hf_type in hf_model_types:
        HF_TO_PIE_ARCH[hf_type] = name


def get_module(arch_name: str) -> ModuleType:
    """Look up model module by architecture name.

    Raises:
        ValueError: If architecture is not registered
    """
    mod = REGISTRY.get(arch_name)
    if mod is None:
        raise ValueError(
            f"Unsupported architecture: {arch_name!r}. "
            f"Registered: {sorted(REGISTRY.keys())}"
        )
    return mod


# =============================================================================
# Register All Architectures
# =============================================================================

import torch
from . import llama3, qwen2, qwen3, gemma2, gemma3, mistral3, olmo3, gpt_oss

register("llama3", llama3, aliases=("l4ma",), hf_model_types=("llama",))
register("qwen2", qwen2, hf_model_types=("qwen2",))
register("qwen3", qwen3, hf_model_types=("qwen3",))
register("gemma2", gemma2, hf_model_types=("gemma2",))
register("gemma3", gemma3, hf_model_types=("gemma3_text",))
register("mistral3", mistral3, hf_model_types=("mistral3",))
register("olmo3", olmo3, hf_model_types=("olmo3",))
register("gptoss", gpt_oss, hf_model_types=("gptoss", "gpt_oss"))

from . import dummy as _dummy_mod
register("dummy", _dummy_mod)
