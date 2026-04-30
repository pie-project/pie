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


REGISTRY: dict[str, str | ModuleType] = {}
"""Architecture name → module path string (lazy) or loaded module (cached after first get_module)."""

HF_TO_PIE_ARCH: dict[str, str] = {}
"""HuggingFace model_type → PIE architecture name."""


class UnknownArchitectureError(ValueError):
    """Raised when an HF config can't be mapped to a pie internal arch."""


def register(
    name: str,
    module_path: str,
    *,
    aliases: tuple[str, ...] = (),
    hf_model_types: tuple[str, ...] = (),
):
    """Register a model architecture (lazy — module loads on first ``get_module``).

    Args:
        name: Primary architecture name (e.g., ``"llama3"``).
        module_path: Dotted path to the per-arch module (e.g.,
            ``"pie_driver.model.llama3"``). Imported lazily on first
            ``get_module(name)`` call so that callers reading only the
            HF→pie table don't pay the ``pie_kernels`` JIT cost (every
            per-arch module ``import pie_kernels as ops`` at top-level).
        aliases: Additional names that map to the same module.
        hf_model_types: HuggingFace ``model_type`` strings that map to
            this architecture (populates ``HF_TO_PIE_ARCH``).
    """
    REGISTRY[name] = module_path
    for alias in aliases:
        REGISTRY[alias] = module_path
    for hf_type in hf_model_types:
        HF_TO_PIE_ARCH[hf_type] = name


def get_module(arch_name: str) -> ModuleType:
    """Look up model module by architecture name (lazy-loads on first call).

    Raises:
        ValueError: If architecture is not registered.
    """
    entry = REGISTRY.get(arch_name)
    if entry is None:
        raise ValueError(
            f"Unsupported architecture: {arch_name!r}. "
            f"Registered: {sorted(REGISTRY.keys())}"
        )
    if isinstance(entry, str):
        import importlib
        entry = importlib.import_module(entry)
        REGISTRY[arch_name] = entry  # cache the loaded module
    return entry


def resolve(hf_config) -> str:
    """Resolve an HF config object to the pie internal arch name.

    Accepts anything with ``model_type`` and ``architectures`` accessors —
    a transformers ``PretrainedConfig``, vllm's wrapped ``hf_config`` /
    ``hf_text_config``, sglang's ``hf_config``, or a plain dict.

    Resolution order:
        1. ``model_type`` lookup in :data:`HF_TO_PIE_ARCH`.
        2. (Future) Disambiguate ``LlamaForCausalLM`` between llama2 /
           llama3 / llama3.1+ via ``rope_scaling.rope_type``.

    Raises :class:`UnknownArchitectureError` rather than silently falling
    through — that silent fallback is what caused #328.
    """
    model_type = _attr_or_key(hf_config, "model_type", "")
    if model_type in HF_TO_PIE_ARCH:
        return HF_TO_PIE_ARCH[model_type]

    architectures = _attr_or_key(hf_config, "architectures", []) or []
    arch0 = architectures[0] if architectures else "<missing>"
    raise UnknownArchitectureError(
        f"Cannot resolve HF model to a pie arch name. "
        f"model_type={model_type!r}, architectures[0]={arch0!r}. "
        f"Add an entry via register(..., hf_model_types=...) below and ensure "
        f"runtime/src/model/instruct.rs has a matching arm."
    )


def _attr_or_key(obj, key, default):
    """Read attribute or dict key, falling back to default."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# =============================================================================
# Register All Architectures (lazy — per-arch modules load on demand)
# =============================================================================
#
# `register()` takes a dotted module path string instead of an imported
# module object so that this file stays cheap to load. The vllm / sgl
# loaders, the CLI compatibility check, and any other caller that just
# needs HF_TO_PIE_ARCH or resolve() do NOT trigger the per-arch modules'
# `import pie_kernels as ops` chain (which JIT-compiles CUDA extensions
# at import time and is irrelevant on the vllm / sgl path).
#
# The native pie_driver path's `get_module(name)` triggers
# `importlib.import_module` on first dispatch — same modules load,
# same cost, just deferred until needed.

register("llama3",   "pie_driver.model.llama3",   aliases=("l4ma",), hf_model_types=("llama",))
register("qwen2",    "pie_driver.model.qwen2",                       hf_model_types=("qwen2",))
register("qwen3",    "pie_driver.model.qwen3",                       hf_model_types=("qwen3",))
register("qwen3_5",  "pie_driver.model.qwen3_5",                     hf_model_types=("qwen3_5",))
register("phi3",     "pie_driver.model.phi3",                        hf_model_types=("phi3",))
register("mixtral",  "pie_driver.model.mixtral",                     hf_model_types=("mixtral",))
register("gemma2",   "pie_driver.model.gemma2",                      hf_model_types=("gemma2",))
register("gemma3",   "pie_driver.model.gemma3",                      hf_model_types=("gemma3_text",))
register("gemma4",   "pie_driver.model.gemma4",                      hf_model_types=("gemma4_text", "gemma4"))
register("mistral3", "pie_driver.model.mistral3",                    hf_model_types=("mistral3",))
register("olmo3",    "pie_driver.model.olmo3",                       hf_model_types=("olmo3",))
register("gptoss",   "pie_driver.model.gpt_oss",                     hf_model_types=("gptoss", "gpt_oss"))
register("dummy",    "pie_driver.model.dummy")
