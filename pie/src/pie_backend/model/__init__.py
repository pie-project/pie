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

TEMPLATES: dict[str, object] = {}
"""Architecture name → ChatTemplate instance."""

HF_TO_PIE_ARCH: dict[str, str] = {}
"""HuggingFace model_type → PIE architecture name."""


def register(
    name: str,
    module: ModuleType,
    chat_template=None,
    *,
    aliases: tuple[str, ...] = (),
    hf_model_types: tuple[str, ...] = (),
):
    """Register a model architecture.

    Args:
        name: Primary architecture name (e.g., "llama3")
        module: Module containing ModelConfig, ForwardPass, create_kv_cache, create_adapter_cache
        chat_template: Optional ChatTemplate instance
        aliases: Additional names that map to the same module
        hf_model_types: HuggingFace model_type strings that map to this architecture
    """
    REGISTRY[name] = module
    for alias in aliases:
        REGISTRY[alias] = module
    if chat_template is not None:
        TEMPLATES[name] = chat_template
        for alias in aliases:
            TEMPLATES[alias] = chat_template
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


def get_chat_template(arch_name: str) -> dict:
    """Look up chat template by architecture name.

    Returns a dict with template_type, template_content, and stop_tokens.
    Returns a 'none' template if the architecture has no registered template.
    """
    template = TEMPLATES.get(arch_name)
    if template is not None:
        return {
            "template_type": template.template_type,
            "template_content": template.template,
            "stop_tokens": template.stop_tokens,
        }
    return {
        "template_type": "none",
        "template_content": "",
        "stop_tokens": [],
    }


# =============================================================================
# Register All Architectures
# =============================================================================

import torch
from . import llama3, qwen2, qwen3, gemma2, gemma3, mistral3, olmo3
from .chat_templates import (
    ChatTemplate,
    Llama3Template,
    Qwen2_5Template,
    Qwen3Template,
    Gemma2Template,
    Gemma3Template,
    Mistral3Template,
    Olmo3Template,
)

register("llama3", llama3, Llama3Template, aliases=("l4ma",), hf_model_types=("llama",))
register("qwen2", qwen2, Qwen2_5Template, hf_model_types=("qwen2",))
register("qwen3", qwen3, Qwen3Template, hf_model_types=("qwen3",))
register("gemma2", gemma2, Gemma2Template, hf_model_types=("gemma2",))
register("gemma3", gemma3, Gemma3Template, hf_model_types=("gemma3_text",))
register("mistral3", mistral3, Mistral3Template, hf_model_types=("mistral3",))
register("olmo3", olmo3, Olmo3Template, hf_model_types=("olmo3",))

# gpt_oss requires CUDA-only features
if torch.cuda.is_available():
    from . import gpt_oss
    from .chat_templates import GPTOSSTemplate
    register("gptoss", gpt_oss, GPTOSSTemplate, hf_model_types=("gptoss", "gpt_oss"))

# Dummy mode: simple ChatML template compatible with most tokenizers
_DUMMY_CHAT_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}<|im_start|>system\n{{ m.content }}<|im_end|>\n"
    "{%- elif m.role == 'user' -%}<|im_start|>user\n{{ m.content }}<|im_end|>\n"
    "{%- elif m.role == 'assistant' -%}<|im_start|>assistant\n{{ m.content }}<|im_end|>\n"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
)
from . import dummy as _dummy_mod
register(
    "dummy", _dummy_mod,
    ChatTemplate(
        template_type="minijinja",
        template=_DUMMY_CHAT_TEMPLATE,
        stop_tokens=["<|im_end|>", "<|endoftext|>"]
    ),
)
