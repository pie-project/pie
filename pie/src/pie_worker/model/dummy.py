"""
Dummy model implementation for PIE backend.

This module provides a DummyForwardPass class that mimics the interface of
real model forward passes (like llama3.ForwardPass) but performs no actual
computation. Used for testing and benchmarking without GPU weight loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..config import RuntimeConfig


@dataclass
class DummyModelConfig:
    """
    Minimal model configuration for dummy mode.

    Provides sensible defaults that match Llama-3.2-1B structure,
    but can be customized as needed.
    """

    vocab_size: int = 128256
    num_layers: int = 16
    num_q_heads: int = 32
    num_kv_heads: int = 8
    dim_head: int = 64
    dim_hidden: int = 2048
    dim_ffn: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, spec: dict) -> "DummyModelConfig":
        """Create config from architecture dict (for compatibility)."""
        return cls(
            vocab_size=spec.get("vocab_size", 128256),
            num_layers=spec.get("num_layers", 16),
            num_q_heads=spec.get("num_q_heads", 32),
            num_kv_heads=spec.get("num_kv_heads", 8),
            dim_head=spec.get("dim_head", 64),
            dim_hidden=spec.get("dim_hidden", 2048),
            dim_ffn=spec.get("dim_ffn", 8192),
        )

    def eval_max_num_kv_pages(self, runtime_config: RuntimeConfig) -> int:
        """
        Return a large number of KV pages for dummy mode.

        Since we don't actually allocate real KV cache memory,
        we can report a high number.
        """
        # Return a reasonable default that won't cause issues
        return 10000


class DummyForwardPass:
    """
    Dummy forward pass that returns random tokens.

    Implements the same interface as llama3.ForwardPass, qwen2.ForwardPass, etc.
    but performs no actual GPU computation. Useful for:
    - Testing the scheduling and batching logic
    - Benchmarking request throughput without GPU overhead
    - Development without requiring a GPU
    """

    def __init__(
        self,
        model_config: DummyModelConfig,
        runtime_config: RuntimeConfig,
        weights=None,  # Ignored - no weights needed
        compute_process_group=None,  # Ignored
    ):
        """
        Initialize the dummy forward pass.

        Args:
            model_config: Dummy model configuration
            runtime_config: Runtime configuration
            weights: Ignored (no weights needed)
            compute_process_group: Ignored
        """
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.vocab_size = model_config.vocab_size
        self.device = runtime_config.device
        self.dtype = runtime_config.activation_dtype
        self.dim_hidden = model_config.dim_hidden

        # Dummy weights dict for compatibility with code that accesses weights.get()
        self.weights = _DummyWeightStore(model_config.vocab_size, model_config.dim_hidden)

    def embed_inputs(self, batch_metadata: dict[str, Any]) -> torch.Tensor:
        """
        Return random embeddings for the input tokens.

        Args:
            batch_metadata: Metadata dictionary containing token_ids

        Returns:
            Random tensor of shape [num_tokens, dim_hidden]
        """
        token_ids = batch_metadata["token_ids"]
        num_tokens = len(token_ids)
        return torch.randn(
            num_tokens, self.dim_hidden, device=self.device, dtype=self.dtype
        )

    def transform(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        adapter_subpass=None,
        total_pages_cpu: int = 0,
    ) -> torch.Tensor:
        """
        No-op transform that passes through the input embeddings.

        In a real model, this would run all transformer layers.
        In dummy mode, we just return the input unchanged.

        Args:
            input_embeds: Input embeddings
            **kwargs: Ignored

        Returns:
            The input embeddings unchanged
        """
        return input_embeds

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return random tokens instead of actual sampling.

        Args:
            hidden_states: Output hidden states (ignored)
            sampling_metadata: Metadata for sampling

        Returns:
            Dictionary with random tokens and empty distributions
        """
        indices_for_logits = sampling_metadata.get("indices_for_logits", [])
        num_samples = len(indices_for_logits)

        if num_samples == 0:
            return {"tokens": [], "dists": [], "nan_indices": []}

        # Generate random tokens within vocabulary range
        random_tokens = torch.randint(
            0, self.vocab_size, (num_samples,), device=self.device
        ).tolist()

        return {
            "tokens": random_tokens,
            "dists": [None] * num_samples,
            "nan_indices": [],
        }

    def warmup_cuda_graphs(self, kv_cache_at_layer: list[torch.Tensor]) -> None:
        """
        No-op warmup for CUDA graphs.

        Dummy mode doesn't use CUDA graphs.
        """
        pass


class _DummyWeightStore:
    """
    Minimal weight store that returns dummy tensors for compatibility.

    Some code paths (like get_tokenizer) access weights.get("embed_token").shape,
    so we need to provide a dummy implementation.
    """

    def __init__(self, vocab_size: int, dim_hidden: int):
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden

    def get(self, name: str) -> "_DummyTensor":
        """Return a dummy tensor with the expected shape."""
        if name == "embed_token":
            return _DummyTensor((self.vocab_size, self.dim_hidden))
        # Default shape for other weights
        return _DummyTensor((self.dim_hidden, self.dim_hidden))


class _DummyTensor:
    """Minimal tensor-like object that provides shape attribute."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def create_kv_cache(
    model_config: DummyModelConfig, runtime_config: RuntimeConfig
) -> list[torch.Tensor]:
    """
    Create minimal KV cache tensors for dummy mode.

    Returns small placeholder tensors that satisfy the interface requirements
    without consuming significant GPU memory.

    Args:
        model_config: Dummy model configuration
        runtime_config: Runtime configuration

    Returns:
        List of KV cache tensors (one per layer)
    """
    device = runtime_config.device
    dtype = runtime_config.activation_dtype
    num_layers = model_config.num_layers

    # Create minimal KV cache - just enough to satisfy the interface
    # Real KV cache would be much larger based on max_num_kv_pages
    kv_cache_at_layer = []
    for _ in range(num_layers):
        # Minimal 1-page cache per layer
        # Shape: [num_pages, 2, num_kv_heads, page_size, head_dim]
        cache = torch.zeros(
            1,  # 1 page
            2,  # K and V
            model_config.num_kv_heads,
            runtime_config.kv_page_size,
            model_config.dim_head,
            device=device,
            dtype=dtype,
        )
        kv_cache_at_layer.append(cache)

    return kv_cache_at_layer


def create_adapter_cache(
    model_config: DummyModelConfig, runtime_config: RuntimeConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Create empty adapter cache for dummy mode.

    Args:
        model_config: Dummy model configuration
        runtime_config: Runtime configuration

    Returns:
        Empty list (no adapters in dummy mode)
    """
    return []
