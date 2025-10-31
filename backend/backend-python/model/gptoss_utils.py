"""GPT OSS Utility Components

This module contains utility functions, constants, and helper layers
used by the GPT OSS model architecture.
"""

from __future__ import annotations

import math
from itertools import islice

import torch
from torch import nn
import flashinfer as ops


# Reference:
# https://github.com/openai/gpt-oss/blob/9ffdd14b89b9dbc1/gpt_oss/torch/weights.py
FP4_VALUES = (
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def chunked_enumerate(iterable, chunk_size):
    """
    Enumerate an iterable in chunks, yielding both indices and items.

    This function takes an iterable and processes it in chunks of the specified size,
    yielding tuples of (indices, items) for each chunk. The indices correspond to
    the original positions in the iterable.

    Args:
        iterable: The iterable to process in chunks
        chunk_size: The maximum size of each chunk

    Yields:
        tuple[list[int], list]: A tuple containing:
            - A list of indices from the original iterable
            - A list of corresponding items from the iterable

    Example:
        >>> list(chunked_enumerate(['a', 'b', 'c', 'd', 'e'], 2))
        [([0, 1], ['a', 'b']), ([2, 3], ['c', 'd']), ([4], ['e'])]
    """
    it = iter(enumerate(iterable))
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        idxs, items = zip(*chunk)
        yield list(idxs), list(items)


class GptOssRMSNorm(nn.Module):
    """GPT OSS RMS Normalization layer, which has a scaling parameter."""

    def __init__(self, hidden_size: int, device: str, eps: float = 1e-6):
        """RMS Normalization layer."""

        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=torch.float32)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """Forward pass through the RMS Normalization layer with scaling parameter."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class GptOssRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN scaling support."""

    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
        max_position_id: int = 131072,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        self.max_position_id = max_position_id

        # Pre-compute concentration and inv_freq since they're constant
        self._concentration, self._inv_freq = self._compute_concentration_and_inv_freq()

        # Pre-compute the full cos/sin cache for all positions up to max_position_id
        self._cos_sin_cache = self._precompute_cos_sin_cache()

    def _compute_concentration_and_inv_freq(self) -> tuple[float, torch.Tensor]:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _precompute_cos_sin_cache(self) -> torch.Tensor:
        """Pre-compute cos/sin cache for all positions up to max_position_id."""
        # Create position indices
        position_ids = torch.arange(
            self.max_position_id, dtype=torch.float32, device=self.device
        )

        # Compute frequencies for all positions
        freqs = torch.einsum("i,j->ij", position_ids, self._inv_freq)

        # Compute cos and sin values with concentration
        cos_cache = freqs.cos() * self._concentration
        sin_cache = freqs.sin() * self._concentration

        # Concatenate cos and sin for FlashInfer format
        # Shape: [max_position_id, head_dim] where head_dim contains
        # [cos_0, cos_1, ..., sin_0, sin_1, ...]
        cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)

        # Ensure float32 precision for numerical accuracy
        return cos_sin_cache.to(torch.float32)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors using position IDs."""
        # Use FlashInfer's optimized RoPE function with pre-computed cache
        ops.apply_rope_with_cos_sin_cache_inplace(
            positions=position_ids.to(torch.int32),
            query=query,
            key=key,
            head_size=self.head_dim,
            cos_sin_cache=self._cos_sin_cache,
            is_neox=True,  # GPT-OSS uses Neox-style RoPE
        )

        return query, key


def calculate_activation_memory_bytes(
    max_tokens: int,
    hidden_size: int,
    intermediate_size: int,  # pylint: disable=unused-argument
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype_size: int,
    num_experts: int = 8,  # Default for GptOss
) -> int:
    """Calculate total memory needed for activation buffers.

    Note: GptOss preallocates QKV, attention output, and router logits buffers.
    Expert intermediate buffers are not preallocated due to the MoE architecture
    with custom einsum operations that don't support buffer reuse.

    Args:
        max_tokens: Maximum number of tokens in a batch
        hidden_size: Hidden layer size
        intermediate_size: MLP intermediate size (unused for GptOss)
        num_query_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads
        head_size: Size of each attention head
        dtype_size: Size of data type in bytes (e.g., 2 for float16)
        num_experts: Number of experts in MoE (default: 8)

    Returns:
        Total bytes needed for activation buffers
    """
    # QKV projection buffer: [max_tokens, (num_query_heads + 2*num_kv_heads) * head_size]
    qkv_size = (
        max_tokens * (num_query_heads + 2 * num_kv_heads) * head_size * dtype_size
    )

    # Attention output buffer: [max_tokens, hidden_size]
    attn_output_size = max_tokens * hidden_size * dtype_size

    # Router logits buffer: [max_tokens, num_experts]
    router_logits_size = max_tokens * num_experts * dtype_size

    return qkv_size + attn_output_size + router_logits_size
