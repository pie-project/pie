"""Shared fixtures for flashinfer_metal tests."""

import time

import pytest
import torch

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available",
)


@pytest.fixture
def device():
    """Return MPS device if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def mps_device():
    """Return MPS device, skip test if unavailable."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend not available")
    return torch.device("mps")


class BenchmarkTimer:
    """Simple benchmark timer for comparing implementations."""

    def __init__(self, name: str, device: torch.device, warmup: int = 5, repeats: int = 50):
        self.name = name
        self.device = device
        self.warmup = warmup
        self.repeats = repeats

    def run(self, fn, *args, **kwargs):
        """Run fn with warmup + timed repeats. Returns (result, median_ms)."""
        # Warmup
        for _ in range(self.warmup):
            result = fn(*args, **kwargs)
        if self.device.type == "mps":
            torch.mps.synchronize()

        times = []
        for _ in range(self.repeats):
            if self.device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            if self.device.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        median = times[len(times) // 2]
        return result, median


def make_paged_kv_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("mps"),
) -> torch.Tensor:
    """Create a paged KV cache tensor: [num_pages, 2, page_size, num_kv_heads, head_dim]."""
    return torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )


def reference_rope_neox(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Pure-PyTorch reference RoPE (GPT-NeoX / non-interleaved).

    Always computes on CPU to avoid MPS float64 limitation.

    Args:
        x: [num_tokens, num_heads, head_dim]
        positions: [num_tokens] int
        theta: rope base frequency
    Returns:
        rotated tensor on CPU (same shape, original dtype)
    """
    orig_dtype = x.dtype
    x_cpu = x.detach().cpu()
    pos_cpu = positions.detach().cpu()

    _, _, head_dim = x_cpu.shape
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) / half))
    angles = pos_cpu.unsqueeze(-1).double() * freqs.unsqueeze(0)  # [n, half]
    cos = angles.cos().float().unsqueeze(1)  # [n, 1, half]
    sin = angles.sin().float().unsqueeze(1)

    x1 = x_cpu[..., :half].float()
    x2 = x_cpu[..., half:].float()
    out = torch.empty_like(x_cpu, dtype=torch.float32)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out.to(orig_dtype)


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Dense single-batch attention reference.

    Always computes on CPU.

    Args:
        query: [num_q, num_heads, head_dim]
        key: [seq_len, num_kv_heads, head_dim]
        value: [seq_len, num_kv_heads, head_dim]
        causal: apply causal mask (query positions attend only to key positions <= own)
    Returns:
        [num_q, num_heads * head_dim] on CPU
    """
    orig_dtype = query.dtype
    query = query.detach().cpu().float()
    key = key.detach().cpu().float()
    value = value.detach().cpu().float()

    num_q, num_heads, head_dim = query.shape
    seq_len, num_kv_heads, _ = key.shape
    gqa_ratio = num_heads // num_kv_heads

    # Expand KV heads for GQA
    if gqa_ratio > 1:
        key = key.repeat_interleave(gqa_ratio, dim=1)
        value = value.repeat_interleave(gqa_ratio, dim=1)

    # [num_heads, num_q, seq_len]
    scale = head_dim ** -0.5
    scores = torch.einsum("qhd,khd->hqk", query, key) * scale

    if causal:
        # For prefill: query and key positions are both [0, 1, ..., seq_len-1]
        q_pos = torch.arange(num_q).unsqueeze(1)  # [num_q, 1]
        k_pos = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
        mask = k_pos > q_pos  # True = masked
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, value)
    return out.to(orig_dtype).reshape(num_q, num_heads * head_dim)
