"""Tests for apply_llama31_rope_pos_ids_inplace (Metal kernel with scaling)."""

import pytest
import torch

from .conftest import requires_mps, BenchmarkTimer


def reference_rope_scaled(
    x: torch.Tensor, positions: torch.Tensor, theta: float, factor: float,
    low_freq_factor: float = 1.0, high_freq_factor: float = 4.0, old_context_len: int = 8192,
) -> torch.Tensor:
    """Reference LLaMA 3.1 RoPE with wavelength-based selective scaling.

    Matches the Metal kernel: selectively scales inv_freq based on wavelength bands.
    Always computes on CPU.
    """
    import math
    orig_dtype = x.dtype
    x_cpu = x.detach().cpu()
    pos_cpu = positions.detach().cpu()

    _, _, head_dim = x_cpu.shape
    half = head_dim // 2

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    inv_freqs = []
    for i in range(half):
        exponent = (2.0 * i) / head_dim
        inv_freq_base = 1.0 / (theta ** exponent)
        wavelen = 2.0 * math.pi / inv_freq_base

        if wavelen > low_freq_wavelen:
            inv_freq = inv_freq_base / factor
        elif wavelen < high_freq_wavelen:
            inv_freq = inv_freq_base
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            scaled = inv_freq_base / factor
            inv_freq = (1.0 - smooth) * scaled + smooth * inv_freq_base
        inv_freqs.append(inv_freq)

    inv_freqs = torch.tensor(inv_freqs, dtype=torch.float64)
    angles = pos_cpu.unsqueeze(-1).double() * inv_freqs.unsqueeze(0)
    cos = angles.cos().float().unsqueeze(1)
    sin = angles.sin().float().unsqueeze(1)

    x1 = x_cpu[..., :half].float()
    x2 = x_cpu[..., half:].float()
    out = torch.empty_like(x_cpu, dtype=torch.float32)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out.to(orig_dtype)


@requires_mps
class TestApplyLlama31RopePosIds:
    """Accuracy and benchmark tests for Metal LLaMA 3.1 RoPE kernel."""

    def test_accuracy_vs_reference(self, mps_device):
        """Metal LLaMA 3.1 RoPE should match scaled reference."""
        from flashinfer_metal import apply_llama31_rope_pos_ids_inplace

        num_tokens, num_heads, head_dim = 32, 8, 128
        theta = 500000.0
        scale = 32.0

        q = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        pos_ids = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)

        q_ref = reference_rope_scaled(q.cpu(), pos_ids.cpu(), theta, scale)
        k_ref = reference_rope_scaled(k.cpu(), pos_ids.cpu(), theta, scale)

        apply_llama31_rope_pos_ids_inplace(
            q, k, pos_ids, rope_theta=theta, rope_scale=scale,
        )

        torch.testing.assert_close(q.cpu(), q_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k.cpu(), k_ref, atol=1e-2, rtol=1e-2)

    def test_rejects_unsupported_params(self, mps_device):
        """Should raise on unsupported rotary_dim / freq factor params."""
        from flashinfer_metal import apply_llama31_rope_pos_ids_inplace

        q = torch.randn(4, 4, 64, device=mps_device)
        k = torch.randn(4, 4, 64, device=mps_device)
        pos = torch.arange(4, dtype=torch.int32, device=mps_device)

        with pytest.raises(ValueError, match="rotary_dim"):
            apply_llama31_rope_pos_ids_inplace(q, k, pos, rotary_dim=32)
        with pytest.raises(ValueError, match="low_freq_factor"):
            apply_llama31_rope_pos_ids_inplace(q, k, pos, low_freq_factor=2.0)
        with pytest.raises(ValueError, match="high_freq_factor"):
            apply_llama31_rope_pos_ids_inplace(q, k, pos, high_freq_factor=8.0)
        with pytest.raises(ValueError, match="old_context_len"):
            apply_llama31_rope_pos_ids_inplace(q, k, pos, old_context_len=4096)

    def test_benchmark(self, mps_device):
        """Benchmark Metal LLaMA 3.1 RoPE."""
        from flashinfer_metal import apply_llama31_rope_pos_ids_inplace

        num_tokens, num_heads, head_dim = 512, 32, 128

        q = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        pos_ids = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)

        timer = BenchmarkTimer("rope_llama31", mps_device)

        def run_metal():
            apply_llama31_rope_pos_ids_inplace(q.clone(), k.clone(), pos_ids)

        _, metal_ms = timer.run(run_metal)
        print(f"\n  rope_llama31 [{num_tokens} tokens]: {metal_ms:.3f} ms")
