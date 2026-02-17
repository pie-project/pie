"""Tests for apply_rope_with_cos_sin_cache_inplace (pure PyTorch)."""

import pytest
import torch

from .conftest import BenchmarkTimer


def build_cos_sin_cache(max_pos: int, head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Build a cos/sin cache: [max_pos, head_dim] with first half cos, second half sin."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) / half))
    positions = torch.arange(max_pos, dtype=torch.float64)
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # [max_pos, half]
    cache = torch.cat([angles.cos(), angles.sin()], dim=-1).float()  # [max_pos, head_dim]
    return cache


def reference_rope_neox_from_cache(
    x: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor, head_size: int,
) -> torch.Tensor:
    """Manually apply NeoX-style RoPE using a cos/sin cache (reference)."""
    half = head_size // 2
    cos = cache[positions.long(), :half].unsqueeze(1).to(x.dtype)  # [n, 1, half]
    sin = cache[positions.long(), half:].unsqueeze(1).to(x.dtype)

    if x.ndim == 2:
        x = x.view(x.shape[0], -1, head_size)

    out = x.clone()
    x1 = x[..., :half]
    x2 = x[..., half:]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out


class TestApplyRopeWithCosSinCache:
    """Accuracy tests for apply_rope_with_cos_sin_cache_inplace."""

    def test_accuracy_neox_3d(self, device):
        """NeoX-style RoPE with 3D input matches reference."""
        from flashinfer_metal import apply_rope_with_cos_sin_cache_inplace

        num_tokens, num_heads, head_dim = 16, 8, 128
        max_pos = 1024

        cache = build_cos_sin_cache(max_pos, head_dim).to(device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        q = torch.randn(num_tokens, num_heads, head_dim, device=device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=device)

        q_ref = reference_rope_neox_from_cache(q, positions, cache, head_dim)
        k_ref = reference_rope_neox_from_cache(k, positions, cache, head_dim)

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions, query=q, key=k,
            head_size=head_dim, cos_sin_cache=cache, is_neox=True,
        )

        torch.testing.assert_close(q, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k, k_ref, atol=1e-5, rtol=1e-5)

    def test_accuracy_neox_2d(self, device):
        """NeoX-style RoPE with 2D (flattened) input matches reference."""
        from flashinfer_metal import apply_rope_with_cos_sin_cache_inplace

        num_tokens, num_heads, head_dim = 16, 4, 64
        max_pos = 256

        cache = build_cos_sin_cache(max_pos, head_dim).to(device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        q_3d = torch.randn(num_tokens, num_heads, head_dim, device=device)
        q_2d = q_3d.reshape(num_tokens, num_heads * head_dim).clone()
        q_ref = reference_rope_neox_from_cache(q_3d.clone(), positions, cache, head_dim)

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions, query=q_2d,
            key=torch.randn(num_tokens, num_heads * head_dim, device=device),
            head_size=head_dim, cos_sin_cache=cache, is_neox=True,
        )

        torch.testing.assert_close(
            q_2d.view(num_tokens, num_heads, head_dim),
            q_ref, atol=1e-5, rtol=1e-5,
        )

    def test_interleaved_style(self, device):
        """Interleaved RoPE (is_neox=False) should rotate even/odd pairs."""
        from flashinfer_metal import apply_rope_with_cos_sin_cache_inplace

        num_tokens, num_heads, head_dim = 4, 2, 8
        max_pos = 32

        cache = build_cos_sin_cache(max_pos, head_dim).to(device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        q = torch.randn(num_tokens, num_heads, head_dim, device=device)
        q_orig = q.clone()

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions, query=q,
            key=torch.randn(num_tokens, num_heads, head_dim, device=device),
            head_size=head_dim, cos_sin_cache=cache, is_neox=False,
        )

        # Should have been modified
        assert not torch.allclose(q, q_orig)

    def test_inplace_modification(self, device):
        """Verify that query and key are modified in-place."""
        from flashinfer_metal import apply_rope_with_cos_sin_cache_inplace

        head_dim = 64
        cache = build_cos_sin_cache(32, head_dim).to(device)
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)

        q = torch.randn(4, 2, head_dim, device=device)
        k = torch.randn(4, 2, head_dim, device=device)
        q_data_ptr = q.data_ptr()
        k_data_ptr = k.data_ptr()

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions, query=q, key=k,
            head_size=head_dim, cos_sin_cache=cache,
        )

        assert q.data_ptr() == q_data_ptr
        assert k.data_ptr() == k_data_ptr

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtypes(self, device, dtype):
        """Should work for all standard dtypes."""
        from flashinfer_metal import apply_rope_with_cos_sin_cache_inplace

        head_dim = 64
        cache = build_cos_sin_cache(32, head_dim).to(device)
        positions = torch.arange(4, dtype=torch.int32, device=device)

        q = torch.randn(4, 2, head_dim, dtype=dtype, device=device)
        k = torch.randn(4, 2, head_dim, dtype=dtype, device=device)

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions, query=q, key=k,
            head_size=head_dim, cos_sin_cache=cache.to(dtype),
        )
        assert q.dtype == dtype

    @pytest.mark.parametrize("num_tokens", [1, 32, 256])
    def test_various_seq_lengths(self, device, num_tokens):
        """Should handle different sequence lengths."""
        from flashinfer_metal import apply_rope_with_cos_sin_cache_inplace

        num_heads, head_dim = 8, 128
        cache = build_cos_sin_cache(512, head_dim).to(device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        q = torch.randn(num_tokens, num_heads, head_dim, device=device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=device)

        q_ref = reference_rope_neox_from_cache(q.clone(), positions, cache, head_dim)

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions, query=q, key=k,
            head_size=head_dim, cos_sin_cache=cache,
        )

        torch.testing.assert_close(q, q_ref, atol=1e-5, rtol=1e-5)

    def test_benchmark(self, device):
        """Benchmark cos/sin cache RoPE."""
        from flashinfer_metal import apply_rope_with_cos_sin_cache_inplace

        num_tokens, num_heads, head_dim = 512, 32, 128
        cache = build_cos_sin_cache(4096, head_dim).to(device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        q = torch.randn(num_tokens, num_heads, head_dim, device=device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=device)

        timer = BenchmarkTimer("rope_cos_sin_cache", device)

        def run():
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions, query=q.clone(), key=k.clone(),
                head_size=head_dim, cos_sin_cache=cache,
            )

        _, ms = timer.run(run)
        print(f"\n  rope_cos_sin_cache [{num_tokens} tokens, {num_heads} heads, {head_dim} dim]: {ms:.3f} ms")
