"""Tests for apply_rope_pos_ids_inplace (Metal kernel)."""

import pytest
import torch

from .conftest import requires_mps, reference_rope_neox, BenchmarkTimer


@requires_mps
class TestApplyRopePosIds:
    """Accuracy and benchmark tests for Metal RoPE kernel."""

    def test_accuracy_vs_reference(self, mps_device):
        """Metal RoPE should match pure-PyTorch reference within tolerance."""
        from flashinfer_metal import apply_rope_pos_ids_inplace

        num_tokens, num_heads, head_dim = 32, 8, 128
        theta = 10000.0

        q = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        pos_ids = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)

        q_ref = reference_rope_neox(q.cpu(), pos_ids.cpu(), theta=theta)
        k_ref = reference_rope_neox(k.cpu(), pos_ids.cpu(), theta=theta)

        apply_rope_pos_ids_inplace(q, k, pos_ids, rope_theta=theta, interleave=False)

        torch.testing.assert_close(q.cpu(), q_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k.cpu(), k_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtypes(self, mps_device, dtype):
        """Metal RoPE should work for all supported dtypes."""
        from flashinfer_metal import apply_rope_pos_ids_inplace

        num_tokens, num_heads, head_dim = 8, 4, 64

        q = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=mps_device)
        k = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=mps_device)
        pos_ids = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)

        # Should not raise
        apply_rope_pos_ids_inplace(q, k, pos_ids)
        assert q.dtype == dtype
        assert k.dtype == dtype

    @pytest.mark.parametrize("num_tokens", [1, 16, 128])
    def test_various_seq_lengths(self, mps_device, num_tokens):
        """Metal RoPE should handle different sequence lengths."""
        from flashinfer_metal import apply_rope_pos_ids_inplace

        num_heads, head_dim = 4, 128
        theta = 10000.0

        q = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        pos_ids = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)

        q_ref = reference_rope_neox(q.cpu(), pos_ids.cpu(), theta=theta)
        k_ref = reference_rope_neox(k.cpu(), pos_ids.cpu(), theta=theta)

        apply_rope_pos_ids_inplace(q, k, pos_ids, rope_theta=theta)

        torch.testing.assert_close(q.cpu(), q_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(k.cpu(), k_ref, atol=1e-2, rtol=1e-2)

    def test_benchmark(self, mps_device):
        """Benchmark Metal RoPE vs PyTorch reference."""
        from flashinfer_metal import apply_rope_pos_ids_inplace

        num_tokens, num_heads, head_dim = 512, 32, 128
        theta = 10000.0

        q = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        k = torch.randn(num_tokens, num_heads, head_dim, device=mps_device)
        pos_ids = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)

        timer = BenchmarkTimer("rope_pos_ids", mps_device)

        def run_metal():
            apply_rope_pos_ids_inplace(q.clone(), k.clone(), pos_ids, rope_theta=theta)

        _, metal_ms = timer.run(run_metal)

        # PyTorch reference (runs on CPU internally)
        def run_ref():
            reference_rope_neox(q.cpu(), pos_ids.cpu(), theta=theta)

        _, ref_ms = timer.run(run_ref)

        print(f"\n  rope_pos_ids [{num_tokens} tokens, {num_heads} heads, {head_dim} dim]:")
        print(f"    Metal kernel: {metal_ms:.3f} ms")
        print(f"    PyTorch ref:  {ref_ms:.3f} ms")
        print(f"    Speedup:      {ref_ms / metal_ms:.2f}x")
