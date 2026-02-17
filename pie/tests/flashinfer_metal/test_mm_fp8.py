"""Tests for mm_fp8 (pure PyTorch FP8 matmul fallback)."""

import pytest
import torch

from .conftest import BenchmarkTimer


class TestMmFp8:
    """Accuracy and benchmark tests for mm_fp8."""

    def test_accuracy_vs_bf16_matmul(self, device):
        """mm_fp8 should match bf16 matmul * alpha within FP8 quantization error."""
        from flashinfer_metal import mm_fp8

        m, k, n = 32, 128, 64
        alpha = 0.5

        # FP8 is not supported on MPS, so test with CPU for FP8 path
        # and bf16 on whatever device is available
        if hasattr(torch, "float8_e4m3fn"):
            x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cpu")
            w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cpu")
            x_fp8 = x_bf16.to(torch.float8_e4m3fn)
            w_fp8 = w_bf16.to(torch.float8_e4m3fn)
            result = mm_fp8(x_fp8, w_fp8, alpha, out_dtype=torch.bfloat16)

            expected = (x_fp8.to(torch.bfloat16) @ w_fp8.to(torch.bfloat16).T) * alpha

            assert result.dtype == torch.bfloat16
            assert result.shape == (m, n)
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
        else:
            pytest.skip("float8_e4m3fn not available in this PyTorch build")

    def test_accuracy_bf16_inputs(self, device):
        """mm_fp8 with bf16 inputs should match standard matmul * alpha."""
        from flashinfer_metal import mm_fp8

        m, k, n = 32, 128, 64
        alpha = 0.5

        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)

        result = mm_fp8(x, w, alpha, out_dtype=torch.bfloat16)
        ref = (x @ w.T) * alpha

        torch.testing.assert_close(result, ref, atol=1e-3, rtol=1e-3)

    def test_output_dtype(self, device):
        """Output dtype should match out_dtype parameter."""
        from flashinfer_metal import mm_fp8

        x = torch.randn(8, 32, dtype=torch.bfloat16, device=device)
        w = torch.randn(16, 32, dtype=torch.bfloat16, device=device)

        for out_dtype in [torch.float32, torch.bfloat16, torch.float16]:
            result = mm_fp8(x, w, alpha=1.0, out_dtype=out_dtype)
            assert result.dtype == out_dtype

    def test_output_shape(self, device):
        """Output shape should be [m, n] for [m, k] x [n, k]^T."""
        from flashinfer_metal import mm_fp8

        m, k, n = 16, 64, 32
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)

        result = mm_fp8(x, w, alpha=1.0)
        assert result.shape == (m, n)

    def test_alpha_scaling(self, device):
        """Alpha should linearly scale the output."""
        from flashinfer_metal import mm_fp8

        x = torch.randn(8, 32, dtype=torch.bfloat16, device=device)
        w = torch.randn(16, 32, dtype=torch.bfloat16, device=device)

        r1 = mm_fp8(x, w, alpha=1.0)
        r2 = mm_fp8(x, w, alpha=2.0)

        torch.testing.assert_close(r2, r1 * 2, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("m,k,n", [(1, 64, 32), (128, 256, 512), (4, 4096, 4096)])
    def test_various_shapes(self, device, m, k, n):
        """mm_fp8 should handle various matrix dimensions."""
        from flashinfer_metal import mm_fp8

        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)

        result = mm_fp8(x, w, alpha=1.0)
        assert result.shape == (m, n)

        ref = x @ w.T
        torch.testing.assert_close(result, ref, atol=1e-3, rtol=1e-3)

    def test_benchmark(self, device):
        """Benchmark mm_fp8."""
        from flashinfer_metal import mm_fp8

        m, k, n = 256, 4096, 4096
        alpha = 1.0

        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)

        timer = BenchmarkTimer("mm_fp8", device)

        def run_mm_fp8():
            return mm_fp8(x, w, alpha)

        _, fp8_ms = timer.run(run_mm_fp8)

        # Baseline: standard bf16 matmul
        def run_bf16():
            return x @ w.T

        _, bf16_ms = timer.run(run_bf16)

        print(f"\n  mm_fp8 [{m}x{k} @ {n}x{k}^T]:")
        print(f"    mm_fp8:       {fp8_ms:.3f} ms")
        print(f"    bf16 matmul:  {bf16_ms:.3f} ms")
        print(f"    Overhead:     {fp8_ms / bf16_ms:.2f}x")
