"""Tests for append_paged_kv_cache (Metal kernel)."""

import pytest
import torch

from .conftest import requires_mps, make_paged_kv_cache, BenchmarkTimer


@requires_mps
class TestAppendPagedKvCache:
    """Accuracy and benchmark tests for Metal KV cache append kernel."""

    def test_single_token_append(self, mps_device):
        """Appending a single token should place K/V at the correct page position."""
        from flashinfer_metal import append_paged_kv_cache

        page_size, num_kv_heads, head_dim = 16, 4, 64
        num_pages = 4

        kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                               dtype=torch.float32, device=mps_device)

        k = torch.ones(1, num_kv_heads, head_dim, device=mps_device)
        v = torch.ones(1, num_kv_heads, head_dim, device=mps_device) * 2.0

        batch_indices = torch.tensor([0], dtype=torch.int32, device=mps_device)
        positions = torch.tensor([0], dtype=torch.int32, device=mps_device)
        kv_indices = torch.tensor([0], dtype=torch.int32, device=mps_device)
        kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=mps_device)
        kv_last_page_len = torch.tensor([1], dtype=torch.int32, device=mps_device)

        append_paged_kv_cache(
            append_key=k, append_value=v,
            batch_indices=batch_indices, positions=positions,
            paged_kv_cache=kv_cache,
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
        )

        # Key should be at page 0, slot 0
        k_stored = kv_cache[0, 0, 0]  # [num_kv_heads, head_dim]
        assert torch.allclose(k_stored.cpu(), torch.ones(num_kv_heads, head_dim))

        # Value should be at page 0, slot 0
        v_stored = kv_cache[0, 1, 0]
        assert torch.allclose(v_stored.cpu(), torch.ones(num_kv_heads, head_dim) * 2.0)

    def test_multi_token_append(self, mps_device):
        """Appending multiple tokens should fill consecutive slots."""
        from flashinfer_metal import append_paged_kv_cache

        page_size, num_kv_heads, head_dim = 16, 2, 32
        num_pages = 4
        num_tokens = 4

        kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                               dtype=torch.float32, device=mps_device)

        k = torch.arange(num_tokens, dtype=torch.float32, device=mps_device).view(
            num_tokens, 1, 1).expand(num_tokens, num_kv_heads, head_dim)
        v = k * 10.0

        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=mps_device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)
        kv_indices = torch.tensor([0], dtype=torch.int32, device=mps_device)
        kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=mps_device)
        kv_last_page_len = torch.tensor([num_tokens], dtype=torch.int32, device=mps_device)

        append_paged_kv_cache(
            append_key=k.contiguous(), append_value=v.contiguous(),
            batch_indices=batch_indices, positions=positions,
            paged_kv_cache=kv_cache,
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
        )

        # Verify each token was placed correctly
        for i in range(num_tokens):
            k_stored = kv_cache[0, 0, i, 0, 0].item()
            assert abs(k_stored - float(i)) < 1e-5, f"Token {i}: expected {i}, got {k_stored}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtypes(self, mps_device, dtype):
        """KV cache append should work for all supported dtypes."""
        from flashinfer_metal import append_paged_kv_cache

        page_size, num_kv_heads, head_dim = 16, 2, 64

        kv_cache = torch.zeros(2, 2, page_size, num_kv_heads, head_dim,
                               dtype=dtype, device=mps_device)
        k = torch.randn(1, num_kv_heads, head_dim, dtype=dtype, device=mps_device)
        v = torch.randn(1, num_kv_heads, head_dim, dtype=dtype, device=mps_device)

        append_paged_kv_cache(
            append_key=k, append_value=v,
            batch_indices=torch.tensor([0], dtype=torch.int32, device=mps_device),
            positions=torch.tensor([0], dtype=torch.int32, device=mps_device),
            paged_kv_cache=kv_cache,
            kv_indices=torch.tensor([0], dtype=torch.int32, device=mps_device),
            kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device=mps_device),
            kv_last_page_len=torch.tensor([1], dtype=torch.int32, device=mps_device),
        )
        assert kv_cache.dtype == dtype

    def test_benchmark(self, mps_device):
        """Benchmark KV cache append."""
        from flashinfer_metal import append_paged_kv_cache

        page_size, num_kv_heads, head_dim = 16, 8, 128
        num_pages, num_tokens = 64, 32

        kv_cache = make_paged_kv_cache(num_pages, page_size, num_kv_heads, head_dim,
                                        device=mps_device)
        k = torch.randn(num_tokens, num_kv_heads, head_dim, device=mps_device)
        v = torch.randn(num_tokens, num_kv_heads, head_dim, device=mps_device)
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=mps_device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=mps_device)
        kv_indices = torch.arange(num_pages // 4, dtype=torch.int32, device=mps_device)
        kv_indptr = torch.tensor([0, num_pages // 4], dtype=torch.int32, device=mps_device)
        kv_last_page_len = torch.tensor([num_tokens % page_size or page_size],
                                        dtype=torch.int32, device=mps_device)

        timer = BenchmarkTimer("append_kv", mps_device)

        def run():
            append_paged_kv_cache(
                append_key=k, append_value=v,
                batch_indices=batch_indices, positions=positions,
                paged_kv_cache=kv_cache,
                kv_indices=kv_indices, kv_indptr=kv_indptr,
                kv_last_page_len=kv_last_page_len,
            )

        _, ms = timer.run(run)
        print(f"\n  append_kv [{num_tokens} tokens, {num_kv_heads} heads, {head_dim} dim]: {ms:.3f} ms")
