"""Tests for BatchPrefillWithPagedKVCacheWrapper (Metal attention kernel)."""

import pytest
import torch

from .conftest import requires_mps, make_paged_kv_cache, reference_attention, BenchmarkTimer


@requires_mps
class TestBatchPrefillWithPagedKVCache:
    """Accuracy and benchmark tests for Metal prefill attention."""

    def _setup_single_seq(self, mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                          page_size=16, dtype=torch.float16):
        """Set up a single-sequence prefill scenario with known K/V."""
        num_pages = (seq_len + page_size - 1) // page_size

        # Create query, key, value
        q = torch.randn(seq_len, num_q_heads, head_dim, dtype=dtype, device=mps_device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=mps_device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=mps_device)

        # Fill paged KV cache with known K/V values
        kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                               dtype=dtype, device=mps_device)
        for i in range(seq_len):
            page_idx = i // page_size
            slot_idx = i % page_size
            kv_cache[page_idx, 0, slot_idx] = k[i]  # key
            kv_cache[page_idx, 1, slot_idx] = v[i]  # value

        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=mps_device)
        kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=mps_device)
        kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=mps_device)
        last_page_len = seq_len - (num_pages - 1) * page_size
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=mps_device)

        return q, k, v, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len

    def test_accuracy_vs_reference(self, mps_device):
        """Metal prefill attention should match dense PyTorch attention."""
        from flashinfer_metal import BatchPrefillWithPagedKVCacheWrapper

        seq_len, num_q_heads, num_kv_heads, head_dim = 8, 4, 4, 128
        page_size = 16

        q, k, v, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = \
            self._setup_single_seq(mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                                   page_size=page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)
        wrapper.plan(
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
            num_q_heads, num_kv_heads, head_dim, page_size,
        )
        result = wrapper.run(q, kv_cache)

        ref = reference_attention(q, k, v, causal=True)

        torch.testing.assert_close(
            result.cpu().float(), ref.float(),
            atol=1e-2, rtol=1e-2,
        )

    def test_gqa(self, mps_device):
        """Grouped Query Attention: fewer KV heads than query heads."""
        from flashinfer_metal import BatchPrefillWithPagedKVCacheWrapper

        seq_len, num_q_heads, num_kv_heads, head_dim = 8, 8, 2, 128
        page_size = 16

        q, k, v, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = \
            self._setup_single_seq(mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                                   page_size=page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)
        wrapper.plan(
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
            num_q_heads, num_kv_heads, head_dim, page_size,
        )
        result = wrapper.run(q, kv_cache)

        ref = reference_attention(q, k, v, causal=True)

        torch.testing.assert_close(
            result.cpu().float(), ref.float(),
            atol=1e-2, rtol=1e-2,
        )

    def test_run_without_plan_raises(self, mps_device):
        """Calling run() without plan() should raise RuntimeError."""
        from flashinfer_metal import BatchPrefillWithPagedKVCacheWrapper

        workspace = torch.empty(1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)

        q = torch.randn(1, 4, 128, device=mps_device)
        kv = torch.randn(1, 2, 16, 4, 128, device=mps_device)

        with pytest.raises(RuntimeError, match="plan"):
            wrapper.run(q, kv)

    def test_benchmark(self, mps_device):
        """Benchmark Metal prefill attention."""
        from flashinfer_metal import BatchPrefillWithPagedKVCacheWrapper

        seq_len, num_q_heads, num_kv_heads, head_dim = 128, 32, 8, 128
        page_size = 16

        q, k, v, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = \
            self._setup_single_seq(mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                                   page_size=page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)
        wrapper.plan(
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
            num_q_heads, num_kv_heads, head_dim, page_size,
        )

        timer = BenchmarkTimer("batch_prefill", mps_device)

        def run_metal():
            return wrapper.run(q, kv_cache)

        _, metal_ms = timer.run(run_metal)

        # Reference: dense attention (runs on CPU internally)
        def run_ref():
            return reference_attention(q, k, v, causal=True)

        _, ref_ms = timer.run(run_ref)

        print(f"\n  batch_prefill [{seq_len} tokens, {num_q_heads}Q/{num_kv_heads}KV heads, {head_dim} dim]:")
        print(f"    Metal kernel: {metal_ms:.3f} ms")
        print(f"    PyTorch ref:  {ref_ms:.3f} ms")
        print(f"    Speedup:      {ref_ms / metal_ms:.2f}x")
