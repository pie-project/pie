"""Tests for BatchDecodeWithPagedKVCacheWrapper (Metal attention kernel)."""

import pytest
import torch

from .conftest import requires_mps, reference_attention, BenchmarkTimer


@requires_mps
class TestBatchDecodeWithPagedKVCache:
    """Accuracy and benchmark tests for Metal decode attention."""

    def _setup_decode(self, mps_device, batch_size, seq_len, num_q_heads, num_kv_heads,
                      head_dim, page_size=16, dtype=torch.float16):
        """Set up a decode scenario: 1 query token per batch, seq_len KV tokens."""
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = num_pages_per_seq * batch_size

        # Query: one token per batch item
        q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=mps_device)

        # Build KV cache with known values
        kv_cache = torch.zeros(total_pages, 2, page_size, num_kv_heads, head_dim,
                               dtype=dtype, device=mps_device)
        keys = []
        values = []
        for b in range(batch_size):
            k_seq = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
            v_seq = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
            keys.append(k_seq)
            values.append(v_seq)
            for i in range(seq_len):
                page_idx = b * num_pages_per_seq + i // page_size
                slot_idx = i % page_size
                kv_cache[page_idx, 0, slot_idx] = k_seq[i].to(mps_device)
                kv_cache[page_idx, 1, slot_idx] = v_seq[i].to(mps_device)

        # Paging metadata
        indptr = torch.arange(0, (batch_size + 1) * num_pages_per_seq, num_pages_per_seq,
                              dtype=torch.int32, device=mps_device)
        indices = torch.arange(total_pages, dtype=torch.int32, device=mps_device)
        last_page_len = seq_len - (num_pages_per_seq - 1) * page_size
        last_page_lens = torch.full((batch_size,), last_page_len,
                                    dtype=torch.int32, device=mps_device)

        return q, keys, values, kv_cache, indptr, indices, last_page_lens

    def test_accuracy_vs_reference(self, mps_device):
        """Metal decode attention should match dense PyTorch attention."""
        from flashinfer_metal import BatchDecodeWithPagedKVCacheWrapper

        batch_size, seq_len = 1, 32
        num_q_heads, num_kv_heads, head_dim = 4, 4, 128
        page_size = 16

        q, keys, values, kv_cache, indptr, indices, last_page_lens = \
            self._setup_decode(mps_device, batch_size, seq_len, num_q_heads,
                               num_kv_heads, head_dim, page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace)
        wrapper.plan(indptr, indices, last_page_lens, num_q_heads, num_kv_heads,
                     head_dim, page_size)
        result = wrapper.run(q, kv_cache)

        # Reference: single-token attention against full KV
        ref = reference_attention(
            q[:1].cpu(), keys[0].cpu(), values[0].cpu(),
        )

        torch.testing.assert_close(
            result.cpu().float(), ref.float(),
            atol=5e-2, rtol=5e-2,
        )

    def test_run_without_plan_raises(self, mps_device):
        """Calling run() without plan() should raise RuntimeError."""
        from flashinfer_metal import BatchDecodeWithPagedKVCacheWrapper

        workspace = torch.empty(1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace)

        with pytest.raises(RuntimeError, match="plan"):
            wrapper.run(
                torch.randn(1, 4, 128, device=mps_device),
                torch.randn(1, 2, 16, 4, 128, device=mps_device),
            )

    def test_benchmark(self, mps_device):
        """Benchmark Metal decode attention."""
        from flashinfer_metal import BatchDecodeWithPagedKVCacheWrapper

        batch_size, seq_len = 8, 512
        num_q_heads, num_kv_heads, head_dim = 32, 8, 128
        page_size = 16

        q, keys, values, kv_cache, indptr, indices, last_page_lens = \
            self._setup_decode(mps_device, batch_size, seq_len, num_q_heads,
                               num_kv_heads, head_dim, page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace)
        wrapper.plan(indptr, indices, last_page_lens, num_q_heads, num_kv_heads,
                     head_dim, page_size)

        timer = BenchmarkTimer("batch_decode", mps_device)

        def run():
            return wrapper.run(q, kv_cache)

        _, ms = timer.run(run)
        print(f"\n  batch_decode [{batch_size} seqs x {seq_len} KV, {num_q_heads}Q/{num_kv_heads}KV heads]: {ms:.3f} ms")
