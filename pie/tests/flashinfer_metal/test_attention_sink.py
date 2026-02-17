"""Tests for BatchAttentionWithAttentionSinkWrapper (Metal attention kernel)."""

import pytest
import torch

from .conftest import requires_mps, reference_attention, BenchmarkTimer


@requires_mps
class TestBatchAttentionWithAttentionSink:
    """Accuracy and benchmark tests for attention sink wrapper."""

    def _setup_single_seq(self, mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                          page_size=16, dtype=torch.float16):
        """Set up a single-sequence scenario with paged KV cache."""
        num_pages = (seq_len + page_size - 1) // page_size

        q = torch.randn(seq_len, num_q_heads, head_dim, dtype=dtype, device=mps_device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=mps_device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=mps_device)

        kv_cache = torch.zeros(num_pages, 2, page_size, num_kv_heads, head_dim,
                               dtype=dtype, device=mps_device)
        for i in range(seq_len):
            page_idx = i // page_size
            slot_idx = i % page_size
            kv_cache[page_idx, 0, slot_idx] = k[i]
            kv_cache[page_idx, 1, slot_idx] = v[i]

        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=mps_device)
        kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=mps_device)
        kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=mps_device)
        last_page_len = seq_len - (num_pages - 1) * page_size
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=mps_device)

        return q, k, v, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len

    def test_accuracy_without_sinks(self, mps_device):
        """Without sinks, should match standard attention."""
        from flashinfer_metal import BatchAttentionWithAttentionSinkWrapper

        seq_len, num_q_heads, num_kv_heads, head_dim = 8, 4, 4, 128
        page_size = 16

        q, k, v, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = \
            self._setup_single_seq(mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                                   page_size=page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace,
            window_left=-1,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
        )
        wrapper.plan(
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
            num_q_heads, num_kv_heads, head_dim, page_size,
            causal=True, window_left=-1,
            q_data_type=torch.float16, kv_data_type=torch.float16,
        )

        result = wrapper.run(q, kv_cache, sinks=None, scaling=head_dim**-0.5)

        ref = reference_attention(q, k, v, causal=True)

        torch.testing.assert_close(
            result.cpu().float(), ref.float(),
            atol=1e-2, rtol=1e-2,
        )

    def test_constructor_params_stored(self, mps_device):
        """Constructor params should be stored for later use."""
        from flashinfer_metal import BatchAttentionWithAttentionSinkWrapper

        workspace = torch.empty(1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace,
            window_left=127,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            head_dim_qk=128,
            head_dim_vo=128,
        )

        assert wrapper._window_left == 127
        assert wrapper._head_dim_qk == 128
        assert wrapper._q_data_type == torch.bfloat16

    def test_run_without_plan_raises(self, mps_device):
        """Calling run() without plan() should raise RuntimeError."""
        from flashinfer_metal import BatchAttentionWithAttentionSinkWrapper

        workspace = torch.empty(1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace,
        )

        with pytest.raises(RuntimeError, match="plan"):
            wrapper.run(
                torch.randn(1, 4, 128, device=mps_device),
                torch.randn(1, 2, 16, 4, 128, device=mps_device),
            )

    def test_window_left_variants(self, mps_device):
        """Both window_left=-1 (full) and window_left>0 (sliding) should run."""
        from flashinfer_metal import BatchAttentionWithAttentionSinkWrapper

        seq_len, num_q_heads, num_kv_heads, head_dim = 8, 4, 4, 128
        page_size = 16

        q, k, v, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = \
            self._setup_single_seq(mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                                   page_size=page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)

        for window_left in [-1, 3]:
            wrapper = BatchAttentionWithAttentionSinkWrapper(
                float_workspace_buffer=workspace, window_left=window_left,
                head_dim_qk=head_dim, head_dim_vo=head_dim,
            )
            wrapper.plan(
                qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
                num_q_heads, num_kv_heads, head_dim, page_size,
                window_left=window_left,
            )
            result = wrapper.run(q, kv_cache)
            assert result.shape == (seq_len, num_q_heads * head_dim)

    def test_sinks_accepted(self, mps_device):
        """run() should accept sinks parameter without error (even if ignored)."""
        from flashinfer_metal import BatchAttentionWithAttentionSinkWrapper

        seq_len, num_q_heads, num_kv_heads, head_dim = 4, 4, 4, 128
        page_size = 16

        q, _, _, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = \
            self._setup_single_seq(mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                                   page_size=page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace,
            head_dim_qk=head_dim, head_dim_vo=head_dim,
        )
        wrapper.plan(
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
            num_q_heads, num_kv_heads, head_dim, page_size,
        )

        sinks = torch.randn(2, num_kv_heads, head_dim, dtype=torch.float16, device=mps_device)
        result = wrapper.run(q, kv_cache, sinks=sinks, scaling=head_dim**-0.5)
        assert result.shape == (seq_len, num_q_heads * head_dim)

    def test_benchmark(self, mps_device):
        """Benchmark attention sink wrapper."""
        from flashinfer_metal import BatchAttentionWithAttentionSinkWrapper

        seq_len, num_q_heads, num_kv_heads, head_dim = 128, 32, 8, 128
        page_size = 16

        q, _, _, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len = \
            self._setup_single_seq(mps_device, seq_len, num_q_heads, num_kv_heads, head_dim,
                                   page_size=page_size)

        workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=mps_device)
        wrapper = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace,
            head_dim_qk=head_dim, head_dim_vo=head_dim,
        )
        wrapper.plan(
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
            num_q_heads, num_kv_heads, head_dim, page_size,
        )

        timer = BenchmarkTimer("attention_sink", mps_device)

        def run():
            return wrapper.run(q, kv_cache)

        _, ms = timer.run(run)
        print(f"\n  attention_sink [{seq_len} tokens, {num_q_heads}Q/{num_kv_heads}KV heads, {head_dim} dim]: {ms:.3f} ms")
