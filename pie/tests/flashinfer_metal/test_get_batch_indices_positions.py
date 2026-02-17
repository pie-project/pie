"""Tests for get_batch_indices_positions utility function."""

import pytest
import torch

from .conftest import requires_mps


@requires_mps
class TestGetBatchIndicesPositions:
    """Accuracy tests for get_batch_indices_positions."""

    def test_basic(self, mps_device):
        """Basic case: two sequences with known token counts and positions."""
        from flashinfer_metal import get_batch_indices_positions

        # Batch of 2 sequences: 3 tokens and 2 tokens
        append_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=mps_device)
        seq_lens = torch.tensor([10, 7], dtype=torch.int32, device=mps_device)
        nnz = 5

        batch_indices, positions = get_batch_indices_positions(append_indptr, seq_lens, nnz)

        expected_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=mps_device)
        # seq0: 10 total, 3 new -> positions 7, 8, 9
        # seq1: 7 total, 2 new -> positions 5, 6
        expected_pos = torch.tensor([7, 8, 9, 5, 6], dtype=torch.int32, device=mps_device)

        torch.testing.assert_close(batch_indices, expected_batch)
        torch.testing.assert_close(positions, expected_pos)

    def test_single_sequence(self, mps_device):
        """Single sequence should produce sequential positions."""
        from flashinfer_metal import get_batch_indices_positions

        append_indptr = torch.tensor([0, 4], dtype=torch.int32, device=mps_device)
        seq_lens = torch.tensor([4], dtype=torch.int32, device=mps_device)
        nnz = 4

        batch_indices, positions = get_batch_indices_positions(append_indptr, seq_lens, nnz)

        expected_batch = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=mps_device)
        expected_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=mps_device)

        torch.testing.assert_close(batch_indices, expected_batch)
        torch.testing.assert_close(positions, expected_pos)

    def test_single_token_per_sequence(self, mps_device):
        """Decode mode: 1 new token per sequence."""
        from flashinfer_metal import get_batch_indices_positions

        batch_size = 4
        append_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=mps_device)
        seq_lens = torch.tensor([100, 50, 200, 10], dtype=torch.int32, device=mps_device)
        nnz = batch_size

        batch_indices, positions = get_batch_indices_positions(append_indptr, seq_lens, nnz)

        expected_batch = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=mps_device)
        # Each has 1 new token, so position = seq_len - 1
        expected_pos = torch.tensor([99, 49, 199, 9], dtype=torch.int32, device=mps_device)

        torch.testing.assert_close(batch_indices, expected_batch)
        torch.testing.assert_close(positions, expected_pos)

    def test_prefill_new_sequence(self, mps_device):
        """Prefill: seq_lens == token counts (all tokens are new)."""
        from flashinfer_metal import get_batch_indices_positions

        # 2 sequences: 5 tokens and 3 tokens, both new
        append_indptr = torch.tensor([0, 5, 8], dtype=torch.int32, device=mps_device)
        seq_lens = torch.tensor([5, 3], dtype=torch.int32, device=mps_device)
        nnz = 8

        batch_indices, positions = get_batch_indices_positions(append_indptr, seq_lens, nnz)

        expected_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=mps_device)
        expected_pos = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.int32, device=mps_device)

        torch.testing.assert_close(batch_indices, expected_batch)
        torch.testing.assert_close(positions, expected_pos)

    def test_large_batch(self, mps_device):
        """Should handle large batches correctly."""
        from flashinfer_metal import get_batch_indices_positions

        batch_size = 128
        tokens_per_seq = 4
        nnz = batch_size * tokens_per_seq

        counts = torch.full((batch_size,), tokens_per_seq, dtype=torch.int32)
        append_indptr = torch.cat([torch.tensor([0]), counts.cumsum(0)]).to(
            torch.int32).to(mps_device)
        seq_lens = torch.full((batch_size,), 100, dtype=torch.int32, device=mps_device)

        batch_indices, positions = get_batch_indices_positions(append_indptr, seq_lens, nnz)

        assert batch_indices.shape[0] == nnz
        assert positions.shape[0] == nnz
        # First token of batch 0 should be at position 96 (100 - 4)
        assert positions[0].item() == 96
