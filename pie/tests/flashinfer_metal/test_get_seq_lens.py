"""Tests for get_seq_lens utility function."""

import pytest
import torch

from .conftest import requires_mps


@requires_mps
class TestGetSeqLens:
    """Accuracy tests for get_seq_lens."""

    def test_basic(self, mps_device):
        """Sequence length = (num_pages - 1) * page_size + last_page_len."""
        from flashinfer_metal import get_seq_lens

        page_size = 16
        # Batch of 3 sequences: 2 pages, 3 pages, 1 page
        kv_page_indptr = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device=mps_device)
        kv_last_page_lens = torch.tensor([10, 8, 3], dtype=torch.int32, device=mps_device)

        result = get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)

        # seq0: (2-1)*16 + 10 = 26
        # seq1: (3-1)*16 + 8  = 40
        # seq2: (1-1)*16 + 3  = 3
        expected = torch.tensor([26, 40, 3], dtype=torch.int32, device=mps_device)
        torch.testing.assert_close(result, expected)

    def test_empty_sequence(self, mps_device):
        """Sequence with 0 pages should have length 0."""
        from flashinfer_metal import get_seq_lens

        kv_page_indptr = torch.tensor([0, 0, 2], dtype=torch.int32, device=mps_device)
        kv_last_page_lens = torch.tensor([0, 8], dtype=torch.int32, device=mps_device)

        result = get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size=16)

        expected = torch.tensor([0, 24], dtype=torch.int32, device=mps_device)
        torch.testing.assert_close(result, expected)

    def test_single_page_sequence(self, mps_device):
        """Single page sequence: length = last_page_len."""
        from flashinfer_metal import get_seq_lens

        kv_page_indptr = torch.tensor([0, 1], dtype=torch.int32, device=mps_device)
        kv_last_page_lens = torch.tensor([5], dtype=torch.int32, device=mps_device)

        result = get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size=16)

        expected = torch.tensor([5], dtype=torch.int32, device=mps_device)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("page_size", [1, 4, 8, 16])
    def test_various_page_sizes(self, mps_device, page_size):
        """Should work for different page sizes."""
        from flashinfer_metal import get_seq_lens

        kv_page_indptr = torch.tensor([0, 3], dtype=torch.int32, device=mps_device)
        kv_last_page_lens = torch.tensor([page_size], dtype=torch.int32, device=mps_device)

        result = get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)
        expected = torch.tensor([3 * page_size], dtype=torch.int32, device=mps_device)
        torch.testing.assert_close(result, expected)

    def test_large_batch(self, mps_device):
        """Should handle large batch sizes efficiently."""
        from flashinfer_metal import get_seq_lens

        batch_size = 256
        page_size = 16
        pages_per_seq = torch.randint(1, 10, (batch_size,))
        indptr = torch.cat([torch.tensor([0]), pages_per_seq.cumsum(0)])
        last_lens = torch.randint(1, page_size + 1, (batch_size,))

        kv_page_indptr = indptr.to(torch.int32).to(mps_device)
        kv_last_page_lens = last_lens.to(torch.int32).to(mps_device)

        result = get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)

        # Verify against manual computation
        expected = ((pages_per_seq - 1) * page_size + last_lens).to(torch.int32).to(mps_device)
        torch.testing.assert_close(result, expected)
