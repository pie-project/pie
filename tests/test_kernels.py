import unittest

import numpy as np
import torch

from gpu_lr1.kernels import (
    make_ell_argmax_advance_plan,
    triton_bitset_argmax,
    triton_bitset_mask_logits,
    triton_byte_dfa_advance,
    triton_csr_argmax,
    triton_csr_argmax_advance,
    triton_csr_argmax_advance_packed,
    triton_dense_advance,
    triton_dense_argmax,
    triton_dense_mask_logits,
)
from gpu_lr1.tables import compile_named_schemas, compile_packed_tables
from gpu_lr1.vocab import Vocabulary
from gpu_lr1.workloads import benchmark_schemas


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schemas = benchmark_schemas(2)
        vocabulary = Vocabulary.synthetic(
            512,
            [item.schema for item in schemas],
        )
        cls.tables = compile_packed_tables(
            compile_named_schemas(schemas),
            vocabulary,
            state_chunk_size=32,
        )
        cls.tensors = cls.tables.torch_tensors()
        nonempty = np.flatnonzero(cls.tables.row_nnz > 0)
        cls.rows = torch.tensor(
            np.resize(nonempty, 16),
            dtype=torch.int32,
            device="cuda",
        )
        torch.manual_seed(11)
        cls.logits = torch.randn(
            (cls.rows.numel(), vocabulary.size),
            dtype=torch.float32,
            device="cuda",
        )

    def test_mask_kernels(self) -> None:
        reference = self.logits.masked_fill(
            self.tensors.dense_mask[self.rows.long()].logical_not(),
            -float("inf"),
        )
        dense = triton_dense_mask_logits(
            self.logits,
            self.tensors.dense_mask,
            self.rows,
        )
        bitset = triton_bitset_mask_logits(
            self.logits,
            self.tensors.bitset_mask,
            self.rows,
        )
        self.assertTrue(torch.equal(dense, reference))
        self.assertTrue(torch.equal(bitset, reference))

    def test_argmax_kernels(self) -> None:
        reference = torch.argmax(
            self.logits.masked_fill(
                self.tensors.dense_mask[self.rows.long()].logical_not(),
                -float("inf"),
            ),
            dim=1,
        ).to(torch.int32)
        dense = triton_dense_argmax(
            self.logits,
            self.tensors.dense_mask,
            self.rows,
        )
        bitset = triton_bitset_argmax(
            self.logits,
            self.tensors.bitset_mask,
            self.rows,
        )
        max_nnz = int(self.tables.row_nnz[self.rows.cpu().numpy()].max())
        csr = triton_csr_argmax(
            self.logits,
            self.tensors.csr_indptr,
            self.tensors.csr_indices,
            self.rows,
            max_row_nnz=max_nnz,
        )
        self.assertTrue(torch.equal(dense, reference))
        self.assertTrue(torch.equal(bitset, reference))
        self.assertTrue(torch.equal(csr, reference))
        csr_tokens, csr_states = triton_csr_argmax_advance(
            self.logits,
            self.tensors.csr_indptr,
            self.tensors.csr_indices,
            self.tensors.csr_next_state,
            self.rows,
            max_row_nnz=max_nnz,
        )
        expected_states = self.tensors.next_state[
            self.rows.long(),
            reference.long(),
        ]
        self.assertTrue(torch.equal(csr_tokens, reference))
        self.assertTrue(torch.equal(csr_states, expected_states))

    def test_advance_kernels(self) -> None:
        tokens = torch.argmax(
            self.logits.masked_fill(
                self.tensors.dense_mask[self.rows.long()].logical_not(),
                -float("inf"),
            ),
            dim=1,
        ).to(torch.int32)
        expected = self.tensors.next_state[
            self.rows.long(),
            tokens.long(),
        ]
        dense = triton_dense_advance(
            self.rows,
            tokens,
            self.tensors.next_state,
        )
        byte_dfa = triton_byte_dfa_advance(
            self.rows,
            tokens,
            self.tensors.token_bytes,
            self.tensors.token_lengths,
            self.tensors.byte_transitions,
        )
        self.assertTrue(torch.equal(dense, expected))
        self.assertTrue(torch.equal(byte_dfa, expected))

    def test_packed_csr_handles_empty_rows_and_ties(self) -> None:
        indptr = torch.tensor(
            [0, 0, 2, 5],
            dtype=torch.int32,
            device="cuda",
        )
        indices = torch.tensor(
            [1, 3, 0, 2, 4],
            dtype=torch.int32,
            device="cuda",
        )
        next_states = torch.tensor(
            [11, 13, 20, 22, 24],
            dtype=torch.int32,
            device="cuda",
        )
        rows = torch.tensor(
            [0, 1, 2, 1, 2],
            dtype=torch.int32,
            device="cuda",
        )
        logits = torch.zeros(
            (rows.numel(), 8),
            dtype=torch.float32,
            device="cuda",
        )
        logits[1, 3] = 4
        logits[2, 2] = 5
        logits[3, 1] = 7
        logits[4, 0] = 9

        for rows_per_program in (1, 2, 4, 8, 16):
            tokens, states = triton_csr_argmax_advance_packed(
                logits,
                indptr,
                indices,
                next_states,
                rows,
                max_row_nnz=3,
                rows_per_program=rows_per_program,
                num_warps=4,
            )
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(
                    tokens,
                    torch.tensor(
                        [-1, 3, 2, 1, 0],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                )
            )
            self.assertTrue(
                torch.equal(
                    states,
                    torch.tensor(
                        [0, 13, 22, 11, 20],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                )
            )

    def test_ell_argmax_advance_matches_csr(self) -> None:
        lengths = torch.tensor(
            [0, 2, 3],
            dtype=torch.int32,
            device="cuda",
        )
        tokens = torch.tensor(
            [[0, 0, 0], [1, 3, 0], [0, 2, 4]],
            dtype=torch.int32,
            device="cuda",
        )
        next_states = torch.tensor(
            [[0, 0, 0], [11, 13, 0], [20, 22, 24]],
            dtype=torch.int32,
            device="cuda",
        )
        rows = torch.tensor(
            [0, 1, 2, 1, 2],
            dtype=torch.int32,
            device="cuda",
        )
        logits = torch.zeros(
            (rows.numel(), 8),
            dtype=torch.float32,
            device="cuda",
        )
        logits[1, 3] = 4
        logits[2, 2] = 5
        logits[3, 1] = 7
        logits[4, 0] = 9
        plan = make_ell_argmax_advance_plan(
            logits,
            lengths,
            tokens,
            next_states,
            rows,
        )

        selected_tokens, selected_states = plan(logits)
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(
                selected_tokens,
                torch.tensor(
                    [-1, 3, 2, 1, 0],
                    dtype=torch.int32,
                    device="cuda",
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                selected_states,
                torch.tensor(
                    [0, 13, 22, 11, 20],
                    dtype=torch.int32,
                    device="cuda",
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
