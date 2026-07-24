import unittest

import numpy as np

from gpu_lr1.tables import (
    compile_named_schemas,
    compile_packed_tables,
    unpack_bitset32,
)
from gpu_lr1.vocab import Vocabulary
from gpu_lr1.workloads import benchmark_schemas


class PackedTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas = benchmark_schemas(3)
        cls.vocabulary = Vocabulary.synthetic(
            512,
            [item.schema for item in cls.schemas],
        )
        cls.compiled = compile_named_schemas(cls.schemas)
        cls.tables = compile_packed_tables(
            cls.compiled,
            cls.vocabulary,
            state_chunk_size=32,
        )

    def test_bitset_roundtrip(self) -> None:
        unpacked = unpack_bitset32(
            self.tables.bitset_mask,
            self.vocabulary.size,
        )
        np.testing.assert_array_equal(unpacked, self.tables.dense_mask)

    def test_flat_offsets_keep_schemas_disjoint(self) -> None:
        for schema_id, compiled in enumerate(self.compiled):
            start = int(self.tables.state_offsets[schema_id])
            end = int(self.tables.state_offsets[schema_id + 1])
            transitions = self.tables.byte_transitions[start:end]
            self.assertGreaterEqual(int(transitions.min()), start)
            self.assertLess(int(transitions.max()), end)
            self.assertEqual(
                int(self.tables.start_states[schema_id]),
                start + compiled.dfa.start_state,
            )

    def test_token_table_matches_byte_dfa(self) -> None:
        rng = np.random.default_rng(7)
        for schema_id, compiled in enumerate(self.compiled):
            offset = int(self.tables.state_offsets[schema_id])
            for _ in range(30):
                local_state = int(rng.integers(0, compiled.dfa.num_states))
                token_id = int(rng.integers(0, self.vocabulary.size))
                token = self.vocabulary.tokens[token_id]
                expected = compiled.dfa.advance(local_state, token)
                if token_id == self.vocabulary.eos_token_id:
                    expected = local_state
                    allowed = bool(compiled.dfa.accepting[local_state])
                else:
                    allowed = expected != compiled.dfa.dead_state
                global_state = offset + local_state
                self.assertEqual(
                    int(self.tables.next_state[global_state, token_id]),
                    offset + expected,
                )
                self.assertEqual(
                    bool(self.tables.dense_mask[global_state, token_id]),
                    allowed,
                )


if __name__ == "__main__":
    unittest.main()

