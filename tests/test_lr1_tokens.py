import unittest

import numpy as np
import torch

from gpu_lr1.lr1 import CanonicalLR1Compiler
from gpu_lr1.lr1_tokens import (
    LR1ConfigurationLimitError,
    LR1TokenVocabulary,
    LR1TokenCompileTimeoutError,
    compile_bounded_lr1_token_automaton,
    pack_bounded_lr1_token_automata,
    select_and_advance_bounded_lr1_cpu,
    triton_bounded_lr1_step,
)
from gpu_lr1.lr1_workloads import (
    arithmetic_grammar,
    balanced_grammar,
    byte_arithmetic_grammar,
)
from gpu_lr1.vocab import Vocabulary


def arithmetic_token_vocabulary():
    compiled = CanonicalLR1Compiler(arithmetic_grammar()).compile()
    vocabulary = Vocabulary(
        (b"", b"id", b"+id", b"*id", b"(", b")", b"bad"),
        name="arithmetic-tokens",
    )
    token_vocabulary = LR1TokenVocabulary.from_symbol_sequences(
        compiled,
        vocabulary,
        [
            (),
            ("id",),
            ("+", "id"),
            ("*", "id"),
            ("(",),
            (")",),
            None,
        ],
    )
    return compiled, token_vocabulary


def byte_token_vocabulary():
    compiled = CanonicalLR1Compiler(byte_arithmetic_grammar()).compile()
    vocabulary = Vocabulary(
        (b"", b"1", b"12", b"+3", b"*4", b"(", b")", b"x"),
        name="byte-arithmetic-tokens",
    )
    alphabet = "0123456789+*()"
    token_vocabulary = LR1TokenVocabulary.from_byte_vocabulary(
        compiled,
        vocabulary,
        {ord(symbol): symbol for symbol in alphabet},
    )
    return compiled, token_vocabulary


class BoundedLR1TokenCompilerTest(unittest.TestCase):
    def test_token_can_span_multiple_grammar_terminals(self) -> None:
        compiled, token_vocabulary = arithmetic_token_vocabulary()
        automaton = compile_bounded_lr1_token_automaton(
            compiled,
            token_vocabulary,
            max_stack_depth=16,
        )

        state = automaton.start_state
        state = automaton.next_state(state, 1)
        self.assertGreaterEqual(state, 0)
        state = automaton.next_state(state, 2)
        self.assertGreaterEqual(state, 0)
        state = automaton.next_state(state, 3)
        self.assertGreaterEqual(state, 0)
        self.assertTrue(automaton.accepting[state])
        self.assertEqual(automaton.next_state(state, 0), state)
        self.assertEqual(automaton.next_state(state, 6), -1)

    def test_real_byte_tokens_bridge_to_lr_terminals(self) -> None:
        compiled, token_vocabulary = byte_token_vocabulary()
        automaton = compile_bounded_lr1_token_automaton(
            compiled,
            token_vocabulary,
            max_stack_depth=24,
        )

        state = automaton.start_state
        for token in (2, 3, 4):
            state = automaton.next_state(state, token)
            self.assertGreaterEqual(state, 0)
        self.assertTrue(automaton.accepting[state])
        self.assertEqual(automaton.next_state(state, 0), state)
        self.assertEqual(automaton.next_state(state, 7), -1)
        self.assertEqual(token_vocabulary.terminal_sequences[7], None)

    def test_depth_bound_removes_overflowing_tokens(self) -> None:
        compiled = CanonicalLR1Compiler(balanced_grammar()).compile()
        vocabulary = Vocabulary(
            (b"", b"(", b"((", b")", b"()"),
            name="balanced-byte-tokens",
        )
        token_vocabulary = LR1TokenVocabulary.from_byte_vocabulary(
            compiled,
            vocabulary,
            {ord("("): "(", ord(")"): ")"},
        )
        automaton = compile_bounded_lr1_token_automaton(
            compiled,
            token_vocabulary,
            max_stack_depth=5,
        )

        deep_state = automaton.next_state(automaton.start_state, 2)
        self.assertGreaterEqual(deep_state, 0)
        self.assertEqual(automaton.config_depths[deep_state], 3)
        self.assertEqual(automaton.next_state(deep_state, 4), -1)
        self.assertGreaterEqual(automaton.next_state(deep_state, 3), 0)
        self.assertGreater(automaton.overflow_edges, 0)

    def test_configuration_limit_fails_explicitly(self) -> None:
        compiled, token_vocabulary = arithmetic_token_vocabulary()
        with self.assertRaises(LR1ConfigurationLimitError):
            compile_bounded_lr1_token_automaton(
                compiled,
                token_vocabulary,
                max_stack_depth=16,
                max_configurations=1,
            )

    def test_compile_timeout_fails_explicitly(self) -> None:
        compiled, token_vocabulary = arithmetic_token_vocabulary()
        with self.assertRaises(LR1TokenCompileTimeoutError):
            compile_bounded_lr1_token_automaton(
                compiled,
                token_vocabulary,
                max_stack_depth=16,
                max_compile_seconds=0.0,
            )

    def test_packs_heterogeneous_bounded_automata(self) -> None:
        compiled, token_vocabulary = byte_token_vocabulary()
        automata = [
            compile_bounded_lr1_token_automaton(
                compiled,
                token_vocabulary,
                max_stack_depth=depth,
            )
            for depth in (8, 16)
        ]
        packed = pack_bounded_lr1_token_automata(automata)

        self.assertEqual(packed.num_grammars, 2)
        self.assertEqual(packed.vocab_size, token_vocabulary.size)
        for grammar_id, automaton in enumerate(automata):
            start = int(packed.state_offsets[grammar_id])
            end = int(packed.state_offsets[grammar_id + 1])
            self.assertEqual(end - start, automaton.num_states)
            next_states = packed.csr_next_state[
                packed.csr_indptr[start] : packed.csr_indptr[end]
            ]
            if next_states.size:
                self.assertGreaterEqual(int(next_states.min()), start)
                self.assertLess(int(next_states.max()), end)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BoundedLR1TokenKernelTest(unittest.TestCase):
    def test_packed_csr_gpu_step_matches_cpu(self) -> None:
        arithmetic_compiled, arithmetic_vocab = arithmetic_token_vocabulary()
        byte_compiled, byte_vocab = byte_token_vocabulary()
        shared_vocabulary = Vocabulary(
            (
                b"",
                b"1",
                b"12",
                b"+3",
                b"*4",
                b"(",
                b")",
                b"x",
            ),
            name="shared-byte-tokens",
        )
        arithmetic_tokens = LR1TokenVocabulary.from_symbol_sequences(
            arithmetic_compiled,
            shared_vocabulary,
            [
                (),
                ("id",),
                ("id",),
                ("+", "id"),
                ("*", "id"),
                ("(",),
                (")",),
                None,
            ],
        )
        byte_tokens = LR1TokenVocabulary(
            vocabulary=shared_vocabulary,
            terminal_sequences=byte_vocab.terminal_sequences,
            name=byte_vocab.name,
        )
        automata = [
            compile_bounded_lr1_token_automaton(
                arithmetic_compiled,
                arithmetic_tokens,
                max_stack_depth=16,
            ),
            compile_bounded_lr1_token_automaton(
                byte_compiled,
                byte_tokens,
                max_stack_depth=24,
            ),
        ]
        tables = pack_bounded_lr1_token_automata(automata)
        tensors = tables.torch_tensors()
        states = tables.start_states.copy()
        logits = np.full(
            (2, tables.vocab_size),
            -20.0,
            dtype=np.float32,
        )
        logits[0, 2] = 20.0
        logits[1, 2] = 20.0
        expected_tokens, expected_states = select_and_advance_bounded_lr1_cpu(
            logits,
            tables,
            states,
        )

        tokens, next_states = triton_bounded_lr1_step(
            torch.from_numpy(logits).cuda(),
            tensors,
            torch.from_numpy(states).cuda(),
        )
        torch.cuda.synchronize()

        np.testing.assert_array_equal(
            tokens.cpu().numpy(),
            expected_tokens,
        )
        np.testing.assert_array_equal(
            next_states.cpu().numpy(),
            expected_states,
        )
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            triton_bounded_lr1_step(
                torch.from_numpy(logits[:, :-1]).cuda(),
                tensors,
                torch.from_numpy(states).cuda(),
            )


if __name__ == "__main__":
    unittest.main()
