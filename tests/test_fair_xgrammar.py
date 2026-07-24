import unittest

import numpy as np
import torch

from gpu_lr1.lr1 import CanonicalLR1Compiler
from gpu_lr1.lr1_tokens import (
    LR1TokenVocabulary,
    compile_bounded_lr1_token_automaton,
)
from gpu_lr1.lr1_workloads import (
    bounded_arithmetic_ebnf,
    bounded_balanced_ebnf,
    bounded_balanced_grammar,
    bounded_byte_arithmetic_grammar,
)
from gpu_lr1.vocab import Vocabulary

try:
    import xgrammar as xgr
except ImportError:
    xgr = None


@unittest.skipIf(xgr is None, "XGrammar baseline is not installed")
class FairXGrammarLanguageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = Vocabulary(
            (
                b"",
                b"0",
                b"1",
                b"2",
                b"3",
                b"+",
                b"*",
                b"(",
                b")",
                b"12",
                b"+3",
                b"*2",
                b"()",
                b"(())",
                b"(1+2)",
                b"invalid",
            ),
            name="fair-xgrammar-test",
        )
        tokenizer_info = xgr.TokenizerInfo(
            list(cls.vocabulary.tokens),
            vocab_type=xgr.VocabType.RAW,
            vocab_size=cls.vocabulary.size,
            stop_token_ids=[cls.vocabulary.eos_token_id],
        )
        cls.xcompiler = xgr.GrammarCompiler(tokenizer_info)

    def test_arithmetic_masks_match_on_reachable_prefixes(self) -> None:
        self._compare(
            bounded_byte_arithmetic_grammar(2),
            bounded_arithmetic_ebnf(2),
            {ord(symbol): symbol for symbol in "0123456789+*()"},
            stack_depth=32,
            samples=100,
        )

    def test_balanced_masks_match_on_all_reachable_prefixes(self) -> None:
        self._compare(
            bounded_balanced_grammar(3),
            bounded_balanced_ebnf(3),
            {ord("("): "(", ord(")"): ")"},
            stack_depth=16,
            samples=None,
        )

    def _compare(
        self,
        grammar,
        ebnf: str,
        byte_terminals: dict[int, str],
        *,
        stack_depth: int,
        samples: int | None,
    ) -> None:
        compiled = CanonicalLR1Compiler(grammar).compile()
        token_vocabulary = LR1TokenVocabulary.from_byte_vocabulary(
            compiled,
            self.vocabulary,
            byte_terminals,
        )
        automaton = compile_bounded_lr1_token_automaton(
            compiled,
            token_vocabulary,
            max_stack_depth=stack_depth,
            max_configurations=50_000,
        )
        self.assertEqual(automaton.overflow_edges, 0)
        xcompiled = self.xcompiler.compile_grammar(
            xgr.Grammar.from_ebnf(ebnf)
        )
        state_ids = np.arange(automaton.num_states)
        if samples is not None and state_ids.size > samples:
            rng = np.random.default_rng(7)
            state_ids = rng.choice(state_ids, samples, replace=False)

        for state in state_ids:
            matcher = xgr.GrammarMatcher(xcompiled)
            for token in automaton.config_witness_tokens[state]:
                self.assertTrue(matcher.accept_token(token))
            mask = torch.empty(
                xgr.get_bitmask_shape(1, self.vocabulary.size),
                dtype=xgr.bitmask_dtype,
            )
            matcher.fill_next_token_bitmask(mask)
            bits = np.unpackbits(
                mask.numpy().view(np.uint8),
                bitorder="little",
            )[: self.vocabulary.size]
            xgrammar_tokens = np.flatnonzero(bits)
            start = int(automaton.csr_indptr[state])
            end = int(automaton.csr_indptr[state + 1])
            np.testing.assert_array_equal(
                xgrammar_tokens,
                automaton.csr_indices[start:end],
            )


if __name__ == "__main__":
    unittest.main()
