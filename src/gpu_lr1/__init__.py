"""The research tree: kernels, benchmarks, and the prototypes that came first.

The library is `gpugrammar`; this is what it is built on and what it was built
out of. The current engine is `gpu_lr1.device_parser`, re-exported as
`gpugrammar.device`. The names below are the earlier pure-Python prototype -
a canonical LR(1) compiler and a byte-DFA schema compiler - kept because their
tests still pass and they are the record of how the design arrived where it is.
Nothing in the current path imports them.
"""

from gpu_lr1.automata import ByteDFA
from gpu_lr1.lr1 import (
    CanonicalLR1Compiler,
    Grammar,
    LR1ConflictError,
    Production,
)
from gpu_lr1.lr1_tokens import (
    BoundedLR1TokenAutomaton,
    LR1ConfigurationLimitError,
    LR1TokenCompileTimeoutError,
    LR1TokenVocabulary,
    compile_bounded_lr1_token_automaton,
    make_bounded_lr1_step_plan,
    pack_bounded_lr1_token_automata,
)
from gpu_lr1.schema import CanonicalJSONSchemaCompiler, UnsupportedSchemaError
from gpu_lr1.vocab import Vocabulary

__all__ = [
    "ByteDFA",
    "BoundedLR1TokenAutomaton",
    "CanonicalJSONSchemaCompiler",
    "CanonicalLR1Compiler",
    "Grammar",
    "LR1ConfigurationLimitError",
    "LR1ConflictError",
    "LR1TokenCompileTimeoutError",
    "LR1TokenVocabulary",
    "Production",
    "UnsupportedSchemaError",
    "Vocabulary",
    "compile_bounded_lr1_token_automaton",
    "make_bounded_lr1_step_plan",
    "pack_bounded_lr1_token_automata",
]
