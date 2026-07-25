"""A vLLM structured-output backend backed by the gpugrammar compiler.

vLLM 0.25 dispatches backends with a hardcoded `if/elif` and has no registry,
unlike SGLang's `register_grammar_backend`. `install()` therefore substitutes
this backend for the name vLLM already knows, which keeps A/B measurement to a
single import and needs no fork. That substitution is a measurement device, not
a shipping plan: the upstream ask is a registry.

What differs from XGrammar underneath: a mask is the union of the token groups
the parser admits, so per-step work is one replay per group — a few hundred,
independent of vocabulary size — rather than a walk over the vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)


def _vocabulary(tokenizer) -> list[bytes]:
    """The tokenizer's byte strings, indexed by token id."""
    tokens: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            tokens.append(b"")
            continue
        try:
            tokens.append(tokenizer.convert_tokens_to_string([piece]).encode("utf-8"))
        except Exception:  # noqa: BLE001
            tokens.append(b"")
    return tokens


@dataclass
class GpuGrammarGrammar(StructuredOutputGrammar):
    matcher: object
    words: int
    stop_token_ids: list[int]
    _terminated: bool = field(default=False, repr=False)
    _processed: int = field(default=0, repr=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        if self._terminated:
            return False
        for token in tokens:
            if token in self.stop_token_ids:
                if not self.matcher.can_terminate():
                    return False
                self.matcher.terminate()
                self._terminated = True
                return True
            if not self.matcher.accept_token(token):
                return False
            self._processed += 1
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """The longest prefix the grammar accepts, leaving the state untouched.

        This is the speculative-decoding hook: a draft is checked here and only
        the accepted prefix is committed afterwards.
        """
        accepted: list[int] = []
        for token in tokens:
            if token in self.stop_token_ids:
                if self.matcher.can_terminate():
                    accepted.append(token)
                break
            if not self.matcher.accept_token(token):
                break
            accepted.append(token)
        if accepted:
            self.matcher.rollback(len(accepted))
        return accepted

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self._processed = max(0, self._processed - num_tokens)
        self._terminated = False

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        row = bitmask[batch_index]
        row.zero_()
        self.matcher.fill_bitmask(row)
        # vLLM stops on a stop token, so it has to remain reachable once the
        # document is complete.
        if self.matcher.can_terminate():
            for stop in self.stop_token_ids:
                row[stop // 32] |= 1 << (stop % 32)

    def is_terminated(self) -> bool:
        return self._terminated

    def reset(self) -> None:
        self.matcher.reset()
        self._terminated = False
        self._processed = 0


@dataclass
class GpuGrammarBackend(StructuredOutputBackend):
    def __post_init__(self) -> None:
        import gpugrammar

        self.compiler = gpugrammar.Compiler(_vocabulary(self.tokenizer))
        self.stop_token_ids = [
            token
            for token in [getattr(self.tokenizer, "eos_token_id", None)]
            if token is not None
        ]
        self.words = (self.vocab_size + 31) // 32
        self.compiled: dict[tuple, object] = {}

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        key = (request_type, grammar_spec)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = self._compile(request_type, grammar_spec)
            self.compiled[key] = compiled
        return GpuGrammarGrammar(
            matcher=compiled.matcher(32),
            words=compiled.bitset_words,
            stop_token_ids=self.stop_token_ids,
        )

    def _compile(self, request_type: StructuredOutputOptions, grammar_spec: str):
        if request_type == StructuredOutputOptions.JSON:
            return self.compiler.compile_json_schema(grammar_spec)
        if request_type == StructuredOutputOptions.JSON_OBJECT:
            return self.compiler.compile_json_schema(json.dumps({"type": "object"}))
        if request_type == StructuredOutputOptions.REGEX:
            return self.compiler.compile_regex(grammar_spec)
        if request_type == StructuredOutputOptions.GRAMMAR:
            return self.compiler.compile_ebnf(grammar_spec, "root")
        raise ValueError(f"gpugrammar does not support {request_type}")

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        # Filled rows are written wholesale; unfilled rows must allow
        # everything, which is what all-ones means here.
        return torch.full(
            (max_num_seqs, self.words), -1, dtype=torch.int32, device="cpu"
        )

    def destroy(self) -> None:
        self.compiled.clear()


def install() -> None:
    """Take over the `xgrammar` backend name so vLLM constructs this instead."""
    import vllm.v1.structured_output as structured_output

    structured_output.XgrammarBackend = GpuGrammarBackend
