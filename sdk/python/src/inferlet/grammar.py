"""
Grammar and Matcher for structured output generation.

Wraps ``pie:core/inference`` Grammar + Matcher resources.
"""

from __future__ import annotations

from wit_world.imports.inference import Grammar as _Grammar
from wit_world.imports.inference import Matcher as _Matcher

from .model import Tokenizer


class Grammar:
    """Describes the structure that LLM output must conform to.

    Usage::

        grammar = Grammar.from_json_schema('{"type": "object", ...}')
        grammar = Grammar.json()
        grammar = Grammar.from_regex(r"\\d{3}-\\d{4}")
        grammar = Grammar.from_ebnf('root ::= "hello" | "world"')
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: _Grammar) -> None:
        self._handle = handle

    @classmethod
    def from_json_schema(cls, schema: str) -> Grammar:
        """Construct from a JSON Schema string."""
        return cls(_Grammar.from_json_schema(schema))

    @classmethod
    def json(cls) -> Grammar:
        """Construct a grammar for any valid JSON."""
        return cls(_Grammar.json())

    @classmethod
    def from_regex(cls, pattern: str) -> Grammar:
        """Construct from a regular expression."""
        return cls(_Grammar.from_regex(pattern))

    @classmethod
    def from_ebnf(cls, ebnf: str) -> Grammar:
        """Construct from an EBNF grammar string."""
        return cls(_Grammar.from_ebnf(ebnf))


class Matcher:
    """Stateful grammar matcher that produces token masks.

    Usage::

        grammar = Grammar.from_json_schema(schema)
        matcher = Matcher(grammar, tokenizer)

        while not matcher.is_terminated:
            mask = matcher.next_token_logit_mask()
            matcher.accept_tokens(generated_tokens)
    """

    __slots__ = ("_handle",)

    def __init__(self, grammar: Grammar, tokenizer: Tokenizer) -> None:
        self._handle = _Matcher(grammar._handle, tokenizer._handle)

    def accept_tokens(self, token_ids: list[int]) -> None:
        """Accept tokens, advancing the matcher state."""
        self._handle.accept_tokens(token_ids)

    def next_token_logit_mask(self) -> list[int]:
        """Get the BRLE-encoded bitmask of valid next tokens."""
        return list(self._handle.next_token_logit_mask())

    @property
    def is_terminated(self) -> bool:
        """Whether the matcher has reached a terminal state."""
        return self._handle.is_terminated()

    def reset(self) -> None:
        """Reset to initial state for reuse."""
        self._handle.reset()
