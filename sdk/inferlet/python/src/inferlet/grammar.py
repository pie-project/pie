"""
Grammar compilation + incremental matching — ``pie:inferlet/grammar``.

:class:`Grammar` compiles a JSON Schema / regex / EBNF source once for the
bound model's vocabulary; :class:`Matcher` walks one generation against it,
exposing the packed allowed-token bitmask that :mod:`inferlet.mask`
interprets::

    from inferlet import grammar, mask
    from inferlet.eta import *

    constraint = grammar.Matcher(grammar.Grammar.from_json_schema(schema))
    allowed = Channel([vocab], dtype.bool)
    allowed.put(mask.unpack_mask(constraint.mask(), vocab))
    ...
    constraint.accept_tokens([token])
"""

from __future__ import annotations

from typing import Sequence

from componentize_py_types import Err as _WitErr
from wit_world.imports import grammar as _grammar


class GrammarError(Exception):
    """A grammar the host refused to compile, or a token the matcher refused."""


def _wit(what: str, fn, *args):
    try:
        return fn(*args)
    except _WitErr as e:
        raise GrammarError(f"{what}: {e.value}") from None


class Grammar:
    """A compiled constraint grammar."""

    __slots__ = ("_inner",)

    def __init__(self, inner) -> None:
        self._inner = inner

    @classmethod
    def from_json_schema(cls, schema: str) -> "Grammar":
        """A grammar for JSON values conforming to ``schema``."""
        return cls(_wit("grammar from JSON schema", _grammar.Grammar.from_json_schema, schema))

    @classmethod
    def json(cls) -> "Grammar":
        """The grammar of any JSON value."""
        return cls(_grammar.Grammar.json())

    @classmethod
    def from_regex(cls, pattern: str) -> "Grammar":
        return cls(_wit("grammar from regex", _grammar.Grammar.from_regex, pattern))

    @classmethod
    def from_ebnf(cls, ebnf: str) -> "Grammar":
        return cls(_wit("grammar from EBNF", _grammar.Grammar.from_ebnf, ebnf))

    def __str__(self) -> str:
        return self._inner.to_string()


class Matcher:
    """An incremental match of one generation against a :class:`Grammar`."""

    __slots__ = ("_inner",)

    def __init__(self, grammar: Grammar | None = None, *, _inner=None) -> None:
        if _inner is not None:
            self._inner = _inner
        elif grammar is not None:
            self._inner = _grammar.Matcher(grammar._inner)
        else:
            raise TypeError("Matcher(grammar)")

    def accept_tokens(self, token_ids: Sequence[int]) -> None:
        """Advance the match by ``token_ids``; a token the grammar forbids
        raises :class:`GrammarError`."""
        _wit("accept tokens", self._inner.accept_tokens, [int(t) for t in token_ids])

    def mask(self) -> list[int]:
        """The packed allowed-token bitmask for the next position (see
        :mod:`inferlet.mask`)."""
        return list(self._inner.mask())

    def is_terminated(self) -> bool:
        return self._inner.is_terminated()

    def reset(self) -> None:
        self._inner.reset()

    def fork(self) -> "Matcher":
        """An independent copy at the current position (branching)."""
        return Matcher(_inner=self._inner.fork())

    def rollback(self, num_tokens: int) -> None:
        self._inner.rollback(num_tokens)

    def rollback_capacity(self) -> int:
        return self._inner.rollback_capacity()
