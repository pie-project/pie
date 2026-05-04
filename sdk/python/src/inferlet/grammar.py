"""
Grammar, Matcher, GrammarConstraint, and Schema for constrained decoding.

The common path is declarative: pass a :class:`Schema` to
``ctx.generate(..., constrain=...)`` (or any class implementing the
:class:`Schema` protocol), and the SDK compiles it into a stateful
matcher and drives it per generated token::

    text = await ctx.generate(
        Sampler.argmax(),
        constrain=Ebnf(grammar),
        max_tokens=512,
    ).collect_text()

Built-in implementors:

* :class:`JsonSchema` — JSON conforming to a JSON Schema string
* :class:`AnyJson`    — any valid JSON
* :class:`Regex`      — strings matching a regex pattern
* :class:`Ebnf`       — custom EBNF grammar

User code can implement the :class:`Schema` protocol on any class with a
``build_constraint(model)`` method — duck-typed, no inheritance required.

For custom logic that isn't a grammar (banned tokens, learned constraints,
etc.), implement the :class:`Constraint` protocol and pass it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from wit_world.imports.inference import Grammar as _Grammar
from wit_world.imports.inference import Matcher as _Matcher

from .model import Model, Tokenizer


# =============================================================================
# Grammar / Matcher (raw resource wrappers)
# =============================================================================


class Grammar:
    """Describes the structure that LLM output must conform to.

    Use the static factories::

        Grammar.from_json_schema('{"type": "object", ...}')
        Grammar.json()
        Grammar.from_regex(r"\\d{3}-\\d{4}")
        Grammar.from_ebnf('root ::= "hello" | "world"')
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
    """Stateful grammar matcher producing token masks.

    Most callers should reach for :class:`GrammarConstraint` instead —
    Matcher is the lower-level resource wrapper.
    """

    __slots__ = ("_handle",)

    def __init__(self, grammar: Grammar, tokenizer: Tokenizer) -> None:
        self._handle = _Matcher(grammar._handle, tokenizer._handle)

    @classmethod
    def _from_handle(cls, handle: _Matcher) -> Matcher:
        """Internal: wrap a pre-existing host matcher."""
        obj = object.__new__(cls)
        obj._handle = handle
        return obj

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


# =============================================================================
# Constraint protocol + GrammarConstraint
# =============================================================================


@runtime_checkable
class Constraint(Protocol):
    """Stateful sampling constraint protocol.

    On each generation step, the Generator passes any newly accepted
    tokens (or ``[]`` on the first step) and gets back the BRLE-encoded
    logit mask for the next position.

    Returning ``[]`` (empty mask) means "no restriction" and is treated
    as transparent during composition.
    """

    def step(self, accepted: list[int]) -> list[int]:
        """Advance internal state with ``accepted`` tokens, then return
        the mask for the next position."""
        ...


class GrammarConstraint:
    """Grammar-driven :class:`Constraint` backed by a host
    :class:`Matcher`.

    Most callers should reach for a :class:`Schema` implementor instead
    — :class:`GrammarConstraint` is the lower-level type for callers
    that want to keep a constraint instance around (e.g., for use with
    :func:`tools.native_matcher`).
    """

    __slots__ = ("_matcher",)

    def __init__(self, matcher: Matcher) -> None:
        self._matcher = matcher

    @classmethod
    def from_grammar(cls, grammar: Grammar, model: Model) -> GrammarConstraint:
        """Build from a pre-compiled grammar (compile once, reuse)."""
        return cls(Matcher(grammar, model.tokenizer()))

    @classmethod
    def from_json_schema(cls, schema: str, model: Model) -> GrammarConstraint:
        """Build from a JSON Schema string."""
        return cls.from_grammar(Grammar.from_json_schema(schema), model)

    @classmethod
    def json(cls, model: Model) -> GrammarConstraint:
        """Build a constraint that accepts any valid JSON."""
        return cls.from_grammar(Grammar.json(), model)

    @classmethod
    def from_regex(cls, pattern: str, model: Model) -> GrammarConstraint:
        """Build from a regular expression pattern."""
        return cls.from_grammar(Grammar.from_regex(pattern), model)

    @classmethod
    def from_ebnf(cls, ebnf: str, model: Model) -> GrammarConstraint:
        """Build from an EBNF grammar string."""
        return cls.from_grammar(Grammar.from_ebnf(ebnf), model)

    def step(self, accepted: list[int]) -> list[int]:
        if accepted:
            self._matcher.accept_tokens(accepted)
        return self._matcher.next_token_logit_mask()


# =============================================================================
# Schema protocol + built-in implementors
# =============================================================================


@runtime_checkable
class Schema(Protocol):
    """Declarative description of a constraint.

    Implementations are passed to ``ctx.generate(constrain=...)`` (or
    :meth:`Generator.constrain`) and compiled into a
    :class:`GrammarConstraint`.

    User code can implement this protocol on any class — duck-typed, no
    inheritance required::

        class MyLark:
            def __init__(self, source): self.source = source
            def build_constraint(self, model):
                g = compile_lark_to_pie_grammar(self.source)
                return GrammarConstraint.from_grammar(g, model)

        await ctx.generate(Sampler.argmax(), constrain=MyLark(grammar)).collect_text()
    """

    def build_constraint(self, model: Model) -> GrammarConstraint:
        """Compile this schema into a :class:`GrammarConstraint`."""
        ...


@dataclass(frozen=True, slots=True)
class JsonSchema:
    """JSON conforming to a JSON Schema string."""

    schema: str

    def build_constraint(self, model: Model) -> GrammarConstraint:
        return GrammarConstraint.from_json_schema(self.schema, model)


@dataclass(frozen=True, slots=True)
class AnyJson:
    """Any valid JSON value."""

    def build_constraint(self, model: Model) -> GrammarConstraint:
        return GrammarConstraint.json(model)


@dataclass(frozen=True, slots=True)
class Regex:
    """Strings matching a regular expression pattern."""

    pattern: str

    def build_constraint(self, model: Model) -> GrammarConstraint:
        return GrammarConstraint.from_regex(self.pattern, model)


@dataclass(frozen=True, slots=True)
class Ebnf:
    """A custom EBNF grammar."""

    source: str

    def build_constraint(self, model: Model) -> GrammarConstraint:
        return GrammarConstraint.from_ebnf(self.source, model)


# =============================================================================
# BRLE intersection (for composing constraint masks)
# =============================================================================


def _brle_and(a: list[int], b: list[int]) -> list[int]:
    """AND two BRLE-encoded masks of equal length."""
    if not a or not b:
        return []
    out: list[int] = []
    a_idx = b_idx = 0
    a_left, b_left = a[0], b[0]
    a_value = b_value = False
    want_value = False
    accum = 0
    while True:
        take = min(a_left, b_left)
        result = a_value and b_value
        if result == want_value:
            accum += take
        else:
            out.append(accum)
            accum = take
            want_value = not want_value
        a_left -= take
        b_left -= take
        if a_left == 0:
            a_idx += 1
            if a_idx == len(a):
                break
            a_left = a[a_idx]
            a_value = not a_value
        if b_left == 0:
            b_idx += 1
            if b_idx == len(b):
                break
            b_left = b[b_idx]
            b_value = not b_value
    out.append(accum)
    return out


def _brle_and_many(masks: list[list[int]]) -> list[int]:
    """Reduce a list of BRLE masks via AND. Empty input returns []."""
    if not masks:
        return []
    if len(masks) == 1:
        return masks[0]
    acc = masks[0]
    for m in masks[1:]:
        acc = _brle_and(acc, m)
    return acc
