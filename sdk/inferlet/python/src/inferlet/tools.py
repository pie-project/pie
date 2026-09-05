"""
Tool calling — ``pie:inferlet/tools``.

Equip tool specs into the prompt, constrain generation to well-formed calls,
and parse the calls back out::

    from inferlet import tools

    prompt = chat.system(...) + tools.equip([spec_json, ...]) + chat.user(...)
    matcher = tools.create_matcher([spec_json, ...])   # a grammar.Matcher
    ...
    for event in tools.Decoder().feed(tokens): ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

from componentize_py_types import Err as _WitErr
from wit_world.imports import tools as _tools

from .grammar import Grammar, GrammarError, Matcher


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments_json: str


@dataclass(frozen=True)
class Start:
    """A tool-call block opened."""


@dataclass(frozen=True)
class Call:
    """A complete tool call was parsed."""

    call: ToolCall


Event = Union[Start, Call]


def equip(tools: Sequence[str]) -> list[int]:
    """Token sequence declaring ``tools`` (JSON specs) to the model."""
    try:
        return list(_tools.equip(list(tools)))
    except _WitErr as e:
        raise GrammarError(f"tools.equip: {e.value}") from None


def answer(name: str, value: str) -> list[int]:
    """Token sequence returning a tool's result to the model."""
    return list(_tools.answer(name, value))


def format(tools: Sequence[str]) -> Grammar | None:  # noqa: A001 — the WIT name
    """The grammar of a well-formed call to one of ``tools``, if the model's
    template has one."""
    g = _tools.format(list(tools))
    return Grammar(g) if g is not None else None


def create_matcher(tools: Sequence[str]) -> Matcher:
    """A matcher over :func:`format`'s grammar."""
    return Matcher(_inner=_tools.create_matcher(list(tools)))


class Decoder:
    """Parses generated tokens into tool-call events."""

    def __init__(self) -> None:
        self._inner = _tools.Decoder()

    def feed(self, tokens: Sequence[int]) -> Event:
        try:
            ev = self._inner.feed([int(t) for t in tokens])
        except _WitErr as e:
            raise GrammarError(f"tools decoder: {e.value}") from None
        if isinstance(ev, _tools.Event_Call):
            return Call(ToolCall(ev.value.name, ev.value.arguments_json))
        return Start()

    def reset(self) -> None:
        self._inner.reset()
