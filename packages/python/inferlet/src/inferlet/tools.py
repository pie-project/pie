"""
Tool calling — ``pie:inferlet/tools``.

Open a conversation with tool specs, constrain generation to well-formed
calls, parse the calls back out, and answer them::

    from inferlet import chat, grammar, tools

    prompt = chat.open("...", [spec_json, ...]) + chat.user(...)
    matcher = grammar.Matcher(tools.format([spec_json, ...]))
    ...
    match tools.Decoder().feed(tokens):
        case tools.Event.Call(call=c): ...
        case tools.Event.Start(): ...
    prompt += tools.answers([(c.name, result)])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

from componentize_py_types import Err as _WitErr
from wit_world.imports import tools as _tools

from .grammar import Grammar, GrammarError


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments_json: str


class Event:
    """Discriminated union of tool-decoder events, spelled like
    :class:`chat.Event` (match with ``match`` / ``case``)."""

    __slots__ = ()

    @dataclass(frozen=True, slots=True)
    class Start:
        """A tool-call block opened."""

    @dataclass(frozen=True, slots=True)
    class Call:
        """A complete tool call was parsed."""

        call: ToolCall


AnyEvent = Union[Event.Start, Event.Call]


def answers(results: Sequence[tuple[str, str]]) -> list[int]:
    """Tool results ``(name, value)`` that follow one another, as the
    model's template writes them together; one result is a list of one."""
    return list(_tools.answers([(name, value) for name, value in results]))


def assistant_calls(message: str, calls: Sequence[ToolCall], current: bool) -> list[int]:
    """An earlier assistant turn: ``message``, then the ``calls`` it made in
    the template's call syntax. ``current`` marks a turn after the
    conversation's last user query, which some templates replay with its
    reasoning."""
    raw = [_tools.ToolCall(c.name, c.arguments_json) for c in calls]
    return list(_tools.assistant_calls(message, raw, current))


def format(tools: Sequence[str]) -> Grammar | None:  # noqa: A001 — the WIT name
    """The grammar of a well-formed call to one of ``tools``, if the model's
    template has one."""
    g = _tools.format(list(tools))
    return Grammar(g) if g is not None else None


class Decoder:
    """Parses generated tokens into tool-call events."""

    def __init__(self) -> None:
        self._inner = _tools.Decoder()

    def feed(self, tokens: Sequence[int]) -> AnyEvent:
        try:
            ev = self._inner.feed([int(t) for t in tokens])
        except _WitErr as e:
            raise GrammarError(f"tools decoder: {e.value}") from None
        if isinstance(ev, _tools.Event_Call):
            return Event.Call(ToolCall(ev.value.name, ev.value.arguments_json))
        return Event.Start()

    def reset(self) -> None:
        self._inner.reset()
