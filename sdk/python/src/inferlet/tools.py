"""
Optional helpers for tool calling.

`inferlet` does not bake a tool-call loop into the :class:`Generator`
surface — the right loop shape varies a lot between agents (ReAct,
CodeAct, JSON-call, native-grammar) and we'd rather give you the pieces
than a framework.

This module exposes the host's tool-template capability so callers that
*do* want the model's native format can reach for it explicitly:

* :func:`equip_prefix` — token sequence that registers tool schemas in
  the chat template (model-specific). Append before your user message.
* :func:`answer_prefix` — token sequence that frames a tool result for
  the next turn.
* :func:`native_matcher` — a grammar matcher that constrains output to
  well-formed tool calls (or ``None`` if the model has no enforceable
  format). Wrap with :class:`GrammarConstraint` and pass to
  :meth:`Generator.constrain` to enforce well-formed output.
* :class:`Decoder` — streaming detector for tool calls inside generated
  text. Feed each step's tokens; collect :class:`Event.Call` events.

For agents that hand-roll their own format (e.g. ``agent-react``'s
``Action: ToolName[input]`` parsing), none of these are required.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Union

from wit_world.imports import tool_use as _tool
from wit_world.imports.tool_use import (
    Event_Call as _RawCall,
    Event_Start as _RawStart,
)

from .grammar import Matcher
from .model import Model


# =============================================================================
# Templating
# =============================================================================


def equip_prefix(model: Model, tools: list[str | dict]) -> list[int]:
    """Token sequence that registers ``tools`` (each a JSON Schema
    string or dict) in the chat template. Append before your user
    message via :meth:`Context.append`."""
    strs = [t if isinstance(t, str) else _json.dumps(t) for t in tools]
    return list(_tool.equip(model._handle, strs))


def answer_prefix(model: Model, name: str, value: str | dict | list) -> list[int]:
    """Token sequence that frames a tool result for the next turn.
    ``name`` matches the call the model made; ``value`` is typically a
    JSON-serializable result (dict/list auto-stringified)."""
    s = value if isinstance(value, str) else _json.dumps(value)
    return list(_tool.answer(model._handle, name, s))


# =============================================================================
# Native matcher (for grammar enforcement)
# =============================================================================


def native_matcher(model: Model, tools: list[str | dict]) -> Matcher | None:
    """Build a grammar :class:`Matcher` that enforces the model's native
    tool-call format. Returns ``None`` if the model has no enforceable
    format — caller should fall through to free-form generation + their
    own parser.

    Pair with :class:`GrammarConstraint` to pass to
    :meth:`Generator.constrain`::

        matcher = tools.native_matcher(model, schemas)
        if matcher:
            gen = gen.constrain(GrammarConstraint(matcher))
    """
    strs = [t if isinstance(t, str) else _json.dumps(t) for t in tools]
    try:
        raw = _tool.create_matcher(model._handle, strs)
    except Exception:
        return None
    return Matcher._from_handle(raw)


# =============================================================================
# Events
# =============================================================================


class Event:
    """Discriminated union of tool-decoder events. Per ``feed``, exactly
    one event fires.

    :class:`Start` fires while a tool-call structure is being assembled
    but the arguments haven't closed yet — it's both "boundary entered"
    and the no-meaningful-event signal during accumulation. Most callers
    can ignore it and only act on :class:`Call`.
    """

    __slots__ = ()

    @dataclass(frozen=True, slots=True)
    class Start:
        """A tool call is in progress — keep feeding."""

    @dataclass(frozen=True, slots=True)
    class Call:
        """Complete tool call. ``args`` is JSON-encoded."""

        name: str
        args: str


AnyEvent = Union[Event.Start, Event.Call]


# =============================================================================
# Decoder
# =============================================================================


class Decoder:
    """Stateful tool-call decoder. Feed each step's tokens; collect
    :class:`Event.Call` events when complete tool calls are detected."""

    __slots__ = ("_inner",)

    def __init__(self, model: Model) -> None:
        self._inner = _tool.create_decoder(model._handle)

    def feed(self, tokens: list[int]) -> AnyEvent:
        """Feed a token batch and get back the event that fired.
        :class:`Event.Start` indicates an in-progress tool call;
        :class:`Event.Call` fires once when the arguments close."""
        ev = self._inner.feed(tokens)
        if isinstance(ev, _RawStart):
            return Event.Start()
        if isinstance(ev, _RawCall):
            name, args = ev.value
            return Event.Call(name, args)
        return Event.Start()

    def reset(self) -> None:
        """Reset to initial state."""
        self._inner.reset()
