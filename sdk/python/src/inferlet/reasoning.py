"""
Reasoning / thinking-block decoder.

Wraps the host's ``pie:instruct/reasoning.decoder``. Emits :class:`Event.Start`
when the model enters a thinking block, :class:`Event.Delta` for each chunk
of reasoning text, and :class:`Event.End` when the block closes (with the
full accumulated reasoning text).

Compose with :class:`chat.Decoder` by feeding the same token batch to both
— the reasoning decoder's events are independent of chat's (no implicit
suppression). The chat decoder handles its own filtering so visible text
and reasoning text don't overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from wit_world.imports import reasoning as _reasoning
from wit_world.imports.reasoning import (
    Event_Complete as _RawComplete,
    Event_Delta as _RawDelta,
    Event_Start as _RawStart,
)


# =============================================================================
# Events
# =============================================================================


class Event:
    """Discriminated union of reasoning-decoder events. Match with
    ``match`` / ``case``::

        match decoder.feed(tokens):
            case reasoning.Event.Start(): in_thinking = True
            case reasoning.Event.Delta(text=t): print(t, end="", file=sys.stderr)
            case reasoning.Event.End(text=full): in_thinking = False
            case _: pass
    """

    __slots__ = ()

    @dataclass(frozen=True, slots=True)
    class Idle:
        """No reasoning boundary crossed in this batch."""

    @dataclass(frozen=True, slots=True)
    class Start:
        """The model entered a reasoning block. No text yet."""

    @dataclass(frozen=True, slots=True)
    class Delta:
        """Streamed chunk of reasoning text (post-detokenization). Always
        non-empty."""

        text: str

    @dataclass(frozen=True, slots=True)
    class End:
        """The block closed — ``text`` is the full accumulated reasoning
        text from ``Start`` to ``End``."""

        text: str


AnyEvent = Union[
    Event.Idle,
    Event.Start,
    Event.Delta,
    Event.End,
]


# =============================================================================
# Decoder
# =============================================================================


class Decoder:
    """Stateful reasoning decoder. Feed token batches in order; one event
    per call."""

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        self._inner = _reasoning.create_decoder()

    def feed(self, tokens: list[int]) -> AnyEvent:
        """Feed a token batch and get back the event that fired. Returns
        :class:`Event.Idle` when the batch landed outside any reasoning
        block, or inside one but on tokens that produced no visible
        reasoning text."""
        ev = self._inner.feed(tokens)
        if isinstance(ev, _RawStart):
            return Event.Start()
        if isinstance(ev, _RawDelta):
            if not ev.value:
                return Event.Idle()
            return Event.Delta(ev.value)
        if isinstance(ev, _RawComplete):
            return Event.End(ev.value)
        return Event.Idle()

    def reset(self) -> None:
        """Reset to initial state."""
        self._inner.reset()
