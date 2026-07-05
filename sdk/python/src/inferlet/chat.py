"""
Chat-template templating + parsing.

Two halves:

1. **Fillers** (:func:`system`, :func:`user`, :func:`assistant`,
   :func:`cue`, :func:`seal`) produce token sequences for the model's
   chat template. The :class:`Context` calls them through its
   ``system`` / ``user`` / ``cue`` / ``seal`` methods; for inferlets
   building prompts manually (no Context buffering), these are the
   public entry points.

2. **Decoder** (:class:`Decoder`, :class:`Event`) parses the model's
   generated tokens back into visible text + structural events.

Both halves wrap the host's ``pie:instruct/chat`` interface — chat
template knowledge lives in the Pie runtime, not in the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from wit_world.imports import chat as _chat
from wit_world.imports.chat import (
    Event_Delta as _RawDelta,
    Event_Done as _RawDone,
    Event_Interrupt as _RawInterrupt,
)


# =============================================================================
# Template fillers
# =============================================================================


def system(message: str) -> list[int]:
    """Token sequence for a system-role message."""
    return list(_chat.system(message))


def user(message: str) -> list[int]:
    """Token sequence for a user-role message."""
    return list(_chat.user(message))


def assistant(message: str) -> list[int]:
    """Token sequence for an assistant-role message (history replay)."""
    return list(_chat.assistant(message))


def cue() -> list[int]:
    """Token sequence for the generation cue (tells the model "your turn")."""
    return list(_chat.cue())


def seal() -> list[int]:
    """Token sequence that seals the current turn (inserts a stop token)."""
    return list(_chat.seal())


def stop_tokens() -> list[int]:
    """Stop-token IDs for the model's chat template — pass to
    :meth:`Generator.stop` for explicit termination control."""
    return list(_chat.stop_tokens())


# =============================================================================
# Events
# =============================================================================
#
# Per ``feed``, exactly one event fires. ``Idle`` is the no-op signal —
# the batch was consumed but didn't cross a semantic boundary worth
# surfacing (e.g. landed on a token whose visible text is empty, or
# inside a region this decoder doesn't report on like a reasoning block).


class Event:
    """Discriminated union of chat-decoder events. Match with
    ``match`` / ``case``::

        match decoder.feed(tokens):
            case chat.Event.Delta(text=t): print(t, end="")
            case chat.Event.Done(text=full): break
            case _: pass
    """

    __slots__ = ()

    @dataclass(frozen=True, slots=True)
    class Idle:
        """No semantic boundary crossed in this batch."""

    @dataclass(frozen=True, slots=True)
    class Delta:
        """Streamed text chunk (post-detokenization, post-template-strip).
        Always non-empty."""

        text: str

    @dataclass(frozen=True, slots=True)
    class Done:
        """End-of-turn reached — ``text`` is the full accumulated text
        since the last ``reset()``."""

        text: str

    @dataclass(frozen=True, slots=True)
    class Interrupt:
        """The model emitted a special / control token that the chat
        template recognized but didn't lower to visible text. The id is
        surfaced raw so the caller can decide what to do.

        Common cases this fires for:

        * Tool-call boundary markers (e.g. ``<|tool_call|>`` in some
          templates) — useful as an early-stop hint when you don't have
          :class:`tools.Decoder` attached.
        * Custom control tokens injected by fine-tuned models.
        * Format markers (turn boundaries, role separators) the host
          template chose to expose rather than swallow.

        Most callers ignore this branch.
        """

        token: int


AnyEvent = Union[
    Event.Idle,
    Event.Delta,
    Event.Done,
    Event.Interrupt,
]


# =============================================================================
# Decoder
# =============================================================================


class Decoder:
    """Stateful chat decoder. Feed token batches in order; one event per
    call. ``reset()`` returns the decoder to its initial state."""

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        self._inner = _chat.create_decoder()

    def feed(self, tokens: list[int]) -> AnyEvent:
        """Feed a token batch and get back the event that fired (one per
        call). Returns :class:`Event.Idle` when nothing semantically
        happened — e.g. the batch landed on a token whose visible text
        is empty, or inside a region this decoder doesn't report on."""
        ev = self._inner.feed(tokens)
        if isinstance(ev, _RawDelta):
            # Empty delta means tokens consumed produced no visible character —
            # surface as Idle, not Delta("") so the user doesn't need a
            # ``if text:`` guard.
            if not ev.value:
                return Event.Idle()
            return Event.Delta(ev.value)
        if isinstance(ev, _RawDone):
            return Event.Done(ev.value)
        if isinstance(ev, _RawInterrupt):
            return Event.Interrupt(ev.value)
        return Event.Idle()

    def reset(self) -> None:
        """Reset to initial state."""
        self._inner.reset()
