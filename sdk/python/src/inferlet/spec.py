"""
Speculator protocol for speculative-decoding drafters.

Plug a :class:`Speculator` into a :class:`Generator` via the
``speculator=`` kwarg to drive draft tokens off your own logic.

For host-driven speculation (where the runtime returns next-iter draft
tokens via the forward-pass output's spec channel), pass
``system_speculation=True`` to ``ctx.generate(...)`` instead — that mode
is built into the Generator and does not need a Speculator implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Speculator(Protocol):
    """A speculative-decoding drafter.

    Each iteration the :class:`Generator` asks for ``draft()`` tokens,
    runs the verifier, then reports ``accept()``. On rejection the
    Generator calls ``rollback()`` so the speculator can truncate any
    state it grew during drafting.

    User-defined speculators don't need to inherit — any class with the
    matching methods satisfies this protocol::

        class MyDrafter:
            def draft(self) -> tuple[list[int], list[int]]:
                # produce (tokens, positions)
                ...
            def accept(self, tokens: list[int]) -> None: ...
            def rollback(self, n: int) -> None: ...
            def reset(self) -> None: ...
    """

    def draft(self) -> tuple[list[int], list[int]]:
        """Produce draft tokens and their absolute positions for the next
        forward pass. Return ``([], [])`` for "no speculation this step"."""
        ...

    def accept(self, tokens: list[int]) -> None:
        """Called with the verifier's accepted token sequence. The first
        accepted token corresponds to the anchor's own next-token
        prediction; the rest (if any) are matched drafts."""
        ...

    def rollback(self, n: int) -> None:
        """Roll back the last ``n`` drafted tokens — used when the
        verifier rejects the tail of the draft sequence and the
        speculator's own internal context needs to mirror that
        truncation."""
        ...

    def reset(self) -> None:
        """Reset the speculator to its initial state."""
        ...
