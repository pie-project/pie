"""
Pie Inferlet SDK — Python bindings for the Pie runtime.

An inferlet is a small program that runs next to the model. The forward-pass
surface (`pie:inferlet/forward*`) takes a traced ETA program; this package
traces it from ordinary Python and encodes the canonical container bytes —
the same bytes the Rust SDK's `eta-dsl` emits for the same program, so a
Python inferlet and a Rust inferlet share the host's program cache.

    from inferlet import model, session, chat, reasoning
    from inferlet.eta import *          # Channel, ForwardPass, Pipeline, ops…

    async def main(input):
        ws = WorkingSet()
        ws.reserve(max_pages)
        fwd = ForwardPass()
        fwd.embed(tokens, indptr)
        fwd.bind_state(ws, KvGeometry(...))

        @fwd.epilogue
        def _():
            tok_out.put(reshape(reduce_argmax(intrinsics.logits()), [1]))

        fwd.submit(pipe)
        token = await tok_out.take_scalar()

`bakery build` (or `pie build`) componentizes a Python inferlet with
`componentize-py`; the generated `bindings/` tree is regenerated from
`crates/inferlet/wit` with

    componentize-py -d crates/inferlet/wit -w inferlet bindings <out>
"""

from __future__ import annotations

from . import chat, eta, grammar, mask, media, model, reasoning, session, tokenizer, tools

__all__ = [
    "chat",
    "eta",
    "grammar",
    "mask",
    "media",
    "model",
    "reasoning",
    "session",
    "tokenizer",
    "tools",
]


# --- Internal: return value plumbing for bakery wrapper ---
_return_value: str | None = None


def set_return(value: str) -> None:
    """Set the return value for the inferlet (internal use by bakery wrapper)."""
    global _return_value
    _return_value = value


def get_return_value() -> str | None:
    """Get the return value for the inferlet (internal use by bakery wrapper)."""
    return _return_value
