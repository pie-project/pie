"""
The Pie inferlet library — Python bindings for the Pie runtime.

An inferlet is a small program that runs next to the model. The forward-pass
surface (`pie:inferlet/forward`) takes a traced ETA program; this package
traces it from ordinary Python and encodes the canonical container bytes —
the same bytes the Rust `inferlet` crate's `eta-dsl` emits for the same program, so a
Python inferlet and a Rust inferlet share the host's program cache.

    from inferlet import model
    from inferlet.eta import *          # Channel, ForwardPass, Pipeline, ops…

    async def main(input):
        ws, pipe = WorkingSet(), Pipeline()
        fwd = ForwardPass()
        fwd.embed(model.encode(input["prompt"]))
        fwd.state([ws], StateGeometry.folding())
        tok_out = fwd.epilogue(lambda: reduce_argmax(intrinsics.output(0, "logits")))
        pipe.submit(fwd)
        token = await tok_out

A Python inferlet is run as source by the Python language component
(`language/build.sh` builds it with `componentize-py`); the generated
`bindings/` tree is regenerated from `crates/inferlet/wit` with

    componentize-py -d crates/inferlet/wit -w inferlet bindings <out>
"""

from __future__ import annotations

from . import chat, eta, frames, grammar, mask, media, model, reasoning, session, tokenizer, tools

__all__ = [
    "chat",
    "eta",
    "frames",
    "grammar",
    "mask",
    "media",
    "model",
    "reasoning",
    "session",
    "tokenizer",
    "tools",
]


# --- Internal: return value plumbing for the language component wrapper ---
_return_value: str | None = None


def set_return(value: str) -> None:
    """Set the return value for the inferlet (internal use by the language component wrapper)."""
    global _return_value
    _return_value = value


def get_return_value() -> str | None:
    """Get the return value for the inferlet (internal use by the language component wrapper)."""
    return _return_value
