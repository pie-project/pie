"""
Pie Inferlet SDK — Python bindings for the Pie runtime.

STATUS: the forward-pass surface is NOT available from Python yet.

The runtime's guest-facing forward-pass surface was replaced: the old
`pie:core/inference` interface, which exposed a fixed host-side sampler
(`Sampler.argmax()`, probes, `Generator`), is gone. In its place the guest
traces a program and ships canonical PTIR container bytes through one of
`pie:inferlet/forward`, `forward-recurrent`, or `forward-hybrid`.

Nothing in this package can produce those bytes. The Rust SDK does it with
`compiler/dsl` (the tracing eDSL) and `compiler/ir` (the container encoder);
neither has a Python counterpart, and the encoder has to agree with the Rust
one byte for byte. Porting it is its own project — see `forward_refactor.md`
section 10.4 and the tracking note in `scripts/check-sdk-interfaces.sh`.

So the modules that were built on the removed interface — `context`,
`forward`, `generation`, `sample`, `grammar`, `tools`, `adapter`, `runtime`,
`zo`, `spec` — have been deleted rather than left importing a surface that no
longer exists. They were not usable; they only looked usable.

What is here is the part of the SDK that survives the move unchanged: the
non-forward interfaces, whose WIT definitions came through the split intact.

    from inferlet import model, session, chat, reasoning

`grammar` and `tools` DO exist as interfaces in the new world, but under
different shapes (`pie:inferlet/grammar`, `pie:inferlet/tools`) than the
deleted modules of those names targeted. They come back with the port, not
before.

The generated `bindings/` tree is current: it is regenerated from
`interface/inferlet/` with

    componentize-py -d interface/inferlet -w inferlet bindings <out>

and every interface in the world above is present there. Only the
hand-written layer is missing.
"""

from __future__ import annotations

from . import chat
from . import model
from . import reasoning
from . import session
from . import tokenizer

__all__ = [
    "chat",
    "model",
    "reasoning",
    "session",
    "tokenizer",
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
