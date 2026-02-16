"""
Pie Inferlet SDK — Python bindings for the Pie runtime.

Usage::

    from inferlet import Model, Context, Sampler, Event, runtime

    model = Model.load(runtime.models()[0])
    ctx = Context(model)

    ctx.system("You are helpful.")
    ctx.user("Hello!")

    async for event in await ctx.generate(Sampler.top_p(), decode=True):
        match event:
            case Event.Text(text=t):
                print(t, end="")
            case Event.Done():
                break
"""

from __future__ import annotations

# --- Core ---
from .model import Model, Tokenizer
from .sampler import Sampler
from .context import Context, TokenStream, EventStream, Event, Decoder

# --- Runtime ---
from . import runtime
from . import messaging
from . import session
from . import mcp
from . import zo

# --- Inference ---
from .forward import ForwardPass
from .adapter import Adapter
from .grammar import Grammar, Matcher

__all__ = [
    # Core
    "Model",
    "Tokenizer",
    "Sampler",
    "Context",
    "TokenStream",
    "EventStream",
    "Event",
    "Decoder",
    # Runtime
    "runtime",
    "messaging",
    "session",
    "mcp",
    "zo",
    # Inference
    "ForwardPass",
    "Adapter",
    "Grammar",
    "Matcher",
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
