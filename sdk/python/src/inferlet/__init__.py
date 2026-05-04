"""
Pie Inferlet SDK — Python bindings for the Pie runtime.

Quickstart::

    from inferlet import Context, Model, Sampler, runtime

    model = Model.load(runtime.models()[0])
    ctx = Context(model)

    ctx.system("You are helpful.").user("What is 2 + 2?")
    text = await ctx.generate(Sampler.argmax(), max_tokens=64).collect_text()

Three-layer surface:

* :class:`Context` — KV cache + chat fillers + ``forward()`` / ``generate()``.
* :class:`Forward` (``ctx.forward()``) — single forward-pass primitive
  with auto page management. For prefill / scoring / custom loops.
* :class:`Generator` (``ctx.generate()``) — multi-step state machine
  over Forward. Iterate with ``async for step in gen``, or use
  ``await gen.collect_text() / .collect_tokens() / .collect_json()``.

Streaming decoders for chat / reasoning / tools live as independent
modules — compose by hand, no implicit suppression::

    from inferlet import chat, reasoning, tools

    chat_dec = chat.Decoder(model)
    async for step in gen:
        out = await step.execute()
        match chat_dec.feed(out.tokens):
            case chat.Event.Delta(text=t): print(t, end="")
            case chat.Event.Done(text=full): break
            case _: pass

Constraint specs (:class:`JsonSchema`, :class:`AnyJson`, :class:`Regex`,
:class:`Ebnf`) implement the :class:`Schema` protocol — duck-typed, so
your own grammar source class plugs in by adding a ``build_constraint``
method. No inheritance required.
"""

from __future__ import annotations

# --- Core ---
from .model import Model, Tokenizer
from .sample import (
    Distribution,
    Entropy,
    Logits,
    Logprob,
    Logprobs,
    Sampler,
)
from .forward import Forward, Output, ProbeHandle, SampleHandle
from .generation import GenStep, Generator
from .context import Context

# --- Decoders + tools (sub-modules; users import as `inferlet.chat`, etc.) ---
from . import chat
from . import reasoning
from . import tools

# --- Constraint surface ---
from .grammar import (
    AnyJson,
    Constraint,
    Ebnf,
    Grammar,
    GrammarConstraint,
    JsonSchema,
    Matcher,
    Regex,
    Schema,
)

# --- Speculation ---
from .spec import Speculator

# --- Runtime / IO ---
from . import runtime
from . import scheduling
from . import messaging
from . import session
from . import mcp
from . import zo

# --- Adapter ---
from .adapter import Adapter


__all__ = [
    # Core
    "Context",
    "Model",
    "Tokenizer",
    "Adapter",
    # Forward primitive
    "Forward",
    "Output",
    "SampleHandle",
    "ProbeHandle",
    # Generator
    "Generator",
    "GenStep",
    # Sampler / Probe
    "Sampler",
    "Logits",
    "Distribution",
    "Logprob",
    "Logprobs",
    "Entropy",
    # Decoders + tools
    "chat",
    "reasoning",
    "tools",
    # Constraints
    "Schema",
    "JsonSchema",
    "AnyJson",
    "Regex",
    "Ebnf",
    "Constraint",
    "GrammarConstraint",
    "Grammar",
    "Matcher",
    # Speculation
    "Speculator",
    # Runtime / IO
    "runtime",
    "scheduling",
    "messaging",
    "session",
    "mcp",
    "zo",
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
