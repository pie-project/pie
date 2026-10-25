"""
Model — global accessors for ``pie:inferlet/model``.

The engine serves exactly one model, so these are module-level functions
over that single bound model (no ``model`` resource handle)::

    from inferlet import model

    print(model.name())

The tokenizer surface (encode/decode/vocabs/special_tokens/split_regex)
lives in the sibling :mod:`inferlet.tokenizer` module and is re-exported
here, so ``model.encode``/``model.decode`` read off ``model`` in inferlet
source the way they do in the Rust `inferlet` crate.
"""

from __future__ import annotations

from dataclasses import dataclass

from wit_world.imports import model as _model

AxisRole = _model.AxisRole
ForwardKind = _model.ForwardKind
InputAxis = _model.InputAxis
InputDecl = _model.InputDecl
LaneDecl = _model.LaneDecl
LaneStream = _model.LaneStream
Readout = _model.Readout
ReadoutKind = _model.ReadoutKind
ScheduleKind = _model.ScheduleKind

from .tokenizer import (  # noqa: F401 — re-exported, as the Rust `model` module does
    encode,
    decode,
    vocabs,
    split_regex,
    special_tokens,
    token_bytes,
    tokens_with_prefix,
)


def name() -> str:
    """Name of the bound model."""
    return _model.name()


def architecture() -> str:
    """Model architecture identifier (e.g. ``"gemma4"``, ``"qwen3_6"``)."""
    return _model.architecture()


def default_system_speculation() -> bool:
    """Whether greedy generation should use the system drafter by default."""
    return _model.default_system_speculation()


def mtp_depth() -> int:
    """The draft head's chain depth; 0 for a model with no draft head."""
    return _model.mtp_depth()


def submit_deadline_us() -> int:
    """How long a pipeline may hold a frame's wait-set, in microseconds."""
    return _model.submit_deadline_us()


@dataclass(frozen=True)
class BlockDrafter:
    """What a guest needs to seed a BLOCK drafter's draft pass."""

    rows: int
    mask_token: int
    proposals_from: int


def draft_block() -> BlockDrafter | None:
    """The bound model's block drafter, if it carries one."""
    d = _model.draft_block()
    return BlockDrafter(d.rows, d.mask_token, d.proposals_from) if d is not None else None


@dataclass(frozen=True)
class CanvasShape:
    """The canvas a diffusion model denoises: tokens per block and the
    trunk's hidden width."""

    length: int
    hidden: int


def canvas() -> CanvasShape | None:
    """The diffusion canvas; ``None`` for every other pass kind."""
    c = _model.canvas()
    return CanvasShape(c.length, c.hidden) if c is not None else None


def run_ahead_window() -> int:
    """Fires one lane may have submitted and not yet taken."""
    return _model.run_ahead_window()


def pass_kind() -> _model.ForwardKind:
    """The bound model's forward-pass interface; never derive it from :func:`architecture`."""
    return _model.pass_kind()


def output_vocab_size() -> int:
    """Logits/output dimension. May exceed the tokenizer's vocabulary size."""
    return _model.output_vocab_size()


def kv_page_size() -> int:
    """Tokens per KV page for the bound model/engine."""
    return _model.kv_page_size()


def frame_size() -> int:
    """Waves per frame (k). A submit takes exactly this many ordered slots."""
    return _model.frame_size()


def channel_capacity() -> int:
    """Host-reader channel capacity, in cells, that sustains the engine's
    run-ahead. Read it per run — unlike :func:`frame_size` it is not
    promised static."""
    return _model.channel_capacity()


def max_readout_rows(kind: ReadoutKind) -> int:
    """The most rows one pass may read out through `kind`; the pass itself may be any length."""
    return _model.max_readout_rows(kind)


def rs_state_size() -> int:
    """Bytes in one folded recurrent-state object. 0 for pure attention."""
    return _model.rs_state_size()


# -- generative facts ----------------------------------------------------------


def readings() -> list:
    """Every reading the bound model declares (`model.ReadingFact`), in
    index order; empty for a text row (one implicit reading)."""
    return _model.readings()


def reading(name: str):
    """The declared reading named `name`, or ``None``."""
    return next((r for r in _model.readings() if r.name == name), None)


def lane(switches: list[tuple[str, str]]) -> LaneDecl:
    """The lane a pass of these switches is (`model.LaneDecl`): the inputs it
    binds (exactly these, by ``ForwardPass.input``), whether it binds
    ``state`` and ``embed``, and what it reads out. Switches are
    ``(name, value)`` pairs — ``("reading", r)``, ``("stream", s)``, and on a
    diffusion model ``("canvas", "encode" | "denoise")``; one left unset takes
    its default."""
    return _model.lane(list(switches))


def latent():
    """The latent space a denoiser works in (`model.LatentSpace`), or ``None``."""
    return _model.latent()


def schedule():
    """The schedule the denoiser was trained under (`model.ScheduleFact`), or ``None``."""
    return _model.schedule()


def max_latent_rows() -> int:
    """The most rows one lane may carry: the guest sizes its row inputs under this."""
    return _model.max_latent_rows()
