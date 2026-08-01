"""
Model — global accessors for ``pie:inferlet/model``.

The engine serves exactly one model, so these are module-level functions
over that single bound model (no ``model`` resource handle)::

    from inferlet import model

    print(model.name())

The tokenizer surface (encode/decode/vocabs/special_tokens/split_regex)
moved to the sibling :mod:`inferlet.tokenizer` module when the WIT split
separated the two interfaces.
"""

from __future__ import annotations

from wit_world.imports import model as _model


def name() -> str:
    """Name of the bound model."""
    return _model.name()


def architecture() -> str:
    """Model architecture identifier (e.g. ``"gemma4"``, ``"qwen3_6"``)."""
    return _model.architecture()


def default_system_speculation() -> bool:
    """Whether greedy generation should use the system drafter by default."""
    return _model.default_system_speculation()


def is_linear() -> bool:
    """Whether the bound model carries irreversibly-folded recurrent state."""
    return _model.is_linear()


def pass_kind() -> _model.ForwardKind:
    """Which forward-pass interface the bound model requires. Selects the
    binding surface; do not derive it by parsing :func:`architecture`."""
    return _model.pass_kind()


def output_vocab_size() -> int:
    """Logits/output dimension. May exceed the tokenizer's vocabulary size."""
    return _model.output_vocab_size()


def kv_page_size() -> int:
    """Tokens per KV page for the bound model/driver."""
    return _model.kv_page_size()


def frame_size() -> int:
    """Waves per frame (k). A submit takes exactly this many ordered slots."""
    return _model.frame_size()


def channel_capacity() -> int:
    """Host-reader channel capacity, in cells, that sustains the engine's
    run-ahead. Read it per run — unlike :func:`frame_size` it is not
    promised static."""
    return _model.channel_capacity()


def max_embed_length() -> int:
    """Max embed tokens in a single pass — the prefill chunk budget."""
    return _model.max_embed_length()


def rs_state_size() -> int:
    """Bytes in one folded recurrent-state object. 0 for pure attention."""
    return _model.rs_state_size()


def rs_buffer_page_size() -> int:
    """Tokens per buffered RS page. 0 if the model has no recurrent state."""
    return _model.rs_buffer_page_size()


def rs_fold_granularity() -> int:
    """Fold granularity in tokens. 1 (or 0) means unconstrained."""
    return _model.rs_fold_granularity()


def arena_block_size() -> int:
    """Bytes in one unified-arena accounting block."""
    return _model.arena_block_size()
