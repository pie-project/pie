"""
Model — global accessors for ``pie:core/model``.

The engine serves exactly one model, so these are module-level functions
over that single bound model (no ``model`` / ``tokenizer`` resource
handles)::

    from inferlet import model

    print(model.name())
    tokens = model.encode("hello")
    text = model.decode(tokens)
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


def encode(text: str) -> list[int]:
    """Encode text into token IDs."""
    return list(_model.encode(text))


def decode(tokens: list[int]) -> str:
    """Decode token IDs back to text."""
    return _model.decode(tokens)


def vocabs() -> tuple[list[int], list[bytes]]:
    """Returns ``(token_ids, token_bytes)`` for the full vocabulary."""
    return _model.vocabs()


def split_regex() -> str:
    """Returns the tokenizer's split regular expression."""
    return _model.split_regex()


def special_tokens() -> tuple[list[int], list[bytes]]:
    """Returns ``(token_ids, token_bytes)`` for special tokens."""
    return _model.special_tokens()
