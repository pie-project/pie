"""
Tokenizer — global accessors for ``pie:inferlet/tokenizer``.

These used to be part of ``pie:core/model``. The WIT split moved them to
their own interface: ``model`` now carries identity and memory-shaping
capabilities only. Same functions, new module::

    from inferlet import tokenizer

    tokens = tokenizer.encode("hello")
    text = tokenizer.decode(tokens)
"""

from __future__ import annotations

from wit_world.imports import tokenizer as _tokenizer


def encode(text: str) -> list[int]:
    """Encode text into token IDs."""
    return list(_tokenizer.encode(text))


def decode(tokens: list[int]) -> str:
    """Decode token IDs back to text."""
    return _tokenizer.decode(tokens)


def vocabs() -> tuple[list[int], list[bytes]]:
    """Returns ``(token_ids, token_bytes)`` for the full vocabulary."""
    return _tokenizer.vocabs()


def split_regex() -> str:
    """Returns the tokenizer's split regular expression."""
    return _tokenizer.split_regex()


def special_tokens() -> tuple[list[int], list[bytes]]:
    """Returns ``(token_ids, token_bytes)`` for special tokens."""
    return _tokenizer.special_tokens()
