"""
Model and Tokenizer — wrappers for ``pie:core/model``.
"""

from __future__ import annotations

from wit_world.imports import model as _model


class Tokenizer:
    """Wraps the WIT tokenizer resource."""

    __slots__ = ("_handle",)

    def __init__(self, handle: _model.Tokenizer) -> None:
        self._handle = handle

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        return list(self._handle.encode(text))

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._handle.decode(tokens)

    def vocabs(self) -> tuple[list[int], list[bytes]]:
        """Returns (token_ids, token_bytes) for the full vocabulary."""
        return self._handle.vocabs()

    def split_regex(self) -> str:
        """Returns the tokenizer's split regular expression."""
        return self._handle.split_regex()

    def special_tokens(self) -> tuple[list[int], list[bytes]]:
        """Returns (token_ids, token_bytes) for special tokens."""
        return self._handle.special_tokens()


class Model:
    """Wraps the WIT model resource.

    Usage::

        model = Model.load("llama-3.2-3b")
        tokenizer = model.tokenizer()
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: _model.Model) -> None:
        self._handle = handle

    @staticmethod
    def load(name: str) -> Model:
        """Load a model by name. Raises on failure."""
        return Model(_model.Model.load(name))

    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer for this model."""
        return Tokenizer(self._handle.tokenizer())

    def __repr__(self) -> str:
        return f"Model({id(self._handle):#x})"


