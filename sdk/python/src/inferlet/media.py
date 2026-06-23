"""
Multimodal media wrappers over ``pie:core/media``.
"""

from __future__ import annotations

from typing import Iterable

from wit_world.imports import media as _media

from .model import Model


def _as_bytes(data: bytes | bytearray | memoryview | Iterable[int]) -> bytes:
    if isinstance(data, bytes):
        return data
    return bytes(data)


class Image:
    """Host-side preprocessed still image."""

    __slots__ = ("_handle",)

    def __init__(self, handle: _media.Image) -> None:
        self._handle = handle

    @classmethod
    def from_bytes(
        cls,
        model: Model,
        data: bytes | bytearray | memoryview | Iterable[int],
    ) -> Image:
        """Decode image bytes for ``model``."""
        return cls(_media.Image.from_bytes(model._handle, _as_bytes(data)))

    def token_count(self) -> int:
        """Soft-token rows / KV slots occupied by this span."""
        return self._handle.token_count()

    def position_span(self) -> int:
        """Sequence-position span consumed by this image."""
        return self._handle.position_span()

    def grid(self) -> tuple[int, int, int]:
        """``(t, h, w)`` merged-token grid."""
        return self._handle.grid()

    def prefix_tokens(self) -> list[int]:
        """Model-specific delimiter tokens before the image span."""
        return list(self._handle.prefix_tokens())

    def suffix_tokens(self) -> list[int]:
        """Model-specific delimiter tokens after the image span."""
        return list(self._handle.suffix_tokens())


class Video:
    """Host-side decoded and sampled video clip."""

    __slots__ = ("_handle",)

    def __init__(self, handle: _media.Video) -> None:
        self._handle = handle

    @classmethod
    def from_bytes(
        cls,
        model: Model,
        data: bytes | bytearray | memoryview | Iterable[int],
        max_frames: int,
    ) -> Video:
        """Decode and uniformly sample up to ``max_frames`` frames."""
        return cls(_media.Video.from_bytes(model._handle, _as_bytes(data), max_frames))

    def frame_count(self) -> int:
        """Number of sampled frames."""
        return self._handle.frame_count()

    def frame(self, index: int) -> Image:
        """The ``index``-th sampled frame as an image span."""
        return Image(self._handle.frame(index))

    def timestamp(self, index: int) -> float:
        """Timestamp in seconds for the ``index``-th sampled frame."""
        return self._handle.timestamp(index)


class Audio:
    """Host-side preprocessed audio clip."""

    __slots__ = ("_handle",)

    def __init__(self, handle: _media.Audio) -> None:
        self._handle = handle

    @classmethod
    def from_bytes(
        cls,
        model: Model,
        data: bytes | bytearray | memoryview | Iterable[int],
    ) -> Audio:
        """Decode audio bytes for ``model``."""
        return cls(_media.Audio.from_bytes(model._handle, _as_bytes(data)))

    def token_count(self) -> int:
        """Soft-token rows / KV slots occupied by this clip."""
        return self._handle.token_count()

    def position_span(self) -> int:
        """Sequence-position span consumed by this audio clip."""
        return self._handle.position_span()

    def prefix_tokens(self) -> list[int]:
        """Model-specific delimiter tokens before the audio span."""
        return list(self._handle.prefix_tokens())

    def suffix_tokens(self) -> list[int]:
        """Model-specific delimiter tokens after the audio span."""
        return list(self._handle.suffix_tokens())
