"""
Multimodal input — ``pie:inferlet/media``.

The inferlet hands the host raw encoded bytes (:meth:`Image.from_bytes` for
PNG/JPEG, :meth:`Video.from_bytes` for animated GIF, :meth:`Audio.from_bytes`
for WAV) and the host decodes + preprocesses per the bound model. A span
enters the sequence as the token run its handle answers, and the handle
crosses again beside the tokens (``ForwardPass.media([img])``) to carry the
payload::

    img = media.Image.from_bytes(png)
    toks = model.encode("Describe: ") + img.tokens() + model.encode(" briefly.")
    fwd.embed(tokens_ch, indptr_ch)
    fwd.media([img])
"""

from __future__ import annotations

from dataclasses import dataclass

from componentize_py_types import Err as _WitErr
from wit_world.imports import media as _media


class MediaError(Exception):
    """Bytes the host could not decode as the requested medium."""


@dataclass(frozen=True)
class MergedGrid:
    t: int
    h: int
    w: int


class _Span:
    __slots__ = ("_inner",)

    def __init__(self, inner) -> None:
        self._inner = inner

    @property
    def handle(self):
        """The WIT resource, for ``ForwardPass.media``."""
        return self._inner

    def tokens(self) -> list[int]:
        """The placeholder run this span enters the sequence as."""
        return list(self._inner.tokens())

    def digest(self) -> bytes:
        return bytes(self._inner.digest())

    def token_count(self) -> int:
        return self._inner.token_count()

    def position_span(self) -> int:
        return self._inner.position_span()

    def prefix_tokens(self) -> list[int]:
        return list(self._inner.prefix_tokens())

    def suffix_tokens(self) -> list[int]:
        return list(self._inner.suffix_tokens())


class Image(_Span):
    @classmethod
    def from_bytes(cls, data: bytes) -> "Image":
        try:
            return cls(_media.Image.from_bytes(bytes(data)))
        except _WitErr as e:
            raise MediaError(f"image: {e.value}") from None

    def grid(self) -> MergedGrid:
        g = self._inner.grid()
        return MergedGrid(g.t, g.h, g.w)


class Audio(_Span):
    @classmethod
    def from_bytes(cls, data: bytes) -> "Audio":
        try:
            return cls(_media.Audio.from_bytes(bytes(data)))
        except _WitErr as e:
            raise MediaError(f"audio: {e.value}") from None


class Video:
    __slots__ = ("_inner",)

    def __init__(self, inner) -> None:
        self._inner = inner

    @classmethod
    def from_bytes(cls, data: bytes, max_frames: int) -> "Video":
        try:
            return cls(_media.Video.from_bytes(bytes(data), max_frames))
        except _WitErr as e:
            raise MediaError(f"video: {e.value}") from None

    def frame_count(self) -> int:
        return self._inner.frame_count()

    def frame(self, index: int) -> Image:
        try:
            return Image(self._inner.frame(index))
        except _WitErr as e:
            raise MediaError(f"video frame {index}: {e.value}") from None

    def timestamp(self, index: int) -> float:
        return self._inner.timestamp(index)
