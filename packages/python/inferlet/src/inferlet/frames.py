"""
Frames — ``pie:inferlet/frames``.

Pictures, clips and sound as host-side handles, encoded on the host: a VAE's
pixels go out as a PNG, a codec's samples as a WAV, without the bytes
passing through the inferlet's memory. ``session.send_frames`` and
``session.send_pcm`` send them::

    from inferlet import frames, session

    pixels = Channel([side * side, 3], dtype.f32)
    decode.epilogue(lambda: pixels.put(intrinsics.output_rows(0, "pixels", side * side)))
    pipe.submit(decode)
    picture = frames.Frames.from_channel(pixels, side, side)
    session.send_frames(picture, frames.ImageFormat.PNG, "picture.png")

    session.send_pcm(frames.Pcm.from_f32(samples, 24000), frames.AudioFormat.WAV, "speech.wav")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from componentize_py_types import Err as _WitErr
from wit_world.imports import frames as _frames

if TYPE_CHECKING:
    from .eta.bridge import Channel

ImageFormat = _frames.ImageFormat
AudioFormat = _frames.AudioFormat


class FramesError(Exception):
    """Pixels or samples the host refused: a length that does not match the
    stated shape, bytes that decode to nothing, parts that do not join."""


def _call(what: str, f, *args):
    try:
        return f(*args)
    except _WitErr as e:
        raise FramesError(f"{what}: {e.value}") from None


class Frames:
    """``count`` frames of ``width`` x ``height`` RGB8, held by the host. A
    still is a handle of one frame."""

    __slots__ = ("_inner",)

    def __init__(self, inner) -> None:
        self._inner = inner

    @property
    def handle(self):
        """The WIT resource, for the raw ``session.send_frames``."""
        return self._inner

    @classmethod
    def from_channel(
        cls, ch: "Channel", width: int, height: int, count: int = 1, fps: float = 0.0
    ) -> "Frames":
        """Take ``ch``'s committed cell as pictures: ``count * height * width * 3``
        f32 in the model's range, [-1, 1], one row a voxel in ``(t, h, w)``
        order, mapped to RGB8 by ``(x + 1) / 2`` and clamped. The cell is
        consumed, like ``take``.

        Take it once the pass that fills it has landed (await a value that
        pass makes first): a cell taken before then can read empty."""
        return cls(
            _call("from_channel", _frames.Frames.from_channel, ch._wit(), width, height, count, fps)
        )

    @classmethod
    def from_rgb8(
        cls, data: bytes, width: int, height: int, count: int = 1, fps: float = 0.0
    ) -> "Frames":
        """Pictures from interleaved RGB8 bytes, ``count * height * width * 3`` of them."""
        return cls(
            _call("from_rgb8", _frames.Frames.from_rgb8, bytes(data), width, height, count, fps)
        )

    @classmethod
    def decode(cls, data: bytes) -> "Frames":
        """An encoded still (PNG, JPEG, GIF or WebP), its format read from its bytes."""
        return cls(_call("decode", _frames.Frames.decode, bytes(data)))

    @classmethod
    def concat(cls, parts: Sequence["Frames"]) -> "Frames":
        """``parts`` joined in order; they share width, height and fps."""
        return cls(_call("concat", _frames.Frames.concat, [p._inner for p in parts]))

    def to_channel(self, ch: "Channel") -> None:
        """Put the pictures on ``ch`` as f32 in [-1, 1], the shape ``from_channel`` takes."""
        _call("to_channel", self._inner.to_channel, ch._wit())

    def encode(self, format: ImageFormat) -> bytes:
        """The pictures encoded, in the inferlet's memory; ``session.send_frames``
        sends them without that copy."""
        return bytes(_call("encode", self._inner.encode, format))

    @property
    def width(self) -> int:
        return self._inner.width()

    @property
    def height(self) -> int:
        return self._inner.height()

    @property
    def count(self) -> int:
        return self._inner.count()

    @property
    def fps(self) -> float:
        """Frames a second; a still reads 0."""
        return self._inner.fps()


class Pcm:
    """Sound held by the host: interleaved samples at ``rate``."""

    __slots__ = ("_inner",)

    def __init__(self, inner) -> None:
        self._inner = inner

    @property
    def handle(self):
        """The WIT resource, for the raw ``session.send_pcm``."""
        return self._inner

    @classmethod
    def from_f32(cls, samples: Sequence[float], rate: int, channels: int = 1) -> "Pcm":
        """Samples in [-1, 1], ``channels`` interleaved."""
        return cls(_call("from_f32", _frames.Pcm.from_f32, list(samples), rate, channels))

    def encode(self, format: AudioFormat) -> bytes:
        return bytes(_call("encode", self._inner.encode, format))

    @property
    def rate(self) -> int:
        return self._inner.rate()

    @property
    def channels(self) -> int:
        return self._inner.channels()
