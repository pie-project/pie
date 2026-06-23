"""
Audio-output wrappers over ``pie:core/audio-out``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from wit_world.imports import audio_out as _audio

if TYPE_CHECKING:
    from .model import Model

Voice = _audio.Voice


class Speech:
    """Generated audio clip."""

    __slots__ = ("_handle",)

    def __init__(self, handle: _audio.Speech) -> None:
        self._handle = handle

    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._handle.sample_rate()

    def channels(self) -> int:
        """Channel count."""
        return self._handle.channels()

    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        return self._handle.duration_ms()

    def pcm(self) -> list[float]:
        """Decoded PCM samples in [-1, 1]."""
        return list(self._handle.pcm())

    def to_wav(self) -> bytes:
        """Encode mono f32 PCM as canonical 16-bit PCM WAV."""
        return write_wav(self.pcm(), self.sample_rate())


class SpeechBuilder:
    """Builder for model-agnostic speech synthesis."""

    __slots__ = ("_model", "_text", "_voice", "_max_duration_ms")

    def __init__(self, model: Model, text: str) -> None:
        self._model = model
        self._text = text
        self._voice: Voice = _audio.Voice_Speaker(0)
        self._max_duration_ms: int | None = None

    def voice(self, voice: Voice) -> SpeechBuilder:
        """Set the voice directly. Defaults to speaker 0."""
        self._voice = voice
        return self

    def speaker(self, id: int) -> SpeechBuilder:
        """Convenience for ``voice(Voice_Speaker(id))``."""
        self._voice = _audio.Voice_Speaker(id)
        return self

    def named(self, name: str) -> SpeechBuilder:
        """Convenience for named voices on models that support them."""
        self._voice = _audio.Voice_Named(name)
        return self

    def max_duration_ms(self, ms: int) -> SpeechBuilder:
        """Cap generated audio length in milliseconds."""
        self._max_duration_ms = max(0, int(ms))
        return self

    def max_duration_seconds(self, seconds: float) -> SpeechBuilder:
        """Cap generated audio length in seconds."""
        return self.max_duration_ms(int(seconds * 1000))

    async def generate(self) -> Speech:
        """Synthesize speech on the bound model."""
        req = _audio.SpeechRequest(
            text=self._text,
            voice=self._voice,
            max_duration_ms=self._max_duration_ms,
        )
        return Speech(_audio.Speech.generate(self._model._handle, req))


def write_wav(pcm: Iterable[float], sample_rate: int) -> bytes:
    """Write mono f32 PCM (``[-1, 1]``) as canonical 16-bit PCM WAV."""
    samples = list(pcm)
    data_bytes = len(samples) * 2
    out = bytearray()
    out.extend(b"RIFF")
    out.extend((36 + data_bytes).to_bytes(4, "little"))
    out.extend(b"WAVE")
    out.extend(b"fmt ")
    out.extend((16).to_bytes(4, "little"))
    out.extend((1).to_bytes(2, "little"))
    out.extend((1).to_bytes(2, "little"))
    out.extend(int(sample_rate).to_bytes(4, "little"))
    out.extend(int(sample_rate * 2).to_bytes(4, "little"))
    out.extend((2).to_bytes(2, "little"))
    out.extend((16).to_bytes(2, "little"))
    out.extend(b"data")
    out.extend(data_bytes.to_bytes(4, "little"))
    for sample in samples:
        clamped = max(-1.0, min(1.0, float(sample)))
        value = int(round(clamped * 32767))
        out.extend(value.to_bytes(2, "little", signed=True))
    return bytes(out)
