"""
Session — ``pie:inferlet/session``.

Client communication: send/receive text and files, and send pictures and
sound encoded on the host (``frames``).
"""

from __future__ import annotations

import json as _json
from typing import Any

import wit_world.imports.session as _session
from componentize_py_types import Err as _WitErr

from .frames import AudioFormat, Frames, FramesError, ImageFormat, Pcm


def send(message: Any) -> None:
    """Send a message to the client.

    Strings are sent verbatim. Anything else is JSON-serialized via
    ``json.dumps`` (dicts, lists, numbers, bools, None). Objects with a
    ``model_dump_json`` method (e.g. future WASM-compatible
    pydantic-shaped classes) are handed off to it instead.

    ::

        session.send("plain text")
        session.send({"event": "tick", "n": 3})       # dict → JSON
        session.send([1, 2, 3])                       # list → JSON
    """
    if isinstance(message, str):
        _session.send(message)
        return
    if hasattr(message, "model_dump_json") and callable(message.model_dump_json):
        _session.send(message.model_dump_json())
        return
    _session.send(_json.dumps(message, default=str))


async def receive() -> str:
    """Receive a text message from the client."""
    result = await _session.receive()
    if result is None:
        raise RuntimeError("Session receive failed")
    return result


def send_file(data: bytes, name: str | None = None) -> None:
    """Send binary file data to the client, under a suggested file ``name``
    if given (a name, not a path: the client keeps its last part)."""
    if name is None:
        _session.send_file(data)
    else:
        _session.send_file_as(data, name)


def send_frames(
    frames: Frames, format: ImageFormat = ImageFormat.PNG, name: str = "picture.png"
) -> None:
    """Send pictures to the client, encoded on the host: the bytes never pass
    through the inferlet's memory."""
    try:
        _session.send_frames(frames.handle, format, name)
    except _WitErr as e:
        raise FramesError(f"send_frames: {e.value}") from None


def send_pcm(pcm: Pcm, format: AudioFormat = AudioFormat.WAV, name: str = "audio.wav") -> None:
    """Send sound to the client, encoded on the host."""
    try:
        _session.send_pcm(pcm.handle, format, name)
    except _WitErr as e:
        raise FramesError(f"send_pcm: {e.value}") from None


async def receive_file() -> bytes:
    """Receive binary file data from the client."""
    result = await _session.receive_file()
    if result is None:
        raise RuntimeError("Session receive_file failed")
    return result


def file(name: str) -> bytes | None:
    """The file the client attached to this run under `name`
    (`pie run --file name=path`), or None."""
    return _session.file(name)
