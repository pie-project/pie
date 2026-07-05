"""
Session — ``pie:core/session``.

Client communication: send/receive text and files.
"""

from __future__ import annotations

import json as _json
from typing import Any

import wit_world.imports.session as _session


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


def send_file(data: bytes) -> None:
    """Send binary file data to the client."""
    _session.send_file(data)


async def receive_file() -> bytes:
    """Receive binary file data from the client."""
    result = await _session.receive_file()
    if result is None:
        raise RuntimeError("Session receive_file failed")
    return result
