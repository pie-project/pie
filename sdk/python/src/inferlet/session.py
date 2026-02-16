"""
Session — ``pie:core/session``.

Client communication: send/receive text and files.
"""

from __future__ import annotations

import wit_world.imports.session as _session

from ._async import await_future


def send(message: str) -> None:
    """Send a text message to the client."""
    _session.send(message)


async def receive() -> str:
    """Receive a text message from the client."""
    future = _session.receive()
    return await await_future(future, "Session receive failed")


def send_file(data: bytes) -> None:
    """Send binary file data to the client."""
    _session.send_file(data)


async def receive_file() -> bytes:
    """Receive binary file data from the client."""
    future = _session.receive_file()
    return await await_future(future, "Session receive_file failed")
