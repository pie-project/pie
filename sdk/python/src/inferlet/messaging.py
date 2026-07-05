"""
Messaging — ``pie:core/messaging``.

Push/pull and broadcast/subscribe for inter-inferlet communication.
"""

from __future__ import annotations

from wit_world.imports import messaging as _msg


def push(topic: str, message: str) -> None:
    """Push a message to a topic (point-to-point)."""
    _msg.push(topic, message)


async def pull(topic: str) -> str:
    """Pull the next message from a topic."""
    return await _msg.pull(topic)


def broadcast(topic: str, message: str) -> None:
    """Broadcast a message to all subscribers of a topic."""
    _msg.broadcast(topic, message)


def subscribe(topic: str) -> Subscription:
    """Subscribe to a topic. Returns an async iterable subscription."""
    return Subscription(_msg.subscribe(topic))


class Subscription:
    """Async iterable subscription to a messaging topic, backed by a
    component-model ``stream<string>``.

    Usage::

        with messaging.subscribe("events") as sub:
            async for message in sub:
                print(message)

    Closing the subscription (``unsubscribe`` / leaving the ``with`` block)
    drops the stream's readable end, which the host observes as an
    unsubscribe.
    """

    __slots__ = ("_reader", "_buffer", "_closed")

    # How many items to request per stream read (the host may return fewer).
    _READ_CHUNK = 16

    def __init__(self, reader: _msg.StreamReader[str]) -> None:
        self._reader = reader
        self._buffer: list[str] = []
        self._closed = False

    async def next(self) -> str | None:
        """Get the next message, or ``None`` once the stream is closed."""
        while not self._buffer:
            if self._closed or self._reader.writer_dropped:
                return None
            items = await self._reader.read(self._READ_CHUNK)
            if not items:
                # The writable end was dropped — no more messages will arrive.
                return None
            self._buffer.extend(items)
        return self._buffer.pop(0)

    def unsubscribe(self) -> None:
        """Unsubscribe from the topic (drops the stream's readable end)."""
        if not self._closed:
            self._closed = True
            self._reader.__exit__(None, None, None)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *args) -> None:
        self.unsubscribe()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        result = await self.next()
        if result is None:
            raise StopAsyncIteration
        return result
