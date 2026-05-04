"""
Messaging — ``pie:core/messaging``.

Push/pull and broadcast/subscribe for inter-inferlet communication.
"""

from __future__ import annotations

import asyncio

from wit_world.imports import messaging as _msg

from ._async import await_future


def push(topic: str, message: str) -> None:
    """Push a message to a topic (point-to-point)."""
    _msg.push(topic, message)


async def pull(topic: str) -> str:
    """Pull the next message from a topic."""
    future = _msg.pull(topic)
    return await await_future(future, f"Pull from '{topic}' failed")


def broadcast(topic: str, message: str) -> None:
    """Broadcast a message to all subscribers of a topic."""
    _msg.broadcast(topic, message)


def subscribe(topic: str) -> Subscription:
    """Subscribe to a topic. Returns an async iterable subscription."""
    return Subscription(_msg.subscribe(topic))


class Subscription:
    """Async iterable subscription to a messaging topic.

    Usage::

        with messaging.subscribe("events") as sub:
            async for message in sub:
                print(message)
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: _msg.Subscription) -> None:
        self._handle = handle

    async def next(self) -> str | None:
        """Get the next message, or ``None`` if no more messages."""
        pollable = self._handle.pollable()
        loop = asyncio.get_event_loop()
        waker: asyncio.Future[None] = loop.create_future()
        loop.wakers.append((pollable, waker))  # type: ignore[attr-defined]
        await waker
        return self._handle.get()

    def unsubscribe(self) -> None:
        """Unsubscribe from the topic."""
        self._handle.unsubscribe()

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
