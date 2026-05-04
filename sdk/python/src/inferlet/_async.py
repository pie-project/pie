"""
Internal async utilities for WASI pollable futures.
"""

import asyncio
from typing import Protocol, TypeVar, runtime_checkable

from wit_world.imports.poll import Pollable

T = TypeVar("T")


@runtime_checkable
class WasiFuture(Protocol[T]):
    """Generic interface for WASI async operations."""

    def pollable(self) -> Pollable: ...
    def get(self) -> T | None: ...


async def await_future(future: WasiFuture[T], error_message: str) -> T:
    """Await a WASI future cooperatively.

    Registers the future's pollable with the asyncio event loop and
    yields control until it becomes ready, allowing other coroutines
    to make progress concurrently.
    """
    pollable = future.pollable()
    loop = asyncio.get_event_loop()
    waker: asyncio.Future[None] = loop.create_future()
    loop.wakers.append((pollable, waker))  # type: ignore[attr-defined]
    await waker

    result = future.get()
    if result is None:
        raise RuntimeError(error_message)
    return result
