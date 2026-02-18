"""
Adapter resource wrapper for ``pie:core/adapter``.

Supports loading, saving, cloning, and locking model adapters (e.g. LoRA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wit_world.imports import adapter as _adapter

from ._async import await_future

if TYPE_CHECKING:
    from .model import Model


class Adapter:
    """Wraps the WIT adapter resource.

    Usage::

        adapter = Adapter.create(model, "my-adapter")
        adapter.load("/path/to/weights")
        adapter.save("/path/to/output")
    """

    __slots__ = ("_handle", "_locked")

    def __init__(self, handle: _adapter.Adapter) -> None:
        self._handle = handle
        self._locked = False

    @staticmethod
    def create(model: Model, name: str) -> Adapter:
        """Create a new adapter for a model."""
        return Adapter(_adapter.Adapter.create(model._handle, name))

    @staticmethod
    def open(model: Model, name: str) -> Adapter | None:
        """Open an existing adapter by name."""
        raw = _adapter.Adapter.open(model._handle, name)
        if raw is None:
            return None
        return Adapter(raw)

    def fork(self, new_name: str) -> Adapter:
        """Fork this adapter with a new name."""
        return Adapter(self._handle.fork(new_name))

    async def acquire_lock(self) -> bool:
        """Acquire an exclusive lock on this adapter."""
        future = self._handle.acquire_lock()
        result = await await_future(future, "Adapter lock failed")
        if result:
            self._locked = True
        return result

    def release_lock(self) -> None:
        """Release the lock on this adapter."""
        self._handle.release_lock()
        self._locked = False

    def load(self, path: str) -> None:
        """Load adapter weights from a file path."""
        self._handle.load(path)

    def save(self, path: str) -> None:
        """Save adapter weights to a file path."""
        self._handle.save(path)

    def __enter__(self) -> Adapter:
        return self

    def __exit__(self, *args) -> None:
        if self._locked:
            self.release_lock()

    def __repr__(self) -> str:
        return f"Adapter({id(self._handle):#x})"
