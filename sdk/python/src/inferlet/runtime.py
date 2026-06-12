"""
Runtime — ``pie:core/runtime``.

Runtime introspection: version, instance ID, available models.
"""

from __future__ import annotations

from wit_world.imports import runtime as _runtime


def version() -> str:
    """Get the runtime version string."""
    return _runtime.version()


def instance_id() -> str:
    """Get the unique instance identifier."""
    return _runtime.instance_id()


def username() -> str:
    """Get the current user's name."""
    return _runtime.username()


def models() -> list[str]:
    """Get names of all available models."""
    return list(_runtime.models())


def metadata_put(namespace: str, key: str, value: bytes) -> None:
    """Store or overwrite engine-lifetime metadata for this caller."""
    _runtime.metadata_put(namespace, key, value)


def metadata_get(namespace: str, key: str) -> bytes | None:
    """Retrieve engine-lifetime metadata for this caller, if present."""
    return _runtime.metadata_get(namespace, key)


def metadata_delete(namespace: str, key: str) -> bool:
    """Delete engine-lifetime metadata for this caller."""
    return _runtime.metadata_delete(namespace, key)
