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
