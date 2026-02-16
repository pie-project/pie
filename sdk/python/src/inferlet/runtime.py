"""
Runtime — ``pie:core/runtime``.

Runtime introspection: version, instance ID, available models, spawning.
"""

from __future__ import annotations

from wit_world.imports import runtime as _runtime

from ._async import await_future


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


async def spawn(program: str, args: list[str] | None = None) -> str:
    """Spawn a child inferlet.

    Args:
        program: Package name (e.g. ``"my-org:my-inferlet@1.0.0"``).
        args: Optional list of string arguments to pass to the child.

    Returns:
        The spawned inferlet's result string.
    """
    future = _runtime.spawn(program, args or [])
    return await await_future(future, f"Spawn '{program}' failed")
