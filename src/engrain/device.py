"""The device layer: the arena, the pool, the batch and the kernels.

`engrain.Engine` wraps this and is what most callers want. This is here for
the ones that do not - a serving integration that manages its own pool, or a
benchmark that needs to time one kernel.

The implementation lives in `engrain._engine`, which is also where the
research tree imports it from. The name here is the public one.
"""

from engrain._engine import (  # noqa: F401
    DeviceBatch,
    DeviceGrammar,
    StackTooDeep,
    WindowTooWide,
    ResidentTables,
)

__all__ = ["DeviceBatch", "DeviceGrammar", "ResidentTables", "StackTooDeep", "WindowTooWide"]
