"""The layers under `engrain.Engine`, for callers that need them.

Nothing here is needed to decode under a grammar. It is here because a research
harness, a differential check and a serving integration each need a different
depth, and hiding the lower ones behind a private name would only mean everyone
imported `engrain._engine` anyway.

    Compiler              the compiler without the device arena around it
    DeviceGrammar         the pool: admission, eviction, the arena
    DeviceBatch           the batch: the kernels and the graph, directly
    pack_configurations   parse states to the wire format the batch takes

`engrain.Engine` and `engrain.Slots` are these two objects with the decisions
made - which representation to read, when to record the graph, what a slot
means - and are what an integration should use. Reaching past them is a
statement that you want to make those decisions yourself.
"""

from __future__ import annotations

from engrain._engrain import Compiler, pack_configurations
from engrain.device import (
    DeviceBatch,
    DeviceGrammar,
    ResidentTables,
    StackTooDeep,
    WindowTooWide,
)

__all__ = [
    "Compiler",
    "DeviceBatch",
    "DeviceGrammar",
    "ResidentTables",
    "StackTooDeep",
    "WindowTooWide",
    "pack_configurations",
]
