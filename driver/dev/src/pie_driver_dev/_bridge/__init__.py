"""Per-driver copy of bridge_py's Python scaffolding.

These files (worker loop, launcher, batching, config, telemetry, etc.)
were vendored from the legacy ``._bridge`` Python package as
part of the bridge consolidation (see ``BRIDGE.md`` at the workspace
root, Phase 3). Per the architectural decision: shared types live in
the compiled ``pie_bridge`` wheel; driver-specific Python is copied
per-driver to avoid the runtime coupling that ``._bridge``
imposed.

Imports within this subpackage use relative paths (``from .X``);
wire/IPC primitives come from the ``pie_bridge`` wheel.
"""
