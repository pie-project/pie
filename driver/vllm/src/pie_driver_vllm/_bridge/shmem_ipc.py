"""Shmem ring server — thin alias for ``pie_bridge.ShmemServer``.

The Rust crate ``pie-bridge`` provides the canonical POSIX shmem ring
server (mmap, slot polling, busy-spin) via its PyO3 bindings. This
shim preserves the legacy ``ShmemServer`` import path inside the
per-driver ``_bridge`` subpackage; new code should import from
``pie_bridge`` directly.
"""

from pie_bridge import ShmemServer

__all__ = ["ShmemServer"]
