"""Driver registry — maps `type = "..."` strings to driver implementations.

Each inference driver (pie_driver_vllm, pie_driver, pie_driver_dummy,
future siblings like pie_driver_sgl) registers a `DriverSpec` here on
package import. `pie/server.py` consults the registry to dispatch a model
to the right worker module without hardcoding driver names.

Driver selection at TOML level:

    [[model]]
    name = "qwen-small"

    [model.driver]
    type = "vllm"           # ← discriminator looked up here

    [model.driver.options]
    # vllm-specific fields, parsed into DriverSpec.config_cls

Each driver's `config_cls` is a typed dataclass whose field names mirror
that driver's native API (e.g., `VllmDriverConfig` mirrors vllm's
`EngineArgs` field names exactly). No translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriverSpec:
    """Everything pie needs to dispatch a model to a driver."""

    name: str
    """Discriminator string. Matches `[model.driver].type` in TOML."""

    config_cls: type
    """Dataclass describing the driver's TOML subsection. Field names mirror
    the driver's native API."""

    worker_module: str
    """Python module path containing `worker_main(...)`. The server imports
    this lazily and passes it to `mp.spawn`."""

    extras: tuple[str, ...] = ()
    """pip extra(s) that install this driver (informational; surfaced in
    'driver not registered' error messages)."""


_REGISTRY: dict[str, DriverSpec] = {}


def register_driver(spec: DriverSpec) -> None:
    """Register a driver. Call from each driver package's `__init__.py`.

    Re-registering the same name overwrites; duplicate registrations are
    not an error so reload-style workflows work cleanly.
    """
    _REGISTRY[spec.name] = spec


def resolve_driver(name: str) -> DriverSpec:
    """Look up a registered driver by name, or raise with a useful hint."""
    if name not in _REGISTRY:
        installed = sorted(_REGISTRY)
        raise ValueError(
            f"Unknown driver type {name!r}. Registered drivers: {installed}. "
            f"If the driver is part of an optional extra, install it via "
            f"`pip install pie-server[<extra>]`."
        )
    return _REGISTRY[name]


def all_drivers() -> list[str]:
    """Names of all currently-registered drivers, sorted."""
    return sorted(_REGISTRY)
