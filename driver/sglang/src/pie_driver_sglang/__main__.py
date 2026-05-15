"""Standalone launcher entry point for the `sglang` driver.

The lifecycle lives in `._bridge._launcher` — sglang depends on
`pie-driver-dev` for the shared worker scaffolding anyway, so importing
the launcher helper from there keeps the dependency graph one-deep.
"""

from ._bridge._launcher import launch

from . import worker
from .config import SGLangDriverConfig
from .utils import validate_cuda_devices


if __name__ == "__main__":
    raise SystemExit(launch(
        prog="pie_driver_sglang",
        config_cls=SGLangDriverConfig,
        worker=worker,
        validate_devices=validate_cuda_devices,
    ))
