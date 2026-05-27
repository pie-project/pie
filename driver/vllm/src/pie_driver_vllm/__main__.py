"""Standalone launcher entry point for the `vllm` driver.

The lifecycle lives in `._bridge._launcher` — vllm depends on
`pie-driver-dev` for the shared worker scaffolding anyway, so importing
the launcher helper from there keeps the dependency graph one-deep.
"""

from ._bridge._launcher import launch

from . import worker
from .config import VllmDriverConfig
from .utils import validate_cuda_devices


if __name__ == "__main__":
    raise SystemExit(launch(
        prog="pie_driver_vllm",
        config_cls=VllmDriverConfig,
        worker=worker,
        validate_devices=validate_cuda_devices,
    ))
