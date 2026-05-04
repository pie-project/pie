"""Standalone launcher entry point for the `vllm` driver.

The lifecycle lives in `pie_driver_dev._launcher` — vllm depends on
`pie-driver-dev` for the shared worker scaffolding anyway, so importing
the launcher helper from there keeps the dependency graph one-deep.
"""

from pie_driver_dev._launcher import launch

from . import worker
from .config import VllmDriverConfig


if __name__ == "__main__":
    raise SystemExit(launch(
        prog="pie_driver_vllm",
        config_cls=VllmDriverConfig,
        worker=worker,
    ))
