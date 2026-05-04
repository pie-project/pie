"""Standalone launcher entry point for the `sglang` driver.

The lifecycle lives in `pie_driver_dev._launcher` — sglang depends on
`pie-driver-dev` for the shared worker scaffolding anyway, so importing
the launcher helper from there keeps the dependency graph one-deep.
"""

from pie_driver_dev._launcher import launch

from . import worker
from .config import SGLangDriverConfig


if __name__ == "__main__":
    raise SystemExit(launch(
        prog="pie_driver_sglang",
        config_cls=SGLangDriverConfig,
        worker=worker,
    ))
