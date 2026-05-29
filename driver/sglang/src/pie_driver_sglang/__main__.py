"""Standalone launcher entry point for the `sglang` driver.

The lifecycle lives in the vendored `._bridge._launcher`; this entry
point imports and runs it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_venv_bin_on_path() -> None:
    bin_dir = str(Path(sys.prefix) / "bin")
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if bin_dir not in parts:
        os.environ["PATH"] = bin_dir + (os.pathsep + path if path else "")


_ensure_venv_bin_on_path()

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
