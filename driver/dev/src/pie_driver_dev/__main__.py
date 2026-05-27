"""Standalone launcher entry point for the `dev` driver.

The lifecycle (TOML parsing, mp.spawn, ready-queue drain, handshake
emission, signal handling, watchdog) lives in `._bridge._launcher`
— this file just selects the driver-specific pieces.
"""

from . import worker
from .utils import validate_cuda_devices
from ._bridge._launcher import launch
from .config import NativeDriverConfig


if __name__ == "__main__":
    raise SystemExit(launch(
        prog="pie_driver_dev",
        config_cls=NativeDriverConfig,
        worker=worker,
        validate_devices=validate_cuda_devices,
    ))
