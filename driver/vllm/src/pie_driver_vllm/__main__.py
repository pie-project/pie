"""Standalone launcher entry point for the `vllm` driver.

The lifecycle lives in the vendored `._bridge._launcher`; this entry
point imports and runs it.
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
