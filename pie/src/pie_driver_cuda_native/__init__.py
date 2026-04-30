"""cuda_native driver — wraps the C++/CUDA binary in `driver/cuda/`.

The actual forward-pass executor is the `pie_driver_cuda` binary built from
`driver/cuda`. This Python module exists so the Pie engine can register and
spawn it through the same `DriverSpec` mechanism it uses for `native`,
`vllm`, etc.
"""

from __future__ import annotations

from pie.drivers import DriverSpec, register_driver

from .config import CudaNativeDriverConfig

register_driver(DriverSpec(
    name="cuda_native",
    config_cls=CudaNativeDriverConfig,
    worker_module="pie_driver_cuda_native.worker",
    extras=("cu128",),
))
