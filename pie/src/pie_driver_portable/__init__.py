"""Portable Pie driver — wraps the C++ binary in `driver/portable/`.

The actual forward-pass executor is the `pie_driver_portable` binary built
from `driver/portable/`. This Python module exists so the Pie engine can
register and spawn it through the same `DriverSpec` mechanism it uses for
`native`, `cuda_native`, `vllm`, etc.

The binary is built on raw libggml (vendored via llama.cpp) and supports
multiple backends (CPU / CUDA / Vulkan / Metal / HIP / SYCL) selected at
build time via `GGML_*=ON` CMake flags. It loads HuggingFace safetensors
directly (no GGUF conversion).
"""

from __future__ import annotations

from pie.drivers import DriverSpec, register_driver

from .config import PortableDriverConfig

register_driver(DriverSpec(
    name="portable",
    config_cls=PortableDriverConfig,
    worker_module="pie_driver_portable.worker",
    extras=(),  # CPU backend is unconditional; GPU backends are linker-side
))
