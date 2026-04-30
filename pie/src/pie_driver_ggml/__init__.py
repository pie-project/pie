"""ggml driver — wraps the C++ binary in `driver/ggml/`.

The actual forward-pass executor is the `pie_driver_ggml` binary built from
`driver/ggml`. This Python module exists so the Pie engine can register and
spawn it through the same `DriverSpec` mechanism it uses for `native`,
`cuda_native`, `vllm`, etc.

The binary loads HuggingFace safetensors directly (no GGUF) and runs on
ggml's CPU backend by default. CUDA / Metal / Vulkan are opt-in CMake flags
on the binary build (M11).
"""

from __future__ import annotations

from pie.drivers import DriverSpec, register_driver

from .config import GgmlDriverConfig

register_driver(DriverSpec(
    name="ggml",
    config_cls=GgmlDriverConfig,
    worker_module="pie_driver_ggml.worker",
    extras=(),  # CPU backend is unconditional; GPU backends are linker-side
))
