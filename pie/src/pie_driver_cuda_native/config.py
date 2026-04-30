"""TOML-shape config for `[model.X.driver.cuda_native]`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CudaNativeDriverConfig:
    """Knobs the C++ binary needs at boot.

    Field names mirror the equivalent fields on `NativeDriverConfig` where
    possible so swapping `type = "native"` ↔ `"cuda_native"` is a one-line
    change.
    """

    # ── binary discovery ──────────────────────────────────────────────────
    # Override path to the `pie_driver_cuda` executable. Empty → resolve via
    # $PIE_CUDA_DRIVER_BIN, then `which pie_driver_cuda`, then a relative
    # search next to the source tree (`<repo>/driver/cuda/build/bin/`).
    binary_path: str = ""

    # ── forward-pass budgets (forwarded to the binary's TOML config) ──────
    gpu_mem_utilization: float = 0.85
    kv_page_size: int = 32
    max_batch_tokens: int = 10240
    max_batch_size: int = 512
    weight_dtype: str = "bfloat16"

    # ── handshake ─────────────────────────────────────────────────────────
    # Seconds to wait for the binary's `READY` line on stdout before giving up.
    ready_timeout_s: float = 120.0

    # Seconds to wait for graceful shutdown after SIGTERM before SIGKILL.
    shutdown_timeout_s: float = 5.0
