"""TOML-shape config for `[model.X.driver.portable]`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortableDriverConfig:
    """Knobs the C++ binary needs at boot.

    Field names mirror `NativeDriverConfig` / `CudaNativeDriverConfig`
    where possible so swapping `type = "portable"` is a one-line change.
    """

    # ── binary discovery ──────────────────────────────────────────────────
    # Override path to the `pie_driver_portable` executable. Empty → resolve via
    # $PIE_PORTABLE_DRIVER_BIN, then `which pie_driver_portable`, then a relative
    # search next to the source tree (`<repo>/driver/portable/build/bin/`).
    binary_path: str = ""

    # ── KV cache / batching budgets ────────────────────────────────────────
    # The runtime owns page allocation; we report `total_pages` and
    # `kv_page_size` in the READY handshake. `max_num_kv_pages` is the
    # global pool size — KV memory scales linearly with it.
    kv_page_size: int = 32
    max_num_kv_pages: int = 1024
    max_batch_tokens: int = 10240
    max_batch_size: int = 512
    # Host-side swap pool capacity (M7). 0 = no swap; runtime sees
    # swap_pool_size=0 in capabilities and won't try to swap. Set > 0 to
    # let the runtime offload pages under memory pressure.
    cpu_pages: int = 0

    # ── runtime knobs ──────────────────────────────────────────────────────
    # Informational max position the binary will accept. Real KV capacity
    # is `max_num_kv_pages * kv_page_size` slots flexibly shared across
    # all in-flight contexts.
    n_ctx: int = 4096
    # 0 = CPU only, -1 = offload all layers to a GPU backend, N = first N.
    # Backend is CPU-only in v1; this knob takes effect once GGML_CUDA /
    # GGML_METAL backends are wired (M11).
    n_gpu_layers: int = 0

    # ── handshake ─────────────────────────────────────────────────────────
    # Seconds to wait for the binary's `READY` line on stdout before giving up.
    ready_timeout_s: float = 120.0

    # Seconds to wait for graceful shutdown after SIGTERM before SIGKILL.
    shutdown_timeout_s: float = 5.0
