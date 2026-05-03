"""TOML-shape config for `[model.driver.options]` when `type = "cuda_native"`."""

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
    # KV cache page count. M1.4 default = 1024 (32k token slots at page=32).
    # M1.6 will compute this from gpu_mem_utilization, matching the Python
    # native driver.
    max_num_kv_pages: int = 1024
    # Pinned host KV-page count for swap-out. 0 = disabled (admission gate
    # never schedules swap). Set non-zero to opt into long-context support.
    swap_pool_size: int = 0
    weight_dtype: str = "bfloat16"

    # Runtime quantization mode. Empty (default) → no quantization;
    # `"fp8"` → per-tensor symmetric FP8_E4M3 on every llama-like
    # projection weight, applied immediately after load. Currently only
    # wired for model_type=qwen3 (other archs follow once their forward
    # paths migrate to ops::gemm_act_x_w). M3 will add `"int4"` for
    # offline GPTQ/AWQ checkpoints.
    runtime_quant: str = ""

    # ── handshake ─────────────────────────────────────────────────────────
    # Seconds to wait for the binary's `READY` line on stdout before giving up.
    # Sized for the largest expected load: Qwen3-30B-A3B-Base ships 18,867
    # per-expert tensors and the MoE fuse + load takes ~240s on a single H200
    # (mmap pass plus 18,432 device-to-device cudaMemcpyAsync into fused 3-D
    # tensors). Smaller models still READY within seconds; this is just an
    # upper bound that prevents premature kills.
    ready_timeout_s: float = 600.0

    # Seconds to wait for graceful shutdown after SIGTERM before SIGKILL.
    shutdown_timeout_s: float = 5.0
