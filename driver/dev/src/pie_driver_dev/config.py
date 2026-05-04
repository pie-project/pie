"""Worker-internal runtime configuration.

Two layers, mirroring the universal/driver-specific split established by the
driver-config redesign:

  * `RuntimeConfig` — universal, every driver consumes it. Identity, devices,
    dtype, telemetry, swap budget, and the engine-computed `max_num_kv_pages`.
    Vllm uses only this.

  * `NativeRuntimeConfig(RuntimeConfig)` — pie native's runtime config. Adds
    fields that only native consumes (KV page size, batch limits, adapter
    pool sizing, weight dtype + quantization, CUDA-graph capture toggle, the
    dummy-mode flag). Native model code reads `runtime_config.X` against this
    subclass with no awareness of the split.

  * `NativeDriverConfig` — TOML-shape `[model.X.driver.native]` block. The
    user-facing config; gets merged into `NativeRuntimeConfig.from_args`.

Vllm's analogue lives in `pie_driver_vllm.config.VllmDriverConfig`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


# Valid weight dtype categories — only consulted by NativeRuntimeConfig.
FLOAT_DTYPES = {"float32", "float16", "bfloat16", "auto"}
QUANT_DTYPES = {"int4", "int8", "float8"}


def _get_pie_home() -> Path:
    if pie_home := os.environ.get("PIE_HOME"):
        return Path(pie_home)
    return Path.home() / ".pie"


def get_program_dir() -> Path:
    return _get_pie_home() / "programs"


def get_adapter_dir() -> Path:
    return _get_pie_home() / "adapters"


# ---------------------------------------------------------------------------
# Universal runtime config (every driver consumes this)
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    """Universal worker-internal config — what every driver needs.

    Driver-specific runtime knobs (gpu_mem_utilization, weight_dtype, batch
    limits, etc.) live on the driver's own subclass / config, not here.
    """

    # Identity
    hf_repo: str
    cache_dir: str
    adapter_path: str

    # Topology
    devices: list[torch.device]
    rank: int
    tensor_parallel_size: int     # 1 = DP only, >1 = TP within a group

    # Universal precision/seed
    activation_dtype: torch.dtype
    random_seed: int

    # Telemetry
    telemetry_enabled: bool
    telemetry_endpoint: str
    telemetry_service_name: str

    # CPU swap budget in bytes. Both drivers respect it (0 = disabled).
    swap_budget_bytes: int = 0

    # Engine-computed at load time. None pre-load; set by the engine.
    max_num_kv_pages: int | None = None

    # NOTE: `kv_page_size` and `max_dist_size` are NativeRuntimeConfig-only.
    # The shared RPC worker (`_handle_fire_batch`) falls back to
    # `engine.capabilities().kv_page_size` for drivers (vllm/sglang) that
    # don't carry them on their config — see pie_driver_dev/worker.py.

    # ---------- properties ----------
    @property
    def device(self) -> torch.device:
        return self.devices[self.rank]

    @property
    def world_size(self) -> int:
        return len(self.devices)

    @property
    def num_groups(self) -> int:
        return max(1, self.world_size // self.tensor_parallel_size)

    @classmethod
    def from_args(
        cls,
        hf_repo: str,
        *,
        device: str | None = None,
        devices: list[str] | None = None,
        rank: int = 0,
        tensor_parallel_size: int = 1,
        activation_dtype: str = "bfloat16",
        random_seed: int = 42,
        telemetry_enabled: bool = False,
        telemetry_endpoint: str = "http://localhost:4317",
        telemetry_service_name: str = "pie",
        cpu_mem_budget_in_gb: int = 0,
        # Subclass-only kwargs are accepted via **extras and ignored here so
        # callers can pass a single merged dict.
        **_extras,
    ) -> "RuntimeConfig":
        return cls(**_resolve_universal_kwargs(
            hf_repo=hf_repo, device=device, devices=devices, rank=rank,
            tensor_parallel_size=tensor_parallel_size,
            activation_dtype=activation_dtype, random_seed=random_seed,
            telemetry_enabled=telemetry_enabled,
            telemetry_endpoint=telemetry_endpoint,
            telemetry_service_name=telemetry_service_name,
            cpu_mem_budget_in_gb=cpu_mem_budget_in_gb,
        ))

    def print(self) -> None:
        print("--- Configuration ---")
        for k, v in asdict(self).items():
            print(f"{k}: {v}")
        print("----------------------")


def _resolve_universal_kwargs(
    *,
    hf_repo: str,
    device: str | None,
    devices: list[str] | None,
    rank: int,
    tensor_parallel_size: int,
    activation_dtype: str,
    random_seed: int,
    telemetry_enabled: bool,
    telemetry_endpoint: str,
    telemetry_service_name: str,
    cpu_mem_budget_in_gb: int,
) -> dict:
    """Convert string-shaped kwargs to typed values for RuntimeConfig.

    Shared helper used by both `RuntimeConfig.from_args` and the
    `NativeRuntimeConfig.from_args` subclass override.
    """
    if devices is not None:
        resolved_devices = [torch.device(d) for d in devices]
    elif device is not None:
        resolved_devices = [torch.device(device)]
    elif torch.cuda.is_available():
        resolved_devices = [torch.device("cuda:0")]
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        resolved_devices = [torch.device("mps")]
    else:
        resolved_devices = [torch.device("cpu")]

    return dict(
        hf_repo=hf_repo,
        cache_dir=str(get_program_dir()),
        adapter_path=str(get_adapter_dir()),
        devices=resolved_devices,
        rank=rank,
        tensor_parallel_size=tensor_parallel_size,
        activation_dtype=getattr(torch, activation_dtype, torch.bfloat16),
        random_seed=random_seed,
        telemetry_enabled=telemetry_enabled,
        telemetry_endpoint=telemetry_endpoint,
        telemetry_service_name=telemetry_service_name,
        swap_budget_bytes=cpu_mem_budget_in_gb * (1 << 30),
    )


# ---------------------------------------------------------------------------
# Native runtime config — what pie's own model code reads
# ---------------------------------------------------------------------------


@dataclass
class NativeRuntimeConfig(RuntimeConfig):
    """Native driver's runtime config — extends `RuntimeConfig` with
    native-only knobs.

    Field names match pie's existing internal vocabulary (kv_page_size,
    gpu_mem_utilization, etc.) so model code under `pie_driver_dev/model/`
    keeps reading `runtime_config.X` unchanged.
    """

    # Memory + KV layout
    gpu_mem_utilization: float = 0.8
    kv_page_size: int = 16

    # Batching limits the native scheduler / kernels enforce
    max_batch_tokens: int = 10240
    max_batch_size: int = 512
    max_dist_size: int = 32
    max_num_embeds: int = 128

    # Adapter pool sizing (CmaesAdapter)
    max_num_adapters: int = 32
    max_adapter_rank: int = 8

    # Weight dtype + quantization stack
    weight_dtype: str = "auto"

    # Set by the dummy driver to skip real weight loading
    dummy_mode: bool = False

    # ---------- native-only properties ----------
    @property
    def needs_quantization(self) -> bool:
        return self.weight_dtype in QUANT_DTYPES

    @property
    def compute_dtype(self) -> torch.dtype:
        if self.weight_dtype == "auto" or self.weight_dtype in QUANT_DTYPES:
            return self.activation_dtype
        return getattr(torch, self.weight_dtype)

    @property
    def quantization(self):
        """Map `weight_dtype` to a torchao quantization config (or None)."""
        import torchao
        match self.weight_dtype:
            case "int4":
                return torchao.quantization.Int4WeightOnlyConfig()
            case "int8":
                return torchao.quantization.Int8WeightOnlyConfig()
            case "float8":
                return torchao.quantization.Float8WeightOnlyConfig()
            case _:
                return None

    @classmethod
    def from_args(
        cls,
        hf_repo: str,
        *,
        # universal
        device: str | None = None,
        devices: list[str] | None = None,
        rank: int = 0,
        tensor_parallel_size: int = 1,
        activation_dtype: str = "bfloat16",
        random_seed: int = 42,
        telemetry_enabled: bool = False,
        telemetry_endpoint: str = "http://localhost:4317",
        telemetry_service_name: str = "pie",
        cpu_mem_budget_in_gb: int = 0,
        # native-specific
        gpu_mem_utilization: float = 0.8,
        kv_page_size: int = 16,
        max_batch_tokens: int = 10240,
        max_batch_size: int = 512,
        max_dist_size: int = 32,
        max_num_embeds: int = 128,
        max_num_adapters: int = 32,
        max_adapter_rank: int = 8,
        weight_dtype: str = "auto",
        dummy_mode: bool = False,
        **_unknown,
    ) -> "NativeRuntimeConfig":
        if weight_dtype not in (FLOAT_DTYPES | QUANT_DTYPES):
            raise ValueError(
                f"Invalid weight_dtype: '{weight_dtype}'. "
                f"Expected one of: {sorted(FLOAT_DTYPES | QUANT_DTYPES)}"
            )

        universal = _resolve_universal_kwargs(
            hf_repo=hf_repo, device=device, devices=devices, rank=rank,
            tensor_parallel_size=tensor_parallel_size,
            activation_dtype=activation_dtype, random_seed=random_seed,
            telemetry_enabled=telemetry_enabled,
            telemetry_endpoint=telemetry_endpoint,
            telemetry_service_name=telemetry_service_name,
            cpu_mem_budget_in_gb=cpu_mem_budget_in_gb,
        )
        return cls(
            **universal,
            gpu_mem_utilization=gpu_mem_utilization,
            kv_page_size=kv_page_size,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            max_dist_size=max_dist_size,
            max_num_embeds=max_num_embeds,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            weight_dtype=weight_dtype,
            dummy_mode=dummy_mode,
        )


# ---------------------------------------------------------------------------
# TOML-shaped driver config (`[model.driver.options]` with type = "dev")
# ---------------------------------------------------------------------------


@dataclass
class NativeDriverConfig:
    """User-facing knobs for the `dev` driver. Splatted into
    `NativeRuntimeConfig.from_args` by the worker.
    """

    gpu_mem_utilization: float = 0.8
    max_batch_tokens: int = 10240
    max_batch_size: int = 512
    max_dist_size: int = 32
    max_num_embeds: int = 128
    max_num_adapters: int = 32
    max_adapter_rank: int = 8
    kv_page_size: int = 16
    weight_dtype: str = "auto"
    cpu_mem_budget_in_gb: int = 0
