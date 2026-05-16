"""Worker-internal runtime configuration.

Bridge owns only the universal `RuntimeConfig`: identity, devices, dtype,
telemetry, swap budget, and the engine-computed `total_pages`. Each
flavor wheel ships its own subclass for flavor-specific knobs and torch
typing (see `pie_driver_dev.config.NativeRuntimeConfig`). Bridge stays
torch-free; the storage fields hold device / dtype as strings, and the
`device` / `activation_dtype` properties expose strings here — flavor
subclasses override them to return torch types.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


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

    Driver-specific runtime knobs (gpu_mem_utilization, weight_dtype,
    batch limits, etc.) live on the driver's own subclass / config, not
    here. Device and dtype are stored as strings ("cuda:0", "bfloat16")
    so the bridge stays torch-free; flavor subclasses resolve to torch
    types via property overrides.
    """

    # Identity
    hf_repo: str
    cache_dir: str
    adapter_path: str

    # Topology — string identifiers ("cuda:0", "cpu", "mps").
    devices: list[str]
    rank: int
    tensor_parallel_size: int     # 1 = DP only, >1 = TP within a group

    # Universal precision/seed — stored as a string identifier
    # ("bfloat16", "float16", "float32") to match torch's
    # `getattr(torch, name)` resolution downstream. Read via the
    # `activation_dtype` property (str here; torch.dtype in flavor
    # subclasses).
    _activation_dtype_str: str
    random_seed: int

    # Telemetry
    telemetry_enabled: bool
    telemetry_endpoint: str
    telemetry_service_name: str

    # CPU swap budget in bytes. Both drivers respect it (0 = disabled).
    swap_budget_bytes: int = 0

    # Engine-computed at load time. None pre-load; set by the engine.
    total_pages: int | None = None

    # NOTE: `kv_page_size` and `max_dist_size` are
    # `NativeRuntimeConfig`-only (see `pie_driver_dev.config`). The shared
    # RPC worker (`_handle_fire_batch`) falls back to
    # `engine.capabilities().kv_page_size` for drivers (vllm/sglang) that
    # don't carry them on their config — see pie_driver_dev/worker.py.

    # ---------- properties ----------
    @property
    def device(self) -> str:
        return self.devices[self.rank]

    @property
    def activation_dtype(self) -> str:
        return self._activation_dtype_str

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
    """Normalize string-shaped kwargs for RuntimeConfig.

    No torch probing happens here — device defaults to `"cuda:0"` when
    no explicit value is supplied. Flavors that want hardware-aware
    auto-detect should override `from_args` in their subclass.
    """
    if devices is not None:
        resolved_devices = list(devices)
    elif device is not None:
        resolved_devices = [device]
    else:
        resolved_devices = ["cuda:0"]

    return dict(
        hf_repo=hf_repo,
        cache_dir=str(get_program_dir()),
        adapter_path=str(get_adapter_dir()),
        devices=resolved_devices,
        rank=rank,
        tensor_parallel_size=tensor_parallel_size,
        _activation_dtype_str=activation_dtype,
        random_seed=random_seed,
        telemetry_enabled=telemetry_enabled,
        telemetry_endpoint=telemetry_endpoint,
        telemetry_service_name=telemetry_service_name,
        swap_budget_bytes=cpu_mem_budget_in_gb * (1 << 30),
    )
