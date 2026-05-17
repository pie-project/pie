"""Native driver runtime + TOML configs.

`NativeRuntimeConfig` extends bridge's `RuntimeConfig` with native-only
state (resolved KV page size, adapter pool sizing, weight dtype +
quantization, dummy-mode flag) and overrides the universal `device` and
`activation_dtype` properties to return torch types. Model code under
`pie_driver_dev/model/` keeps reading `runtime_config.device`
(`torch.device`) and `runtime_config.activation_dtype` (`torch.dtype`)
unchanged.

`NativeDriverConfig` is the TOML-shape `[model.driver.options]` block
(with `type = "dev"`), merged into `NativeRuntimeConfig.from_args` by
the worker.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._bridge.config import RuntimeConfig, _resolve_universal_kwargs


# Valid weight dtype categories.
FLOAT_DTYPES = {"float32", "float16", "bfloat16", "auto"}
QUANT_DTYPES = {"int4", "int8", "float8"}
DEFAULT_KV_PAGE_SIZE = 16


def derive_kv_page_size() -> int:
    """Return the dev driver's resolved FlashInfer page size."""
    return DEFAULT_KV_PAGE_SIZE


@dataclass
class NativeRuntimeConfig(RuntimeConfig):
    """Native driver's runtime config — extends `RuntimeConfig` with
    native-only knobs and torch-typed device/dtype properties.

    Field names match pie's existing internal vocabulary (kv_page_size,
    gpu_mem_utilization, etc.) so model code under `pie_driver_dev/model/`
    keeps reading `runtime_config.X` unchanged.
    """

    # Memory + KV layout
    gpu_mem_utilization: float = 0.8
    kv_page_size: int = DEFAULT_KV_PAGE_SIZE

    max_dist_size: int = 32
    max_num_embeds: int = 128

    # Adapter pool sizing (CmaesAdapter)
    max_num_adapters: int = 32
    max_adapter_rank: int = 8

    # Weight dtype + quantization stack
    weight_dtype: str = "auto"

    # Set by the dummy driver to skip real weight loading
    dummy_mode: bool = False

    # ---------- torch-typed property overrides ----------
    @property
    def device(self) -> torch.device:
        return torch.device(self.devices[self.rank])

    @property
    def activation_dtype(self) -> torch.dtype:
        return getattr(torch, self._activation_dtype_str)

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

        # Pre-resolve devices when not explicitly supplied — flavor-side
        # torch probing belongs here, not in bridge.
        if devices is None and device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

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
            kv_page_size=derive_kv_page_size(),
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
    max_dist_size: int = 32
    max_num_embeds: int = 128
    max_num_adapters: int = 32
    max_adapter_rank: int = 8
    weight_dtype: str = "auto"
    cpu_mem_budget_in_gb: int = 0
