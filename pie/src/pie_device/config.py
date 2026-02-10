"""
Runtime configuration for PIE backend.

This module contains the RuntimeConfig dataclass that encapsulates all
configuration options for the inference runtime.
"""

from __future__ import annotations


from typing import TYPE_CHECKING
from dataclasses import dataclass, asdict

import torch

from .utils import resolve_cache_dir, resolve_adapter_path

# Valid weight dtype categories
FLOAT_DTYPES = {"float32", "float16", "bfloat16", "auto"}
QUANT_DTYPES = {"int4", "int8", "float8"}


@dataclass
class RuntimeConfig:
    """
    Configuration for the PIE inference runtime.

    This class consolidates the old model.Config and RuntimeConfig,
    providing both runtime settings (model, cache_dir) and device/dtype
    configuration used by model forward passes.
    """

    # Model identification
    hf_repo: str  # HuggingFace repo (e.g., "meta-llama/Llama-3.2-1B-Instruct")

    cache_dir: str
    kv_page_size: int
    max_dist_size: int
    max_num_embeds: int
    max_batch_tokens: int | None
    max_batch_size: int | None
    max_num_adapters: int
    max_adapter_rank: int
    gpu_mem_utilization: float
    random_seed: int
    use_cuda_graphs: bool
    tensor_parallel_size: int  # TP degree (1 = DP only, >1 = model split across GPUs)

    # Telemetry settings
    telemetry_enabled: bool
    telemetry_endpoint: str
    telemetry_service_name: str

    # Adapter storage path
    adapter_path: str

    # Evaluated at runtime
    max_num_kv_pages: int | None

    # Device and dtype config (formerly in model.Config)
    devices: list[torch.device]  # Renamed from device for clarity
    rank: int
    activation_dtype: torch.dtype
    weight_dtype: (
        str  # "auto", "float32", "float16", "bfloat16", "int4", "int8", "float8"
    )

    # Dummy mode - skip GPU weight loading, return random tokens
    dummy_mode: bool = False

    # =========================================================================
    # Convenience Properties (formerly in model.Config)
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Get the device for the current rank."""
        return self.devices[self.rank]

    @property
    def world_size(self) -> int:
        """Get the number of devices (total GPUs)."""
        return len(self.devices)

    @property
    def num_groups(self) -> int:
        """Get the number of Data Parallel groups."""
        return max(1, self.world_size // self.tensor_parallel_size)

    @property
    def needs_quantization(self) -> bool:
        """True if weight_dtype is a quantization type (int4, int8, float8)."""
        return self.weight_dtype in QUANT_DTYPES

    @property
    def compute_dtype(self) -> torch.dtype:
        """
        Get the compute dtype for weights.

        For float types: returns the specified torch dtype.
        For 'auto': returns activation_dtype (weights match activations).
        For quantized types: returns activation_dtype (compute in activation precision).
        """
        if self.weight_dtype == "auto" or self.weight_dtype in QUANT_DTYPES:
            return self.activation_dtype
        # Float types: float32, float16, bfloat16
        return getattr(torch, self.weight_dtype)

    @property
    def quantization(
        self,
    ) -> (
        "torchao.quantization.Int4WeightOnlyConfig"
        | "torchao.quantization.Int8WeightOnlyConfig"
        | "torchao.quantization.Float8WeightOnlyConfig"
        | None
    ):
        """Derive quantization config from weight_dtype (only for quantization types)."""
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
        cache_dir: str | None = None,
        kv_page_size: int = 16,
        max_dist_size: int = 64,
        max_num_embeds: int = 128,
        max_batch_tokens: int = 10240,
        max_batch_size: int = 128,
        max_num_adapters: int = 48,
        max_adapter_rank: int = 8,
        gpu_mem_utilization: float = 0.9,
        device: str | None = None,
        devices: list[str] | None = None,
        rank: int = 0,
        world_size: int = 1,
        activation_dtype: str = "bfloat16",
        weight_dtype: str = "auto",
        telemetry_enabled: bool = False,
        telemetry_endpoint: str = "http://localhost:4317",
        telemetry_service_name: str = "pie",
        random_seed: int = 42,
        use_cuda_graphs: bool = True,
        tensor_parallel_size: int = 1,
        adapter_path: str | None = None,
        dummy_mode: bool = False,
    ) -> "RuntimeConfig":
        """
        Factory method to build a validated and resolved RuntimeConfig.

        Args:
            hf_repo: HuggingFace repo (e.g., "meta-llama/Llama-3.2-1B-Instruct")
            cache_dir: Directory for model cache (resolved automatically if None)
            kv_page_size: Size of KV cache pages
            max_dist_size: Maximum distribution size for sampling
            max_num_embeds: Maximum number of embeddings
            max_batch_tokens: Maximum tokens per batch
            max_num_adapters: Maximum number of adapters
            max_adapter_rank: Maximum adapter rank
            gpu_mem_utilization: GPU memory utilization (0.0 - 1.0).
            device: Single device string (backward compatibility)
            devices: List of device strings (preferred for multi-GPU)
            rank: Rank of the current process
            world_size: Total number of processes
            activation_dtype: Activation tensor dtype
            weight_dtype: Weight dtype - "auto", "float32", "float16", "bfloat16", "int4", "int8", "float8"
            telemetry_enabled: Enable telemetry/tracing
            telemetry_endpoint: OTLP endpoint for traces
            telemetry_service_name: Service name in traces
            dummy_mode: Skip GPU weight loading and return random tokens

        Returns:
            Configured RuntimeConfig instance
        """
        # Resolution
        resolved_cache_dir = resolve_cache_dir(cache_dir)
        resolved_adapter_path = resolve_adapter_path(adapter_path)

        # Resolve devices
        resolved_devices: list[torch.device] = []

        if devices is not None:
            resolved_devices = [torch.device(d) for d in devices]
        elif device is not None:
            resolved_devices = [torch.device(device)]
        else:
            # Auto-detect if no devices provided
            if torch.cuda.is_available():
                resolved_devices = [torch.device("cuda:0")]
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_devices = [torch.device("mps")]
            else:
                resolved_devices = [torch.device("cpu")]

        # Adjust world_size if needed
        if world_size > 1 and len(resolved_devices) != world_size:
            # If we have a mismatch (e.g. only 1 device passed but world_size=2),
            # we should probably trust the devices list if it's explicit,
            # or fill with duplicates if it's single?
            # For correctness in get_available_memory, we want distinct devices if possible.
            # But if we assume 1 process per device, init_process will pass the full list.
            # So if len != world_size, it's a potential config error or single-device emulation.
            pass

        # Resolve activation dtype
        resolved_activation_dtype = getattr(torch, activation_dtype, torch.bfloat16)

        # Validate weight_dtype
        valid_weight_dtypes = FLOAT_DTYPES | QUANT_DTYPES
        if weight_dtype not in valid_weight_dtypes:
            raise ValueError(
                f"Invalid weight_dtype: '{weight_dtype}'. "
                f"Expected one of: {', '.join(sorted(valid_weight_dtypes))}"
            )

        # Create the config instance
        return cls(
            hf_repo=hf_repo,
            cache_dir=resolved_cache_dir,
            kv_page_size=kv_page_size,
            max_dist_size=max_dist_size,
            max_num_embeds=max_num_embeds,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            gpu_mem_utilization=gpu_mem_utilization,
            random_seed=random_seed,
            use_cuda_graphs=use_cuda_graphs,
            telemetry_enabled=telemetry_enabled,
            telemetry_endpoint=telemetry_endpoint,
            telemetry_service_name=telemetry_service_name,
            max_num_kv_pages=None,  # Populated by runtime based on memory
            devices=resolved_devices,
            rank=rank,
            activation_dtype=resolved_activation_dtype,
            weight_dtype=weight_dtype,
            tensor_parallel_size=tensor_parallel_size,
            adapter_path=resolved_adapter_path,
            dummy_mode=dummy_mode,
        )

    def print(self) -> None:
        """
        Utility to print configuration in a consistent format.
        """
        print("--- Configuration ---")
        for key, value in asdict(self).items():
            print(f"{key}: {value}")
        print("----------------------")
