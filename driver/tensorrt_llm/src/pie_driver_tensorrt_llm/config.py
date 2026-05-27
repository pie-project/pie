"""TensorRT-LLM driver configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._bridge.config import RuntimeConfig, _resolve_universal_kwargs


@dataclass
class TensorRTLLMRuntimeConfig(RuntimeConfig):
    """Universal Pie config plus the server-resolved HF snapshot path."""

    snapshot_dir: str = ""

    @classmethod
    def from_args(
        cls,
        hf_repo: str,
        *,
        snapshot_dir: str = "",
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
        **_extras,
    ) -> "TensorRTLLMRuntimeConfig":
        kwargs = _resolve_universal_kwargs(
            hf_repo=hf_repo,
            device=device,
            devices=devices,
            rank=rank,
            tensor_parallel_size=tensor_parallel_size,
            activation_dtype=activation_dtype,
            random_seed=random_seed,
            telemetry_enabled=telemetry_enabled,
            telemetry_endpoint=telemetry_endpoint,
            telemetry_service_name=telemetry_service_name,
            cpu_mem_budget_in_gb=cpu_mem_budget_in_gb,
        )
        return cls(**kwargs, snapshot_dir=snapshot_dir)


@dataclass
class TensorRTLLMDriverConfig:
    """Curated TensorRT-LLM LLM API knobs.

    `llm_kwargs` is a deliberate escape hatch for TensorRT-LLM options that
    are version-specific. The stable Pie-facing options below are used for
    capability reporting and the common LLM constructor fields.
    """

    trust_remote_code: bool = True
    skip_tokenizer_init: bool = True
    backend: str | None = None
    attn_backend: str | None = None
    enable_chunked_prefill: bool | None = None
    max_seq_len: int | None = None
    max_batch_size: int | None = None
    max_num_tokens: int | None = None
    kv_cache_free_gpu_memory_fraction: float | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    # "generate" uses the public TensorRT-LLM LLM.generate API. "pyexecutor"
    # drives TensorRT-LLM 1.2.1's private PyExecutor objects directly so KV
    # stays resident across Pie decode steps.
    execution_mode: str = "generate"
    pyexecutor_max_tokens: int = 4096
    pyexecutor_worker_stop_timeout_s: float = 30.0
    pyexecutor_lookahead: bool = False
    pyexecutor_lookahead_min_batch_size: int | None = None
    pyexecutor_direct_token_limit: int | None = None
    pyexecutor_speculative_lookahead: bool = False

    # Pie scheduler capacities advertised by this high-level API driver.
    max_concurrent_requests: int = 128
    max_batched_tokens: int = 8192
    virtual_kv_page_size: int = 16
    virtual_total_pages: int = 65536

    # Bound the driver's replay histories; `None` keeps full histories.
    max_session_histories: int = 65536
    max_history_tokens: int | None = None

    # Generate short deterministic continuations with TensorRT-LLM and drain
    # them through Pie one token at a time. This amortizes the high-level
    # LLM.generate request cost while preserving Pie's per-step API.
    lookahead_tokens: int = 16

    # Use TensorRT-LLM's cache salt field so its own KV reuse stays isolated
    # per Pie context when the backend supports prefix reuse.
    enable_cache_salt: bool = True
