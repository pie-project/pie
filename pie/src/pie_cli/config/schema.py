"""Typed configuration schema for Pie.

Replaces the untyped `dict` configs that previously flowed through the system.
Uses stdlib dataclasses to avoid adding external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AuthConfig:
    """Authentication configuration."""

    enabled: bool = True


@dataclass
class TelemetryConfig:
    """OpenTelemetry tracing configuration."""

    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    service_name: str = "pie"


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    hf_repo: str = ""
    device: list[str] = field(default_factory=lambda: ["cuda:0"])
    tensor_parallel_size: int | None = None
    activation_dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    kv_page_size: int = 16
    max_batch_tokens: int | None = None
    max_dist_size: int = 32
    max_num_embeds: int = 128
    max_num_adapters: int = 32
    max_adapter_rank: int = 8
    gpu_mem_utilization: float = 0.8
    use_cuda_graphs: bool = False
    random_seed: int = 42
    dummy_mode: bool = False

    # Name derived from hf_repo if not explicitly set
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_repo


@dataclass
class EngineConfig:
    """Top-level engine configuration.

    This is the single source of truth passed through the system,
    replacing the untyped dict pairs (engine_config, model_configs).
    """

    host: str = "127.0.0.1"
    port: int = 8080
    verbose: bool = False
    registry: str = "https://registry.pie-project.org/"

    auth: AuthConfig = field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    models: list[ModelConfig] = field(default_factory=list)

    @property
    def primary_model(self) -> ModelConfig:
        """Get the first (primary) model config."""
        if not self.models:
            raise ValueError("No model configuration found")
        return self.models[0]

    def to_legacy_dicts(self) -> tuple[dict, list[dict]]:
        """Convert to legacy (engine_dict, model_dicts) pair.

        Temporary bridge until engine/process.py accepts EngineConfig directly.
        """
        from dataclasses import asdict

        engine_dict = {
            "host": self.host,
            "port": self.port,
            "verbose": self.verbose,
            "registry": self.registry,
            "auth": asdict(self.auth),
            "telemetry": asdict(self.telemetry),
        }

        model_dicts = []
        for m in self.models:
            d = asdict(m)
            d.pop("name", None)
            model_dicts.append(d)

        return engine_dict, model_dicts
