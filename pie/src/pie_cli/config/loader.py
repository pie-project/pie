"""Configuration loading and merging for Pie.

Reads TOML config files and merges CLI overrides into typed EngineConfig.
Extracted from serve.py's load_config().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from pie_cli import path as pie_path
from pie_cli.config.schema import (
    AuthConfig,
    EngineConfig,
    ModelConfig,
    TelemetryConfig,
)


def load_config(
    config_path: Path | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    enable_auth: bool | None = None,
    no_auth: bool = False,
    verbose: bool = False,
    registry: str | None = None,
    dummy_mode: bool = False,
) -> EngineConfig:
    """Load configuration from TOML file and merge CLI overrides.

    Args:
        config_path: Path to TOML config file (defaults to ~/.pie/config.toml).
        host: Override host address.
        port: Override port.
        enable_auth: Explicitly enable/disable auth.
        no_auth: Shorthand to disable auth.
        verbose: Enable verbose logging.
        registry: Override registry URL.
        dummy_mode: Enable dummy mode on all models.

    Returns:
        Fully resolved EngineConfig.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If no model configuration is found.
    """
    file_path = config_path or pie_path.get_default_config_path()
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration not found at {file_path}")

    raw = toml.loads(file_path.read_text())

    engine_section = raw.get("engine", {})
    auth_section = raw.get("auth", {})
    telemetry_section = raw.get("telemetry", {})

    # --- Auth ---
    if no_auth:
        auth_enabled = False
    elif enable_auth is not None:
        auth_enabled = enable_auth
    else:
        auth_enabled = auth_section.get(
            "enabled", engine_section.get("enable_auth", True)
        )

    # --- Models ---
    model_configs_raw = raw.get("model", [])
    if isinstance(model_configs_raw, dict):
        model_configs_raw = [model_configs_raw]
    if not model_configs_raw:
        raise ValueError("No model configuration found")

    models = []
    for mc in model_configs_raw:
        # Normalize device to list
        device = mc.get("device", ["cuda:0"])
        if isinstance(device, str):
            device = [device]

        m = ModelConfig(
            hf_repo=mc.get("hf_repo", ""),
            device=device,
            tensor_parallel_size=mc.get("tensor_parallel_size"),
            activation_dtype=mc.get("activation_dtype", "bfloat16"),
            weight_dtype=mc.get("weight_dtype", "bfloat16"),
            kv_page_size=mc.get("kv_page_size", 16),
            max_batch_tokens=mc.get("max_batch_tokens"),
            max_dist_size=mc.get("max_dist_size", 32),
            max_num_embeds=mc.get("max_num_embeds", 128),
            max_num_adapters=mc.get("max_num_adapters", 32),
            max_adapter_rank=mc.get("max_adapter_rank", 8),
            gpu_mem_utilization=mc.get("gpu_mem_utilization", 0.8),
            use_cuda_graphs=mc.get("use_cuda_graphs", False),
            random_seed=mc.get("random_seed", 42),
            dummy_mode=dummy_mode or mc.get("dummy_mode", False),
            name=mc.get("name", ""),
        )
        models.append(m)

    return EngineConfig(
        host=host or engine_section.get("host", raw.get("host", "127.0.0.1")),
        port=port or engine_section.get("port", raw.get("port", 8080)),
        verbose=verbose or engine_section.get("verbose", raw.get("verbose", False)),
        registry=(
            registry
            or engine_section.get(
                "registry",
                raw.get("registry", "https://registry.pie-project.org/"),
            )
        ),
        auth=AuthConfig(enabled=auth_enabled),
        telemetry=TelemetryConfig(
            enabled=telemetry_section.get("enabled", False),
            endpoint=telemetry_section.get("endpoint", "http://localhost:4317"),
            service_name=telemetry_section.get("service_name", "pie"),
        ),
        models=models,
    )
