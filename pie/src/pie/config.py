"""Pie configuration: typed schema, TOML loader, and platform defaults.

Schema layout (TOML):

    [server]
    host = "127.0.0.1"
    port = 8080
    # primary_model = "qwen-small"

    [auth]
    enabled = false

    [telemetry]
    enabled = false
    endpoint = "http://localhost:4317"

    [model.qwen-small]
    hf_repo = "Qwen/Qwen3-0.6B"
    default_token_budget = 10000          # admission policy lives flat under [model.X]
    default_endowment_pages = 64
    oversubscription_factor = 4.0

    [model.qwen-small.driver]
    type = "vllm"                         # discriminator
    device = ["cuda:0"]
    tensor_parallel_size = 1
    activation_dtype = "bfloat16"

    [model.qwen-small.driver.vllm]
    # Driver-specific knobs in the driver's native vocabulary.
    # Loaded into VllmDriverConfig (or whichever config_cls the driver registers).
    attention_backend = "FLASHINFER"
    gpu_memory_utilization = 0.85

Three concerns are kept separate by section:
  - `[model.X]`               : model identity + admission policy
  - `[model.X.driver]`        : driver-agnostic execution (discriminator,
                                device, parallelism, dtype)
  - `[model.X.driver.<type>]` : driver-specific knobs in that driver's
                                vocabulary (no translation layer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import toml

from pie import path as pie_path


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    verbose: bool = False
    registry: str = "https://registry.pie-project.org/"
    allow_filesystem: bool = False
    max_concurrent_processes: int | None = None
    python_snapshot: bool = True
    primary_model: str | None = None     # which model is "primary"; default = first


@dataclass
class AuthConfig:
    enabled: bool = True


@dataclass
class TelemetryConfig:
    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    service_name: str = "pie"


@dataclass
class RuntimeConfig:
    """Tokio runtime tuning (the `[runtime]` block).

    All fields are opt-in — leaving the section out yields the stock
    `tokio::runtime::Runtime::new()` defaults.

    `worker_threads`: number of tokio worker threads. `None` (default)
    lets tokio pick `num_cpus`. On boxes with high logical-core counts
    and many in-flight tasks (e.g. 96 cores at high request
    concurrency), the default can cause heavy migration / context-switch
    overhead; lowering this (commonly to 8) substantially improves
    throughput. For real GPU backends where Python compute dominates,
    the default is fine — leave this unset unless a profile shows
    benefit.
    """

    worker_threads: int | None = None

    def __post_init__(self):
        if self.worker_threads is not None and self.worker_threads <= 0:
            raise ValueError(
                f"runtime.worker_threads must be > 0 if set "
                f"(got {self.worker_threads!r})"
            )


@dataclass
class SchedulerConfig:
    """The `[model.X.scheduler]` block.

    `policy` selects the batch-firing policy (see
    `runtime/src/inference/adaptive_policy.rs` for the choices and their
    tradeoffs). Recognized values: `"adaptive"` (default), `"eager"`,
    `"greedy"`.
    """

    request_timeout_secs: int = 120
    policy: str | None = None

    def __post_init__(self):
        if self.request_timeout_secs <= 0:
            raise ValueError(
                f"scheduler.request_timeout_secs must be > 0 "
                f"(got {self.request_timeout_secs!r})"
            )
        if self.policy is not None and self.policy not in ("adaptive", "eager", "greedy"):
            raise ValueError(
                f"scheduler.policy must be one of "
                f"'adaptive' | 'eager' | 'greedy' (got {self.policy!r})"
            )


@dataclass
class DriverConfig:
    """The `[model.X.driver]` block.

    `type` is the discriminator (looked up in the driver registry). The
    universal fields (device, tensor_parallel_size, activation_dtype) apply
    to every driver. `options` is the raw `[model.X.driver.<type>]` subsection
    as a dict; it gets parsed into the driver's typed config dataclass at
    server.py / worker.py time, after the registry resolves which class.
    """

    type: str
    device: list[str]
    tensor_parallel_size: int = 1
    activation_dtype: str = "bfloat16"
    random_seed: int = 42
    options: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """One model entry — `[model.<name>]`.

    `name` is the TOML table key (also the inferlet-side lookup key for
    `Model::load(name)`).
    """

    name: str
    hf_repo: str
    driver: DriverConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    # Admission policy (pie's market mechanism)
    default_token_budget: int | None = None
    default_endowment_pages: int = 64
    oversubscription_factor: float = 4.0

    def __post_init__(self):
        if self.default_token_budget is not None and self.default_token_budget <= 0:
            raise ValueError(
                f"Model {self.name!r}: default_token_budget must be > 0 if set "
                f"(got {self.default_token_budget!r})"
            )
        if self.default_endowment_pages <= 0:
            raise ValueError(
                f"Model {self.name!r}: default_endowment_pages must be > 0 "
                f"(got {self.default_endowment_pages!r})"
            )
        if self.oversubscription_factor <= 0.0:
            raise ValueError(
                f"Model {self.name!r}: oversubscription_factor must be > 0 "
                f"(got {self.oversubscription_factor!r})"
            )


@dataclass
class Config:
    """Top-level pie config."""

    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)

    @property
    def primary_model(self) -> ModelConfig:
        if not self.models:
            raise ValueError("No model configurations defined")
        if self.server.primary_model is not None:
            if self.server.primary_model not in self.models:
                raise ValueError(
                    f"server.primary_model = {self.server.primary_model!r} but "
                    f"no [model.{self.server.primary_model}] section exists. "
                    f"Available: {sorted(self.models)}"
                )
            return self.models[self.server.primary_model]
        return next(iter(self.models.values()))


# ---------------------------------------------------------------------------
# Defaults / template
# ---------------------------------------------------------------------------


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def get_default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_default_config_content() -> str:
    device = get_default_device()
    return f"""\
[server]
host = "127.0.0.1"
port = 8080
verbose = false
registry = "https://registry.pie-project.org/"
python_snapshot = true
# max_concurrent_processes = 64
# primary_model = "default"

[auth]
enabled = false

[telemetry]
enabled = false
endpoint = "http://localhost:4317"
service_name = "pie"

# [runtime]
# Tokio runtime tuning. Leave the section out for stock defaults.
# Lower `worker_threads` on boxes with many logical cores when you see
# high context-switch / migration overhead under concurrency.
# worker_threads = 8

[model.default]
hf_repo = "{DEFAULT_MODEL}"
# Admission policy (per-model)
default_endowment_pages = 64
oversubscription_factor = 4.0

[model.default.driver]
type = "native"
device = ["{device}"]
tensor_parallel_size = 1
activation_dtype = "bfloat16"
random_seed = 42

[model.default.driver.native]
gpu_mem_utilization = 0.8
max_batch_tokens = 10240
max_batch_size = 512
"""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_DRIVER_UNIVERSAL_KEYS = {
    "type", "device", "tensor_parallel_size", "activation_dtype", "random_seed",
}


def _parse_driver(model_name: str, raw: dict) -> DriverConfig:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Model {model_name!r}: [model.{model_name}.driver] must be a TOML "
            f"table, got {type(raw).__name__}."
        )
    if "type" not in raw:
        raise ValueError(
            f"Model {model_name!r}: [model.{model_name}.driver] is missing the "
            f"`type` field. Set type = \"vllm\" / \"native\" / \"dummy\" / etc."
        )
    if "device" not in raw:
        raise ValueError(
            f"Model {model_name!r}: [model.{model_name}.driver] is missing the "
            f"`device` field. Set device = [\"cuda:0\"] (or similar)."
        )

    driver_type = raw["type"]
    device = raw["device"]
    if isinstance(device, str):
        device = [device]

    # Type-specific subsection: `[model.X.driver.<type>]` lives at raw[<type>]
    # and is the only sibling of the universal fields that's a sub-table.
    type_specific = raw.get(driver_type, {})
    if not isinstance(type_specific, dict):
        raise ValueError(
            f"Model {model_name!r}: [model.{model_name}.driver.{driver_type}] "
            f"must be a table, got {type(type_specific).__name__}."
        )

    # Refuse stray top-level keys that aren't universal and aren't a known
    # driver subsection — catches typos like `attn_backend` placed at driver
    # level instead of inside `[model.X.driver.vllm]`.
    extra = [k for k in raw
             if k not in _DRIVER_UNIVERSAL_KEYS
             and not isinstance(raw[k], dict)]
    if extra:
        raise ValueError(
            f"Model {model_name!r}: unexpected key(s) under "
            f"[model.{model_name}.driver]: {extra}. Driver-specific knobs "
            f"belong under [model.{model_name}.driver.{driver_type}]."
        )

    return DriverConfig(
        type=driver_type,
        device=device,
        tensor_parallel_size=int(raw.get("tensor_parallel_size", 1)),
        activation_dtype=str(raw.get("activation_dtype", "bfloat16")),
        random_seed=int(raw.get("random_seed", 42)),
        options=type_specific,
    )


def _parse_model(name: str, raw: dict) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ValueError(
            f"[model.{name}] must be a TOML table, got {type(raw).__name__}."
        )
    if "hf_repo" not in raw:
        raise ValueError(f"[model.{name}] is missing required `hf_repo`.")
    if "driver" not in raw:
        raise ValueError(
            f"[model.{name}] is missing the [model.{name}.driver] subsection."
        )

    driver = _parse_driver(name, raw["driver"])
    sched_raw = raw.get("scheduler", {})
    if not isinstance(sched_raw, dict):
        raise ValueError(
            f"[model.{name}.scheduler] must be a TOML table, "
            f"got {type(sched_raw).__name__}."
        )
    scheduler = SchedulerConfig(
        request_timeout_secs=int(sched_raw.get("request_timeout_secs", 120)),
        policy=sched_raw.get("policy"),
    )

    return ModelConfig(
        name=name,
        hf_repo=str(raw["hf_repo"]),
        driver=driver,
        scheduler=scheduler,
        default_token_budget=raw.get("default_token_budget"),
        default_endowment_pages=int(raw.get("default_endowment_pages", 64)),
        oversubscription_factor=float(raw.get("oversubscription_factor", 4.0)),
    )


def load_config(
    config_path: Path | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    no_auth: bool = False,
    verbose: bool = False,
    registry: str | None = None,
) -> Config:
    """Load configuration from TOML and merge CLI overrides."""
    file_path = config_path or pie_path.get_default_config_path()
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration not found at {file_path}")

    raw = toml.loads(file_path.read_text())

    server_raw = raw.get("server", {})
    auth_raw = raw.get("auth", {})
    telemetry_raw = raw.get("telemetry", {})
    runtime_raw = raw.get("runtime", {})
    model_raw = raw.get("model", {})

    if not isinstance(model_raw, dict) or not model_raw:
        raise ValueError(
            "No models configured. At least one [model.<name>] section is required."
        )

    models: dict[str, ModelConfig] = {}
    for name, m in model_raw.items():
        models[name] = _parse_model(name, m)

    server = ServerConfig(
        host=host if host is not None else server_raw.get("host", "127.0.0.1"),
        port=port if port is not None else int(server_raw.get("port", 8080)),
        verbose=verbose or bool(server_raw.get("verbose", False)),
        registry=registry if registry is not None
                 else str(server_raw.get("registry", "https://registry.pie-project.org/")),
        allow_filesystem=bool(server_raw.get("allow_filesystem", False)),
        max_concurrent_processes=server_raw.get("max_concurrent_processes"),
        python_snapshot=bool(server_raw.get("python_snapshot", True)),
        primary_model=server_raw.get("primary_model"),
    )

    auth_enabled = (not no_auth) and bool(auth_raw.get("enabled", True))

    if not isinstance(runtime_raw, dict):
        raise ValueError(
            f"[runtime] must be a TOML table, got {type(runtime_raw).__name__}."
        )
    worker_threads = runtime_raw.get("worker_threads")
    runtime_cfg = RuntimeConfig(
        worker_threads=int(worker_threads) if worker_threads is not None else None,
    )

    return Config(
        server=server,
        auth=AuthConfig(enabled=auth_enabled),
        telemetry=TelemetryConfig(
            enabled=bool(telemetry_raw.get("enabled", False)),
            endpoint=str(telemetry_raw.get("endpoint", "http://localhost:4317")),
            service_name=str(telemetry_raw.get("service_name", "pie")),
        ),
        runtime=runtime_cfg,
        models=models,
    )
