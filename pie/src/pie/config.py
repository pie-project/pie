"""Pie configuration: typed schema, TOML loader, and platform defaults.

Schema layout (TOML):

    [server]
    host = "127.0.0.1"
    port = 8080

    [auth]
    enabled = false

    [telemetry]
    enabled = false
    endpoint = "http://localhost:4317"

    [runtime]
    # Per-instance security policies + tokio + wasmtime tuning.
    allow_fs = false
    allow_network = true
    network_allowed_hosts = ["*"]

    [[model]]
    name = "default"
    hf_repo = "Qwen/Qwen3-0.6B"

    [model.scheduler]
    batch_policy = "adaptive"
    request_timeout_secs = 120
    default_endowment_pages = 64
    admission_oversubscription_factor = 4.0
    restore_pause_at_utilization = 0.85

    [model.driver]
    type = "native"
    device = ["cuda:0"]
    tensor_parallel_size = 1
    activation_dtype = "bfloat16"

    [model.driver.options]
    gpu_mem_utilization = 0.8

Concerns kept separate by section:
  - `[server]`                : host/port + global admission cap
  - `[runtime]`               : tokio + wasmtime + per-instance security
  - `[[model]]`               : one entry per model — identity only
  - `[model.scheduler]`       : batch policy + per-process admission/market knobs
  - `[model.driver]`          : driver discriminator + universal driver fields
  - `[model.driver.options]`  : driver-specific knobs in that driver's vocabulary

Multi-model: append additional `[[model]]` blocks. Each model's `name` must
be unique and its `device` list must not overlap with any other model's.
The first `[[model]]` is the implicit default for inferlets that don't
specify a model name.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    max_concurrent_processes: int | None = None
    python_snapshot: bool = True

    def __post_init__(self):
        if self.max_concurrent_processes is not None and self.max_concurrent_processes <= 0:
            raise ValueError(
                f"server.max_concurrent_processes must be > 0 if set "
                f"(got {self.max_concurrent_processes!r})"
            )


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
    """The `[runtime]` block: tokio + wasmtime + per-instance security.

    Defaults are explicit Python values — Rust applies them unconditionally,
    no fallback logic. Wasmtime / tokio defaults are pinned in this file so
    pie's behavior is decoupled from upstream changes.
    """

    # Tokio
    worker_threads: int = field(default_factory=lambda: os.cpu_count() or 4)

    # Wasmtime engine pool
    # Defaults match wasmtime 41 on 64-bit (pinned here so pie's behavior
    # is decoupled from upstream wasmtime version bumps). 4 GiB is large
    # but it's a per-slot virtual reservation; only touched memory is
    # actually mapped. Lower wasm_max_memory_mb to e.g. 64 if you need
    # tight RSS control and your inferlets fit.
    wasm_max_instances: int = 1000
    wasm_max_memory_mb: int = 4096
    wasm_warm_memory_mb: int = 0
    wasm_warm_slots: int = 100

    # Filesystem
    allow_fs: bool = False
    fs_scratch_dir: str = field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "pie")
    )

    # Network
    #
    # `network_allowed_hosts` filters wasi:sockets only. wasi:http bypasses
    # the per-socket hook (its host stack does its own DNS). Set
    # `allow_network = false` for tight outbound HTTP control.
    allow_network: bool = True
    network_allowed_hosts: list[str] = field(default_factory=lambda: ["*"])

    # Uploads
    max_upload_mb: int = 256

    def __post_init__(self):
        if self.worker_threads <= 0:
            raise ValueError(
                f"runtime.worker_threads must be > 0 (got {self.worker_threads!r})"
            )
        if self.wasm_max_instances <= 0:
            raise ValueError(
                f"runtime.wasm_max_instances must be > 0 (got {self.wasm_max_instances!r})"
            )
        if self.wasm_max_memory_mb <= 0:
            raise ValueError(
                f"runtime.wasm_max_memory_mb must be > 0 (got {self.wasm_max_memory_mb!r})"
            )
        if self.wasm_warm_memory_mb < 0:
            raise ValueError(
                f"runtime.wasm_warm_memory_mb must be >= 0 (got {self.wasm_warm_memory_mb!r})"
            )
        if self.wasm_warm_slots < 0:
            raise ValueError(
                f"runtime.wasm_warm_slots must be >= 0 (got {self.wasm_warm_slots!r})"
            )
        if self.max_upload_mb <= 0:
            raise ValueError(
                f"runtime.max_upload_mb must be > 0 (got {self.max_upload_mb!r})"
            )
        if not isinstance(self.network_allowed_hosts, list) or not all(
            isinstance(h, str) for h in self.network_allowed_hosts
        ):
            raise ValueError(
                f"runtime.network_allowed_hosts must be a list of strings "
                f"(got {self.network_allowed_hosts!r})"
            )


@dataclass
class SchedulerConfig:
    """The `[model.scheduler]` block: batch-firing policy + per-process
    admission/market knobs.

    `batch_policy` selects the batch-firing strategy (see
    `runtime/src/inference/adaptive_policy.rs`): `"adaptive"`, `"eager"`,
    or `"greedy"`.

    The `default_*` and `admission_*` fields apply at process admission
    when the launch request doesn't specify them explicitly.
    """

    # Policy
    batch_policy: str = "adaptive"
    request_timeout_secs: int = 120

    # Per-process admission & market policy
    default_token_limit: int | None = None
    default_endowment_pages: int = 64
    admission_oversubscription_factor: float = 4.0
    restore_pause_at_utilization: float = 0.85

    def __post_init__(self):
        if self.batch_policy not in ("adaptive", "eager", "greedy"):
            raise ValueError(
                f"scheduler.batch_policy must be one of "
                f"'adaptive' | 'eager' | 'greedy' (got {self.batch_policy!r})"
            )
        if self.request_timeout_secs <= 0:
            raise ValueError(
                f"scheduler.request_timeout_secs must be > 0 "
                f"(got {self.request_timeout_secs!r})"
            )
        if self.default_token_limit is not None and self.default_token_limit <= 0:
            raise ValueError(
                f"scheduler.default_token_limit must be > 0 if set "
                f"(got {self.default_token_limit!r})"
            )
        if self.default_endowment_pages <= 0:
            raise ValueError(
                f"scheduler.default_endowment_pages must be > 0 "
                f"(got {self.default_endowment_pages!r})"
            )
        if (
            self.admission_oversubscription_factor <= 0.0
            or not math.isfinite(self.admission_oversubscription_factor)
        ):
            raise ValueError(
                f"scheduler.admission_oversubscription_factor must be a "
                f"finite > 0 number "
                f"(got {self.admission_oversubscription_factor!r})"
            )
        if not 0.0 < self.restore_pause_at_utilization <= 1.0:
            raise ValueError(
                f"scheduler.restore_pause_at_utilization must be in (0.0, 1.0] "
                f"(got {self.restore_pause_at_utilization!r})"
            )


@dataclass
class DriverConfig:
    """The `[model.driver]` block.

    `type` is the discriminator (looked up in the driver registry). The
    universal fields apply to every driver. `options` is the
    `[model.driver.options]` sub-table — driver-specific knobs in the
    driver's own vocabulary, parsed by the driver's typed config dataclass
    at server.py / worker.py time.
    """

    type: str
    device: list[str]
    tensor_parallel_size: int = 1
    activation_dtype: str = "bfloat16"
    random_seed: int = 42
    options: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """One `[[model]]` entry — model identity + dispatched subsections.

    `name` is the inferlet-side lookup key for `Model::load(name)`. Must be
    unique across `[[model]]` entries.

    Per-process admission policy lives in `[model.scheduler]`, not here.
    """

    name: str
    hf_repo: str
    driver: DriverConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self):
        if not self.name:
            raise ValueError("model.name must be a non-empty string")


@dataclass
class Config:
    """Top-level pie config. Models are an ordered list; the first entry
    is the implicit default."""

    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: list[ModelConfig] = field(default_factory=list)


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
    """Render the default config TOML.

    Default values for `[runtime]` come from `RuntimeConfig()` so the
    template can't drift from the dataclass.
    """
    device = get_default_device()
    rt = RuntimeConfig()
    fs_scratch_dir = rt.fs_scratch_dir.replace("\\", "\\\\")  # TOML escape

    return f"""\
[server]
host = "127.0.0.1"
port = 8080
verbose = false
registry = "https://registry.pie-project.org/"
python_snapshot = true
# max_concurrent_processes = 64           # global cap on in-flight inferlets

[auth]
enabled = false

[telemetry]
enabled = false
endpoint = "http://localhost:4317"
service_name = "pie"

[runtime]
# Tokio + wasmtime tuning. Defaults below are pinned by pie (decoupled
# from upstream wasmtime changes). Edit any value to override.
worker_threads = {rt.worker_threads}
wasm_max_instances = {rt.wasm_max_instances}
wasm_max_memory_mb = {rt.wasm_max_memory_mb}
wasm_warm_memory_mb = {rt.wasm_warm_memory_mb}
wasm_warm_slots = {rt.wasm_warm_slots}

# Filesystem. allow_fs = true mounts a per-process /scratch dir with full RW.
allow_fs = {str(rt.allow_fs).lower()}
fs_scratch_dir = "{fs_scratch_dir}"

# Network. `network_allowed_hosts` filters wasi:sockets only — wasi:http
# bypasses the per-socket hook (its host stack does its own DNS). Set
# allow_network = false for tight outbound HTTP control.
allow_network = {str(rt.allow_network).lower()}
network_allowed_hosts = ["*"]
# Examples:
#   network_allowed_hosts = ["10.0.0.0/8", "127.0.0.1"]
#   network_allowed_hosts = ["10.0.0.0/8:443"]

# Uploads
max_upload_mb = {rt.max_upload_mb}

[[model]]
name = "default"
hf_repo = "{DEFAULT_MODEL}"

[model.scheduler]
# Batch firing — adaptive | eager | greedy
# (see runtime/src/inference/adaptive_policy.rs)
batch_policy = "adaptive"

# Wall-clock cap on a single forward-pass request (seconds).
request_timeout_secs = 120

# Per-process compute cap (tokens). Uncomment for a hard limit.
# default_token_limit = 100000

# Per-process GPU-page claim under contention (KV pages). Bigger = more
# pages this process is guaranteed to hold when others compete.
default_endowment_pages = 64

# Admission overbook ratio: Σ endowment ≤ total_pages × this. 1.0 = no
# overbook (every claim honored at all times); 4.0 = sell 4× capacity,
# betting on non-peak duty cycles (like an airline).
admission_oversubscription_factor = 4.0

# Pause restoring suspended processes when any device exceeds this GPU
# page utilization (0.0–1.0). Prevents evict→restore→re-evict thrash.
restore_pause_at_utilization = 0.85

[model.driver]
type = "native"
device = ["{device}"]
tensor_parallel_size = 1
activation_dtype = "bfloat16"
random_seed = 42

[model.driver.options]
gpu_mem_utilization = 0.8
max_batch_tokens = 10240
max_batch_size = 512

# To add a second model, append another [[model]] block with a unique name
# and a non-overlapping device list. The first [[model]] is the implicit
# default for inferlets that don't specify a model name.
"""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_DRIVER_UNIVERSAL_KEYS = {
    "type", "device", "tensor_parallel_size", "activation_dtype", "random_seed",
}

_SERVER_KEYS = {
    "host", "port", "verbose", "registry", "max_concurrent_processes",
    "python_snapshot",
}
_RUNTIME_KEYS = {
    "worker_threads",
    "wasm_max_instances", "wasm_max_memory_mb",
    "wasm_warm_memory_mb", "wasm_warm_slots",
    "allow_fs", "fs_scratch_dir",
    "allow_network", "network_allowed_hosts",
    "max_upload_mb",
}
_TELEMETRY_KEYS = {"enabled", "endpoint", "service_name"}
_AUTH_KEYS = {"enabled"}
_MODEL_KEYS = {"name", "hf_repo", "scheduler", "driver"}
_SCHEDULER_KEYS = {
    "batch_policy", "request_timeout_secs",
    "default_token_limit", "default_endowment_pages",
    "admission_oversubscription_factor", "restore_pause_at_utilization",
}


def _reject_unknown(section: str, raw: dict, allowed: set[str]) -> None:
    extra = sorted(k for k in raw if k not in allowed)
    if extra:
        raise ValueError(
            f"[{section}]: unknown key(s) {extra}. "
            f"Allowed: {sorted(allowed)}."
        )


def _parse_driver(model_name: str, raw: dict) -> DriverConfig:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Model {model_name!r}: [model.driver] must be a TOML table, "
            f"got {type(raw).__name__}."
        )
    if "type" not in raw:
        raise ValueError(
            f"Model {model_name!r}: [model.driver] is missing the `type` field."
        )
    if "device" not in raw:
        raise ValueError(
            f"Model {model_name!r}: [model.driver] is missing the `device` field."
        )

    options = raw.get("options", {})
    if not isinstance(options, dict):
        raise ValueError(
            f"Model {model_name!r}: [model.driver.options] must be a table, "
            f"got {type(options).__name__}."
        )

    # Refuse stray top-level keys at [model.driver]. Driver-specific knobs
    # belong inside [model.driver.options].
    extra = [
        k for k in raw
        if k not in _DRIVER_UNIVERSAL_KEYS and k != "options"
    ]
    if extra:
        raise ValueError(
            f"Model {model_name!r}: unexpected key(s) under [model.driver]: "
            f"{sorted(extra)}. Driver-specific knobs belong under "
            f"[model.driver.options]."
        )

    device = raw["device"]
    if isinstance(device, str):
        device = [device]
    elif not isinstance(device, list) or not all(isinstance(d, str) for d in device):
        raise ValueError(
            f"Model {model_name!r}: [model.driver].device must be a string "
            f"or a list of strings (got {device!r})."
        )
    if not device:
        raise ValueError(
            f"Model {model_name!r}: [model.driver].device must be non-empty."
        )

    return DriverConfig(
        type=str(raw["type"]),
        device=list(device),
        tensor_parallel_size=int(raw.get("tensor_parallel_size", 1)),
        activation_dtype=str(raw.get("activation_dtype", "bfloat16")),
        random_seed=int(raw.get("random_seed", 42)),
        options=options,
    )


def _parse_model(raw: dict, index: int) -> ModelConfig:
    """Parse one `[[model]]` entry."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"[[model]] #{index} must be a TOML table, got {type(raw).__name__}."
        )
    if "name" not in raw:
        raise ValueError(f"[[model]] #{index} is missing the required `name` field.")
    if "hf_repo" not in raw:
        raise ValueError(
            f"[[model]] {raw['name']!r}: missing the required `hf_repo` field."
        )
    if "driver" not in raw:
        raise ValueError(
            f"[[model]] {raw['name']!r}: missing the [model.driver] subsection."
        )

    # Friendly migration errors for fields that moved to [model.scheduler].
    moved = {
        "default_token_budget": "default_token_limit",
        "default_endowment_pages": "default_endowment_pages",
        "oversubscription_factor": "admission_oversubscription_factor",
    }
    for old_key, new_key in moved.items():
        if old_key in raw:
            renamed = "" if old_key == new_key else f" (renamed to `{new_key}`)"
            raise ValueError(
                f"[[model]] {raw['name']!r}: `{old_key}` has moved to "
                f"[model.scheduler]{renamed}. Move admission/market policy "
                f"under [model.scheduler]."
            )

    _reject_unknown(f"[[model]] {raw['name']!r}", raw, _MODEL_KEYS)

    name = str(raw["name"])
    driver = _parse_driver(name, raw["driver"])

    sched_raw = raw.get("scheduler", {})
    if not isinstance(sched_raw, dict):
        raise ValueError(
            f"Model {name!r}: [model.scheduler] must be a table, "
            f"got {type(sched_raw).__name__}."
        )
    if "policy" in sched_raw:
        raise ValueError(
            f"Model {name!r}: [model.scheduler].policy has been renamed to "
            f"`batch_policy`."
        )
    _reject_unknown(f"model.{name}.scheduler", sched_raw, _SCHEDULER_KEYS)
    sched_kwargs: dict[str, Any] = {
        k: sched_raw[k] for k in _SCHEDULER_KEYS if k in sched_raw
    }
    scheduler = SchedulerConfig(**sched_kwargs)

    return ModelConfig(
        name=name,
        hf_repo=str(raw["hf_repo"]),
        driver=driver,
        scheduler=scheduler,
    )


def _check_devices_disjoint(models: list[ModelConfig]) -> None:
    """Each device string must belong to exactly one model. Sharing a GPU
    across models in one pie process isn't supported — run two pies if
    you need it."""
    seen: dict[str, str] = {}
    for m in models:
        for dev in m.driver.device:
            if dev in seen:
                raise ValueError(
                    f"Device {dev!r} claimed by both model {seen[dev]!r} and "
                    f"model {m.name!r}. Each device must belong to exactly one "
                    f"[[model]]."
                )
            seen[dev] = m.name


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
    model_raw = raw.get("model", [])

    # Old-shape detection: [model.foo] / [model.bar] would parse as a dict.
    # Reject with a migration message — the new shape is [[model]].
    if isinstance(model_raw, dict):
        raise ValueError(
            "Config schema migrated: dotted-key models like [model.default] "
            "are no longer supported. Convert each [model.<name>] to an "
            "array-of-tables [[model]] entry with `name = \"<name>\"` as a "
            "field. Driver-specific options move from "
            "[model.<name>.driver.<type>] to [model.driver.options]. "
            "Run `pie config init` for the new template."
        )
    if not isinstance(model_raw, list) or not model_raw:
        raise ValueError(
            "At least one [[model]] section is required."
        )

    # Server
    if "allow_filesystem" in server_raw:
        raise ValueError(
            "[server].allow_filesystem has moved to [runtime].allow_fs. "
            "Update your config: remove `allow_filesystem` from [server] and "
            "add `allow_fs = true` to [runtime] (or omit it for the default "
            "of false)."
        )
    if "primary_model" in server_raw:
        raise ValueError(
            "[server].primary_model has been removed. The first [[model]] "
            "entry is the implicit default for inferlets that don't specify "
            "a model name. Reorder your [[model]] blocks to change which is "
            "default."
        )
    _reject_unknown("server", server_raw, _SERVER_KEYS)

    server = ServerConfig(
        host=host if host is not None else server_raw.get("host", "127.0.0.1"),
        port=port if port is not None else int(server_raw.get("port", 8080)),
        verbose=verbose or bool(server_raw.get("verbose", False)),
        registry=registry if registry is not None
                 else str(server_raw.get("registry", "https://registry.pie-project.org/")),
        max_concurrent_processes=server_raw.get("max_concurrent_processes"),
        python_snapshot=bool(server_raw.get("python_snapshot", True)),
    )

    # Auth
    _reject_unknown("auth", auth_raw, _AUTH_KEYS)
    auth_enabled = (not no_auth) and bool(auth_raw.get("enabled", True))

    # Telemetry
    _reject_unknown("telemetry", telemetry_raw, _TELEMETRY_KEYS)

    # Runtime — pass only present keys so dataclass defaults fire for missing ones.
    if not isinstance(runtime_raw, dict):
        raise ValueError(
            f"[runtime] must be a TOML table, got {type(runtime_raw).__name__}."
        )
    _reject_unknown("runtime", runtime_raw, _RUNTIME_KEYS)
    runtime_kwargs: dict[str, Any] = {
        k: runtime_raw[k] for k in _RUNTIME_KEYS if k in runtime_raw
    }
    runtime_cfg = RuntimeConfig(**runtime_kwargs)

    # Models
    models: list[ModelConfig] = []
    seen_names: set[str] = set()
    for i, m in enumerate(model_raw):
        cfg = _parse_model(m, index=i)
        if cfg.name in seen_names:
            raise ValueError(
                f"Duplicate [[model]] name: {cfg.name!r}. Each [[model]] must "
                f"have a unique name."
            )
        seen_names.add(cfg.name)
        models.append(cfg)
    _check_devices_disjoint(models)

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
