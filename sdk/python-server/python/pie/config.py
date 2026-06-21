"""Config dataclasses for the embedded `pie.server.Server`.

These mirror the Rust `crate::config::*` types in `worker/src/config.rs`
field-for-field. Each dataclass serializes itself to TOML (via
`Config.to_toml()`); the resulting string is what the pyo3 layer
hands to `serve::start_engine`. The same TOML the `pie serve --config`
binary consumes — single source of truth on the Rust side.

Fields default to `None` so we don't have to mirror Rust defaults here;
`to_toml()` skips Nones and lets the Rust deserializer fill them in
via `serde(default = "...")`. The exception is `ServerConfig.port = 0`,
which is a Python-side feature ("auto-pick a free port") that pre-dates
serialization — handled by `Server.__aenter__` before TOML-ifying.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# [server]
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    host: Optional[str] = None
    port: Optional[int] = None
    verbose: Optional[bool] = None
    registry: Optional[str] = None
    max_concurrent_processes: Optional[int] = None
    python_snapshot: Optional[bool] = None


# ---------------------------------------------------------------------------
# [auth]
# ---------------------------------------------------------------------------

@dataclass
class AuthConfig:
    enabled: Optional[bool] = None
    authorized_users_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# [telemetry]
# ---------------------------------------------------------------------------

@dataclass
class TelemetryConfig:
    enabled: Optional[bool] = None
    endpoint: Optional[str] = None
    service_name: Optional[str] = None


# ---------------------------------------------------------------------------
# [runtime]
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    worker_threads: Optional[int] = None
    wasm_max_instances: Optional[int] = None
    wasm_max_memory_mb: Optional[int] = None
    wasm_warm_memory_mb: Optional[int] = None
    wasm_warm_slots: Optional[int] = None
    allow_fs: Optional[bool] = None
    fs_scratch_dir: Optional[str] = None
    allow_network: Optional[bool] = None
    network_allowed_hosts: Optional[list[str]] = None
    max_upload_mb: Optional[int] = None


# ---------------------------------------------------------------------------
# [[model]] / [model.driver] / [model.scheduler]
# ---------------------------------------------------------------------------

@dataclass
class DriverConfig:
    """`[model.driver]` — `type` is required, others have Rust defaults."""
    type: str = "dev"
    device: list[str] = field(default_factory=list)
    tensor_parallel_size: Optional[int] = None
    activation_dtype: Optional[str] = None
    random_seed: Optional[int] = None
    ipc_profile: Optional[str] = None
    spin_budget_us: Optional[int] = None
    options: dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    batch_policy: Optional[str] = None
    request_timeout_secs: Optional[int] = None
    default_token_limit: Optional[int] = None
    default_endowment_pages: Optional[int] = None
    admission_oversubscription_factor: Optional[float] = None
    restore_pause_at_utilization: Optional[float] = None
    speculation_depth: Optional[int] = None


@dataclass
class ModelConfig:
    name: str = "default"
    hf_repo: str = ""
    driver: DriverConfig = field(default_factory=DriverConfig)
    scheduler: Optional[SchedulerConfig] = None


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: list[ModelConfig] = field(default_factory=list)

    def to_toml(self) -> str:
        """Serialize to the same TOML schema `pie serve --config` reads."""
        buf = io.StringIO()
        _emit_table(buf, "server", _block(self.server))
        _emit_table(buf, "auth", _block(self.auth))
        _emit_table(buf, "telemetry", _block(self.telemetry))
        _emit_table(buf, "runtime", _block(self.runtime))
        for m in self.models:
            buf.write("\n[[model]]\n")
            _emit_kv(buf, "name", m.name)
            _emit_kv(buf, "hf_repo", m.hf_repo)
            _emit_table(buf, "model.driver", _driver_block(m.driver),
                        leading_newline=True)
            if m.driver.options:
                buf.write("\n[model.driver.options]\n")
                for k, v in m.driver.options.items():
                    _emit_kv(buf, k, v)
            if m.scheduler is not None:
                _emit_table(buf, "model.scheduler", _block(m.scheduler),
                            leading_newline=True)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# TOML emission helpers — minimal hand-rolled serializer that skips None
# and renders the small set of types we use here. We do this rather than
# pull in `tomli_w` to keep the install footprint small.
# ---------------------------------------------------------------------------

def _block(obj) -> dict:
    """Reflect an `Optional[...]`-heavy dataclass into `{key: value}`,
    dropping fields whose value is None. `default_factory=list/dict`
    fields stay even when empty (needed for `device = []` etc.)."""
    from dataclasses import fields, is_dataclass
    out = {}
    if not is_dataclass(obj):
        return out
    for f in fields(obj):
        v = getattr(obj, f.name)
        if v is None:
            continue
        out[f.name] = v
    return out


def _driver_block(d: DriverConfig) -> dict:
    """`[model.driver]` block — skip `options` (it goes in its own
    sub-table) and never None-suppress `type`/`device` (they're required)."""
    out = {"type": d.type, "device": d.device}
    for name in (
        "tensor_parallel_size",
        "activation_dtype",
        "random_seed",
        "ipc_profile",
        "spin_budget_us",
    ):
        v = getattr(d, name)
        if v is not None:
            out[name] = v
    return out


def _emit_table(buf, name: str, kv: dict, leading_newline: bool = False) -> None:
    if not kv:
        return
    if leading_newline:
        buf.write("\n")
    buf.write(f"[{name}]\n")
    for k, v in kv.items():
        _emit_kv(buf, k, v)


def _emit_kv(buf, k: str, v) -> None:
    buf.write(f"{k} = {_render(v)}\n")


def _render(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return _toml_str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_render(x) for x in v) + "]"
    raise TypeError(f"don't know how to TOML-render {type(v).__name__}: {v!r}")


def _toml_str(s: str) -> str:
    """Minimal TOML string escaper — handles backslashes, double-quotes,
    and ASCII control chars. Matches `toml::Value::String`'s parser."""
    escaped = s.replace("\\", "\\\\").replace("\"", "\\\"")
    return f'"{escaped}"'
