"""Config dataclasses for `pie.server.Server`.

They mirror the worker's `config::*` types; `Config.to_toml()` renders the
document `pie serve --config` reads (what `pie config init` writes), so the
file and the embedded server share one schema. Fields default to `None`
and are left out of the TOML, so the Rust side supplies its own defaults.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional


@dataclass
class ServerConfig:
    host: Optional[str] = None
    port: Optional[int] = None  # 0 asks the OS for a free port
    verbose: Optional[bool] = None
    worker_threads: Optional[int] = None
    max_upload_mb: Optional[int] = None


@dataclass
class TelemetryConfig:
    enabled: Optional[bool] = None
    endpoint: Optional[str] = None
    service_name: Optional[str] = None


@dataclass
class RuntimeConfig:
    # Frame geometry; absent means the engine's own defaults. These exist so
    # a measurement can hold the geometry fixed while something else varies.
    frame_size: Optional[int] = None
    frame_dispatch_depth: Optional[int] = None
    # Durations are written with their unit ("50ms", "120s").
    submit_deadline: Optional[str] = None
    max_concurrent_processes: Optional[int] = None


@dataclass
class SandboxConfig:
    max_instances: Optional[int] = None
    max_memory_mb: Optional[int] = None
    warm_memory_mb: Optional[int] = None
    warm_slots: Optional[int] = None
    allow_fs: Optional[bool] = None
    fs_scratch_dir: Optional[str] = None
    allow_network: Optional[bool] = None
    network_allowed_hosts: Optional[list[str]] = None


@dataclass
class EngineConfig:
    """`[engine]`: `type` and `device` are required; `options` are the
    engine's own knobs, written beside the common keys."""
    type: str = "cuda_native"
    device: list[str] = field(default_factory=list)
    tensor_parallel_size: Optional[int] = None
    activation_dtype: Optional[str] = None
    options: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "default"
    hf_repo: str = ""
    # Which SKU of the checkpoint to serve; None lets the load identify one.
    sku: Optional[str] = None
    weight_dtype: Optional[str] = None
    # Sizes with a unit ("20GiB"); omit for uncapped.
    device_weight_budget: Optional[str] = None
    host_weight_budget: Optional[str] = None
    engine: EngineConfig = field(default_factory=EngineConfig)


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_toml(self) -> str:
        """The document `pie serve --config` reads: `[server]` (telemetry
        folded in), `[model]`, `[engine]` (common keys and the engine's
        options side by side), `[runtime]`, `[sandbox]`."""

        def put(table: dict, key: str, value) -> None:
            if value is not None:
                table[key] = value

        s = self.server
        server: dict = {}
        for name in ("host", "port", "verbose", "worker_threads"):
            put(server, name, getattr(s, name))
        put(server, "max_upload", _mib(s.max_upload_mb))
        t = self.telemetry
        put(server, "telemetry", t.enabled)
        put(server, "otlp_endpoint", t.endpoint)
        put(server, "service_name", t.service_name)

        m = self.model
        model: dict = {"name": m.name, "model": m.hf_repo}
        for name in ("sku", "weight_dtype", "device_weight_budget", "host_weight_budget"):
            put(model, name, getattr(m, name))

        engine: dict = {"type": m.engine.type, "device": m.engine.device}
        put(engine, "tensor_parallel_size", m.engine.tensor_parallel_size)
        put(engine, "activation_dtype", m.engine.activation_dtype)
        for key, value in m.engine.options.items():
            if key in engine:
                raise ValueError(f"engine option {key!r} collides with a common [engine] key")
            engine[key] = value

        buf = io.StringIO()
        _emit_table(buf, "server", server)
        _emit_table(buf, "model", model, leading_newline=True)
        _emit_table(buf, "engine", engine, leading_newline=True)
        _emit_table(buf, "runtime", _block(self.runtime), leading_newline=True)
        _emit_table(buf, "sandbox", _block(self.sandbox), leading_newline=True)
        return buf.getvalue().lstrip("\n")


def _block(obj) -> dict:
    """A dataclass as `{key: value}` without its `None`s. `*_mb` fields are
    counts here and size strings in the file (`warm_memory_mb=64` ->
    `warm_memory="64MiB"`)."""
    out = {}
    if not is_dataclass(obj):
        return out
    for f in fields(obj):
        v = getattr(obj, f.name)
        if v is None:
            continue
        if f.name.endswith("_mb"):
            out[f.name[: -len("_mb")]] = _mib(v)
        else:
            out[f.name] = v
    return out


def _mib(value: Optional[int]) -> Optional[str]:
    return None if value is None else f"{value}MiB"


def _emit_table(buf, name: str, kv: dict, leading_newline: bool = False) -> None:
    if not kv:
        return
    if leading_newline:
        buf.write("\n")
    buf.write(f"[{name}]\n")
    for k, v in kv.items():
        buf.write(f"{k} = {_render(v)}\n")


def _render(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        return "[" + ", ".join(_render(x) for x in v) + "]"
    raise TypeError(f"cannot render {type(v).__name__} as TOML: {v!r}")
