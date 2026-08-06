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
from dataclasses import dataclass, field, fields
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
# [model] / [model.driver] / [model.scheduler]
# ---------------------------------------------------------------------------

@dataclass
class DriverConfig:
    """`[model.driver]` — `type` is required, others have Rust defaults."""
    type: str = "dev"
    device: list[str] = field(default_factory=list)
    tensor_parallel_size: Optional[int] = None
    activation_dtype: Optional[str] = None
    random_seed: Optional[int] = None
    options: dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    request_timeout_secs: Optional[int] = None
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
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def _emit_worker_tables(self, buf, prefix: str) -> None:
        """Emit the worker-domain tables under `prefix` (`""` or `"worker."`).

        Both consumers want the same tables and differ only in nesting, so
        this is written once. Keeping them in one place is what stops the two
        emitters from drifting apart again.
        """
        _emit_table(buf, f"{prefix}server", _block(self.server))
        _emit_table(buf, f"{prefix}telemetry", _block(self.telemetry))
        _emit_table(buf, f"{prefix}runtime", _block(self.runtime))
        m = self.model
        buf.write(f"\n[{prefix}model]\n")
        _emit_kv(buf, "name", m.name)
        _emit_kv(buf, "hf_repo", m.hf_repo)
        _emit_table(buf, f"{prefix}model.driver", _driver_block(m.driver),
                    leading_newline=True)
        if m.driver.options:
            buf.write(f"\n[{prefix}model.driver.options]\n")
            for k, v in m.driver.options.items():
                _emit_kv(buf, k, v)
        if m.scheduler is not None:
            _emit_table(buf, f"{prefix}model.scheduler", _block(m.scheduler),
                        leading_newline=True)

    def to_engine_toml(self) -> str:
        """Serialize to the FLAT document the embedded engine parses.

        `pie._engine.bootstrap` deserializes straight into the worker's own
        `ServeConfig`, which knows nothing about roles — it rejects a
        `[controller]` or `[worker]` table outright. The embedded wheel is
        always single-node and IS the worker, so there is no role to select.
        """
        buf = io.StringIO()
        self._emit_worker_tables(buf, "")
        return buf.getvalue().lstrip("\n")

    def to_toml(self) -> str:
        """Serialize to the six-section document `pie serve --config` reads.

        The file layout and the struct layout are deliberately different
        shapes; `worker/src/config_layout.rs` is the one place they meet, and
        this is its Python mirror. The file has six sections:

            [server]   where it listens, what it fetches from, what it reports
            [model]    which weights
            [driver]   which hardware, and every knob that hardware has
            [runtime]  batching and timeouts
            [sandbox]  what an inferlet may do, and how big it may get
            [cluster]  distributed only; absent for single-node

        `[controller]`, `[gateway]` and `[worker]` are retired and now rejected
        outright (`config_layout::reshape`), as are `[model.driver]` and
        `[model.scheduler]`. That is why this cannot just prefix the internal
        tables the way `to_engine_toml` does: the embedded engine deserializes
        the INTERNAL spelling, and only this method speaks the file's.

        `[auth]` is not emitted: the section no longer exists in the file, and
        the embedded server is always single-node and unauthenticated.
        """
        server: dict = {}
        runtime: dict = {}
        sandbox: dict = {}

        def put(table: dict, key: str, value) -> None:
            if value is not None:
                table[key] = value

        s = self.server
        for name in ("host", "port", "verbose", "registry", "python_snapshot"):
            put(server, name, getattr(s, name))
        # Admission is scheduling, so it sits in `[runtime]` in the file even
        # though it is a `ServerConfig` field internally.
        put(runtime, "max_concurrent_processes", s.max_concurrent_processes)

        # Observability is three keys that belong with the process emitting
        # them rather than in a section of their own.
        t = self.telemetry
        put(server, "telemetry", t.enabled)
        put(server, "otlp_endpoint", t.endpoint)
        put(server, "service_name", t.service_name)

        # `[runtime]` internally is the sandbox: the box an inferlet runs in.
        # The `wasm_` prefix drops off, since the section already says which
        # box these size, and the `_mb` fields become size strings.
        r = self.runtime
        put(server, "worker_threads", r.worker_threads)
        put(server, "max_upload", _mib(r.max_upload_mb))
        put(sandbox, "max_instances", r.wasm_max_instances)
        put(sandbox, "max_memory", _mib(r.wasm_max_memory_mb))
        put(sandbox, "warm_memory", _mib(r.wasm_warm_memory_mb))
        put(sandbox, "warm_slots", r.wasm_warm_slots)
        for name in ("allow_fs", "fs_scratch_dir", "allow_network",
                     "network_allowed_hosts"):
            put(sandbox, name, getattr(r, name))

        m = self.model
        model: dict = {"name": m.name, "model": m.hf_repo}

        # `[model.driver.options]` was four levels deep; the section is now
        # split by NAME, so the common keys and the driver's own knobs sit
        # side by side in one `[driver]`.
        driver = _driver_block(m.driver)
        for key, value in m.driver.options.items():
            if key in driver:
                raise ValueError(
                    f"driver option {key!r} collides with a common [driver] key"
                )
            driver[key] = value

        if m.scheduler is not None:
            sch = m.scheduler
            put(runtime, "request_timeout", _secs(sch.request_timeout_secs))
            for f in fields(sch):
                if f.name in ("request_timeout_secs",):
                    continue
                put(runtime, f.name, getattr(sch, f.name))

        buf = io.StringIO()
        _emit_table(buf, "server", server)
        _emit_table(buf, "model", model, leading_newline=True)
        _emit_table(buf, "driver", driver, leading_newline=True)
        _emit_table(buf, "runtime", runtime, leading_newline=True)
        _emit_table(buf, "sandbox", sandbox, leading_newline=True)
        return buf.getvalue().lstrip("\n")


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


def _mib(value: Optional[int]) -> Optional[str]:
    """`*_mb` fields are counts; the file's size keys are size STRINGS."""
    return None if value is None else f"{value}MiB"


def _secs(value: Optional[int]) -> Optional[str]:
    """Likewise for `*_secs` against the file's duration strings."""
    return None if value is None else f"{value}s"


def _driver_block(d: DriverConfig) -> dict:
    """`[model.driver]` block — skip `options` (it goes in its own
    sub-table) and never None-suppress `type`/`device` (they're required)."""
    out = {"type": d.type, "device": d.device}
    for name in (
        "tensor_parallel_size",
        "activation_dtype",
        "random_seed",
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
