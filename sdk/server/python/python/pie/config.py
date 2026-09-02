"""Config dataclasses for the embedded `pie.server.Server`.

These mirror the Rust `crate::config::*` types in `worker/src/config.rs`
field-for-field. Each dataclass serializes itself to TOML (via
`Config.to_toml()`); the resulting string is what the pyo3 layer
hands to `serve::start_runtime`. The same TOML the `pie serve --config`
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
    # How the PROCESS is sized. These used to sit in the old `[runtime]`
    # because that was where tuning went; they size the server, so they are
    # the server's.
    worker_threads: Optional[int] = None
    max_upload_mb: Optional[int] = None


# ---------------------------------------------------------------------------
# [telemetry]
# ---------------------------------------------------------------------------

@dataclass
class TelemetryConfig:
    enabled: Optional[bool] = None
    endpoint: Optional[str] = None
    service_name: Optional[str] = None


# ---------------------------------------------------------------------------
# [runtime] — batching and timeouts
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    request_timeout_secs: Optional[int] = None
    # Frame geometry. Absent means the engine's own defaults, which is what
    # every ordinary run wants; these exist so a measurement can hold the
    # geometry fixed while something else varies. `pie config tune` moves the
    # same three through `scheduler::reconfigure`, but only inside its own
    # sweep -- without these there is no way to ask the QUESTION of a bench
    # shape, and the sweep's synthetic fleet is not every shape.
    frame_size: Optional[int] = None
    frame_submit_depth: Optional[int] = None
    frame_dispatch_depth: Optional[int] = None
    # Durations are written with their unit ("50ms", "120s").
    submit_deadline: Optional[str] = None
    # Admission is scheduling, so it sits with the batching knobs rather than
    # with the server that happens to run them.
    max_concurrent_processes: Optional[int] = None


# ---------------------------------------------------------------------------
# [sandbox] — the box an inferlet runs in: its walls, and its size
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    # No `wasm_` prefix: the section already says which box these size.
    max_instances: Optional[int] = None
    max_memory_mb: Optional[int] = None
    warm_memory_mb: Optional[int] = None
    warm_slots: Optional[int] = None
    allow_fs: Optional[bool] = None
    fs_scratch_dir: Optional[str] = None
    allow_network: Optional[bool] = None
    network_allowed_hosts: Optional[list[str]] = None
    python_snapshot: Optional[bool] = None


# ---------------------------------------------------------------------------
# [model] / [model.engine]
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """`[model.engine]` — `type` is required, others have Rust defaults."""
    type: str = "dev"
    device: list[str] = field(default_factory=list)
    tensor_parallel_size: Optional[int] = None
    activation_dtype: Optional[str] = None
    # `random_seed`, `kv_pages`, `ready_timeout` and `shutdown_timeout`
    # STOOD HERE. All four retired from the Rust schema (no reader survived
    # the subprocess era), and the worker refuses a retired key by name --
    # so carrying a seat for one here would emit a config that cannot boot.
    options: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "default"
    hf_repo: str = ""
    # Which SKU of that checkpoint to serve, or None to let the load identify
    # one. A vision artifact fits its family's text row AND its own, and
    # identification takes the cheap one first, so the tower is asked for by
    # name. The id space is `model::catalog()`'s.
    sku: Optional[str] = None
    # What the CHECKPOINT holds, which is a model fact. `activation_dtype`
    # and `kv_cache_dtype` stay on the engine because they are what the
    # engine computes and stores in.
    weight_dtype: Optional[str] = None
    engine: EngineConfig = field(default_factory=EngineConfig)


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
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
        _emit_table(buf, f"{prefix}sandbox", _block(self.sandbox))
        m = self.model
        buf.write(f"\n[{prefix}model]\n")
        _emit_kv(buf, "name", m.name)
        # `model`, not `hf_repo`: the field is spelled `model` internally and
        # this emitter feeds the internal deserializer directly.
        _emit_kv(buf, "model", m.hf_repo)
        if m.sku is not None:
            _emit_kv(buf, "sku", m.sku)
        if m.weight_dtype is not None:
            _emit_kv(buf, "weight_dtype", m.weight_dtype)
        _emit_table(buf, f"{prefix}model.engine", _engine_block(m.engine),
                    leading_newline=True)
        if m.engine.options:
            buf.write(f"\n[{prefix}model.engine.options]\n")
            for k, v in m.engine.options.items():
                _emit_kv(buf, k, v)

    def to_engine_toml(self) -> str:
        """Serialize to the FLAT document the embedded engine parses.

        `pie._engine.bootstrap` deserializes straight into the worker's own
        `ServeConfig`, which knows nothing about roles. The embedded wheel is
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
            [engine]   which hardware, and every knob that hardware has
            [runtime]  batching and timeouts
            [sandbox]  what an inferlet may do, and how big it may get
            [cluster]  distributed only; absent for single-node

        Only three things move between the two shapes now: telemetry folds
        into `[server]`, `[engine]` is `[model.engine]` flattened together
        with the options the engine's own struct takes, and `hf_repo` is
        written under its current name. `[runtime]` and `[sandbox]` are the
        same keys in the same sections either way.

        `[auth]` has no counterpart here or in `to_engine_toml`: the section
        is not part of the config, so emitting it made every embedded-engine
        boot fail with "unknown field `auth`".
        """
        server: dict = {}

        def put(table: dict, key: str, value) -> None:
            if value is not None:
                table[key] = value

        s = self.server
        for name in ("host", "port", "verbose", "registry"):
            put(server, name, getattr(s, name))
        put(server, "worker_threads", s.worker_threads)
        put(server, "max_upload", _mib(s.max_upload_mb))

        # Observability is three keys that belong with the process emitting
        # them rather than in a section of their own.
        t = self.telemetry
        put(server, "telemetry", t.enabled)
        put(server, "otlp_endpoint", t.endpoint)
        put(server, "service_name", t.service_name)

        # `_block` does the unit work: `*_mb` and `*_secs` fields carry
        # COUNTS and the file spells the same knobs as size and duration
        # STRINGS under the bare stem.
        runtime = _block(self.runtime)
        sandbox = _block(self.sandbox)

        m = self.model
        model: dict = {"name": m.name, "model": m.hf_repo}
        put(model, "sku", m.sku)
        put(model, "weight_dtype", m.weight_dtype)

        # The section is split by NAME rather than by nesting, so the common
        # keys and the engine's own knobs sit side by side in one `[engine]`.
        engine = _engine_block(m.engine)
        for key, value in m.engine.options.items():
            if key in engine:
                raise ValueError(
                    f"engine option {key!r} collides with a common [engine] key"
                )
            engine[key] = value

        buf = io.StringIO()
        _emit_table(buf, "server", server)
        _emit_table(buf, "model", model, leading_newline=True)
        _emit_table(buf, "engine", engine, leading_newline=True)
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
    fields stay even when empty (needed for `device = []` etc.).

    `*_mb` / `*_secs` fields carry COUNTS, while every schema this feeds
    spells the same knob as a size/duration STRING under the bare stem
    (`warm_memory_mb=64` -> `warm_memory="64MiB"`). `to_engine_toml`
    has always converted them; this emitter did not, so any `_mb` field set
    through the embedded path was rejected by the engine as an unknown key.
    `warm_slots` hid it — it is the one sized-sounding knob whose name
    has no suffix, so it was the only one anybody had reason to set.
    """
    from dataclasses import fields, is_dataclass
    out = {}
    if not is_dataclass(obj):
        return out
    for f in fields(obj):
        v = getattr(obj, f.name)
        if v is None:
            continue
        if f.name.endswith("_mb"):
            out[f.name[: -len("_mb")]] = _mib(v)
        elif f.name.endswith("_secs"):
            out[f.name[: -len("_secs")]] = _secs(v)
        else:
            out[f.name] = v
    return out


def _mib(value: Optional[int]) -> Optional[str]:
    """`*_mb` fields are counts; the file's size keys are size STRINGS."""
    return None if value is None else f"{value}MiB"


def _secs(value: Optional[int]) -> Optional[str]:
    """Likewise for `*_secs` against the file's duration strings."""
    return None if value is None else f"{value}s"


def _engine_block(d: EngineConfig) -> dict:
    """`[model.engine]` block — skip `options` (it goes in its own
    sub-table) and never None-suppress `type`/`device` (they're required)."""
    out = {"type": d.type, "device": d.device}
    for name in (
        "tensor_parallel_size",
        "activation_dtype",
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
