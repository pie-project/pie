"""Config dataclasses for `pie.server.Server`.

They mirror the worker's `config::*` types; `Config.to_toml()` renders the
document `pie serve --config` reads (what `pie config init` writes), so the
file and the embedded server share one schema. Fields default to `None`
and are left out of the TOML, so the Rust side supplies its own defaults.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")


@dataclass
class ServerConfig:
    listen: Optional[str] = None  # "host:port"; port 0 asks the OS for a free one
    verbose: Optional[bool] = None
    worker_threads: Optional[int] = None
    max_upload_mb: Optional[int] = None


@dataclass
class TelemetryConfig:
    # An OTLP collector; unset exports none.
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
    # The worker-plane policy: a built-in name or a .wasm path.
    worker_policy: Optional[str] = None


@dataclass
class SandboxConfig:
    max_instances: Optional[int] = None
    max_memory_mb: Optional[int] = None
    warm_memory_mb: Optional[int] = None
    warm_slots: Optional[int] = None
    allow_fs: Optional[bool] = None
    scratch_dir: Optional[str] = None
    allow_network: Optional[bool] = None
    network_allowed_hosts: Optional[list[str]] = None


@dataclass
class ClusterConfig:
    # The cluster policies the in-process controller runs.
    cluster_policies: list[str] = field(default_factory=list)


@dataclass
class EngineConfig:
    """`[engine]`: `type` and `device` are required; `options` are the
    engine's own knobs, written beside the common keys."""
    type: str = "cuda"
    device: list[str] = field(default_factory=list)
    tensor_parallel_size: Optional[int] = None
    activation_dtype: Optional[str] = None
    options: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "default"
    # A store name ("Qwen/Qwen3.5-0.8B") or a path to a `.zt` artifact.
    source: str = ""
    # What the deployment changes about the one its checkpoint states:
    # the precision it expects ("u4g64", "u4g64+u2g64"), the kv dtype, the
    # parts left unloaded ("vision") and the drafter ("mtp", or "none").
    weight_precision: Optional[str] = None
    kv_dtype: Optional[str] = None
    skip_parts: list[str] = field(default_factory=list)
    drafter: Optional[str] = None
    weight_dtype: Optional[str] = None
    # Sizes with a unit ("20GiB"); omit for uncapped.
    device_weight_budget: Optional[str] = None
    host_weight_budget: Optional[str] = None
    # Device bytes for the kv pages and state slots ("20GiB"); omit to fill
    # what device_memory_share leaves past the weights.
    device_kv_budget: Optional[str] = None
    # The share of the device's memory the load may take; omit for 0.9.
    device_memory_share: Optional[float] = None
    # Pinned host bytes for suspended kv; omit for every seat at full context.
    host_kv_budget: Optional[str] = None
    # Disk bytes suspended kv spills to below the host (the only tier on metal
    # and vulkan); omit for the engine's default, "0B" for off.
    disk_kv_budget: Optional[str] = None
    disk_kv_dir: Optional[str] = None
    # Keep the disk tier for the next run (cuda): a restart takes back the
    # sealed pages it holds as cache.
    disk_kv_persistent: Optional[bool] = None
    max_patches: Optional[int] = None
    max_images: Optional[int] = None
    max_voxels: Optional[int] = None
    max_clips: Optional[int] = None

    def __post_init__(self) -> None:
        if self.host_kv_budget is None and os.environ.get("PIE_HOST_KV_BUDGET_DEFAULT", "1") == "1":
            self.host_kv_budget = "1GiB"
        if self.source and not self.source.startswith(("/", "./", "../")):
            from pathlib import Path

            pie_home = Path(os.environ.get("PIE_HOME", "~/.pie")).expanduser()
            has_art = any(
                (pie_home / "artifacts" / slug).is_dir()
                and any((pie_home / "artifacts" / slug).glob("*.zt"))
                for slug in (self.source, self.source.replace("/", "--"))
            )
            if not has_art:
                encoded = f"models--{self.source.replace('/', '--')}"
                roots = []
                if cache := os.environ.get("HF_HUB_CACHE"):
                    roots.append(Path(cache).expanduser())
                if home := os.environ.get("HF_HOME"):
                    roots.append(Path(home).expanduser() / "hub")
                roots.extend(
                    (
                        Path("/workspace/.cache/huggingface/hub"),
                        Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
                        / "huggingface"
                        / "hub",
                    )
                )
                for root in roots:
                    repo = root / encoded
                    for ref_name in ("main", "master"):
                        ref = repo / "refs" / ref_name
                        if ref.is_file():
                            rev = ref.read_text().strip()
                            snap = repo / "snapshots" / rev
                            if snap.is_dir():
                                self.source = str(snap.resolve())
                                return
                    snaps_dir = repo / "snapshots"
                    if snaps_dir.is_dir():
                        snaps = sorted(p for p in snaps_dir.iterdir() if p.is_dir())
                        if snaps:
                            self.source = str(snaps[-1].resolve())
                            return


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    def to_toml(self) -> str:
        """The document `pie serve --config` reads: one table a section,
        `[engine]` with the engine's options beside its common keys."""

        def put(table: dict, key: str, value) -> None:
            if value is not None:
                table[key] = value

        m = self.model
        source = m.source
        if source and "/models--" in source:
            import re
            match = re.search(r"/models--([^/]+)--([^/]+)/snapshots/", source)
            if match:
                repo_id = f"{match.group(1)}/{match.group(2)}"
                pie_home = os.environ.get("PIE_HOME", os.path.expanduser("~/.pie"))
                art_dir = os.path.join(pie_home, "artifacts", f"{match.group(1)}--{match.group(2)}")
                if os.path.isdir(art_dir):
                    source = repo_id
        model: dict = {"name": m.name, "source": source}
        for name in ("weight_precision", "kv_dtype", "drafter", "weight_dtype", "device_weight_budget",
                     "host_weight_budget", "device_kv_budget", "device_memory_share", "host_kv_budget",
                     "disk_kv_budget", "disk_kv_dir", "disk_kv_persistent",
                     "max_patches", "max_images", "max_voxels", "max_clips"):
            put(model, name, getattr(m, name))
        if "host_kv_budget" not in model:
            env_hkv = os.environ.get("PIE_HOST_KV_BUDGET", "1GiB")
            if env_hkv:
                model["host_kv_budget"] = env_hkv
        if "device_kv_budget" not in model:
            max_conc = self.runtime.max_concurrent_processes
            env_dkv = os.environ.get("PIE_DEVICE_KV_BUDGET")
            if env_dkv:
                model["device_kv_budget"] = env_dkv
            elif max_conc is None or max_conc <= 64:
                model["device_kv_budget"] = "32GiB"
            else:
                model["device_kv_budget"] = "64GiB"
        if m.skip_parts:
            model["skip_parts"] = list(m.skip_parts)

        e = self.engine
        engine: dict = {"type": e.type, "device": e.device}
        put(engine, "tensor_parallel_size", e.tensor_parallel_size)
        put(engine, "activation_dtype", e.activation_dtype)
        for key, value in e.options.items():
            if key in engine:
                raise ValueError(f"engine option {key!r} collides with a common [engine] key")
            engine[key] = value
        if "max_model_len" not in engine:
            env_mml = os.environ.get("PIE_MAX_MODEL_LEN", "4096")
            if env_mml:
                engine["max_model_len"] = int(env_mml)

        cluster: dict = {}
        if self.cluster.cluster_policies:
            cluster["cluster_policies"] = list(self.cluster.cluster_policies)

        buf = io.StringIO()
        _emit_table(buf, "server", _block(self.server))
        _emit_table(buf, "telemetry", _block(self.telemetry), leading_newline=True)
        _emit_table(buf, "model", model, leading_newline=True)
        _emit_table(buf, "engine", engine, leading_newline=True)
        _emit_table(buf, "runtime", _block(self.runtime), leading_newline=True)
        _emit_table(buf, "sandbox", _block(self.sandbox), leading_newline=True)
        _emit_table(buf, "cluster", cluster, leading_newline=True)
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
