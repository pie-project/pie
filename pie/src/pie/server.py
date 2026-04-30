"""Pie server lifecycle.

Manages the full lifecycle: spawn workers → bootstrap Rust runtime →
run workload → shut down.  The ``Server`` async context manager is the
sole public API.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import queue
import socket
import time
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pie_client import PieClient

from rich.console import Console

from pie.config import Config

log = logging.getLogger(__name__)


def _ensure_py_runtime_quiet() -> None:
    """Make sure ~/.pie/py-runtime/ is populated before the engine starts.

    Best-effort: any failure (network, missing dep, partial extract) is
    logged and swallowed. The engine will still try to start; if a
    Python inferlet is then loaded, instantiation will fail with the
    original linker error and at least the user has a clear log line.
    """
    try:
        from bakery import py_runtime
    except ImportError:
        log.debug("bakery not installed; skipping py-runtime auto-install")
        return
    if py_runtime.is_installed():
        return
    try:
        py_runtime.ensure_installed(quiet=True)
        log.info("Installed Python WASM runtime at %s", py_runtime.get_runtime_dir())
    except Exception as exc:
        log.warning(
            "Could not auto-install Python WASM runtime (%s); "
            "Python inferlets will fail to instantiate until you run "
            "`pie config init` manually.",
            exc,
        )


# -- Public API ---------------------------------------------------------------


class Server:
    """Async context manager that owns a Pie runtime.

    Usage::

        from pie.server import Server
        from pie.config import (
            Config, ModelConfig, ServerConfig, AuthConfig, DriverConfig,
        )

        cfg = Config(
            server=ServerConfig(),
            auth=AuthConfig(enabled=False),
            models={"default": ModelConfig(
                name="default",
                hf_repo="Qwen/Qwen3-0.6B",
                driver=DriverConfig(type="native", device=["cuda:0"]),
            )},
        )
        async with Server(cfg) as server:
            client = await server.connect()
            await client.install_program(wasm_path, manifest_path)
            await client.launch_daemon("my-inferlet@0.1.0", 8080)
    """

    def __init__(self, config: Config):
        self._config = copy.copy(config)
        # Auto-assign a free port if not specified.
        if self._config.server.port == 0:
            self._config.server.port = _find_free_port()

        # Filled during __aenter__
        self._handle: Any = None
        self._workers: list = []
        self._clients: list[Any] = []

    async def __aenter__(self) -> Server:
        console = Console(quiet=True)

        # Pre-warm CUDA in the main thread before bootstrap. From a fresh
        # asyncio.to_thread worker, torch.cuda.device_count() can return 0
        # even with GPUs present — both NVML and CUDA-driver paths falter,
        # yielding a spurious "no GPUs visible" error. init() is required
        # because device_count() only caches once torch.cuda._initialized
        # is True. Both safe to call without a GPU.
        import torch
        try:
            torch.cuda.init()
        except Exception:
            pass
        _ = torch.cuda.device_count()

        # Make sure the Python WASM runtime is on disk before the engine
        # spins up. The runtime tarball provides the `componentize-py-runtime`
        # core module that Python inferlets link against; without it,
        # instantiation fails with a cryptic linker error. Best-effort —
        # if the network is unavailable, defer the failure to the moment
        # the engine actually tries to load a Python inferlet.
        await asyncio.to_thread(_ensure_py_runtime_quiet)

        # _bootstrap is synchronous (mp.spawn + queue.get), run on thread
        self._handle, self._workers = await asyncio.to_thread(
            _bootstrap, self._config, console
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close all clients created via connect()
        for client in self._clients:
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

        # Shut down server and workers (blocking, run on thread)
        await asyncio.to_thread(_terminate, self._handle, self._workers)
        return False

    async def connect(self) -> PieClient:
        """Create and return an authenticated ``PieClient``."""
        from pie_client import PieClient as _PieClient

        client = _PieClient(self.url)
        await client.connect()
        await client.auth_by_token(self.token)
        self._clients.append(client)
        return client

    async def wait(self):
        """Block until the runtime exits or a worker dies."""
        while True:
            if not _check(self._workers):
                break
            if (
                self._handle
                and hasattr(self._handle, "is_running")
                and not self._handle.is_running()
            ):
                break
            await asyncio.sleep(1.0)

    @property
    def url(self) -> str:
        """WebSocket URL for client connections."""
        return f"ws://{self._config.server.host}:{self._config.server.port}"

    @property
    def token(self) -> str:
        """Internal auth token (available after ``__aenter__``)."""
        if self._handle is None:
            raise RuntimeError("Server is not started; use 'async with Server(cfg) as server:'")
        return self._handle.internal_token

    @property
    def config(self) -> Config:
        return self._config


# -- Internal -----------------------------------------------------------------


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _bootstrap(
    config: Config,
    console: Console,
    timeout: float = 1200.0,
) -> tuple[Any, list]:
    """Spawn workers, collect ready signals, bootstrap the Rust runtime."""
    import importlib

    from pie import _runtime as pie_runtime
    from pie import path as pie_path
    from pie.drivers import resolve_driver
    import torch
    import torch.multiprocessing as mp

    if len(config.models) > 1:
        # Multi-model is in the schema (`Config.models: dict[str, ModelConfig]`)
        # but the bootstrap loop below spawns workers for one model only. When
        # multi-model is wired up, replace this with a per-model worker pool.
        raise NotImplementedError(
            f"Multi-model bootstrap not yet supported: {sorted(config.models)}. "
            "Configure exactly one [model.<name>] section for now."
        )

    model = config.primary_model
    driver = model.driver

    # Resolve the driver via the registry. This validates that the
    # `type = "..."` discriminator names a registered driver and gives us
    # the worker module path + typed config class.
    spec = resolve_driver(driver.type)
    worker = importlib.import_module(spec.worker_module)

    # Build the typed driver config dataclass from the [model.X.driver.<type>]
    # subsection. Unknown keys raise (the dataclass field set is the schema).
    try:
        driver_options = spec.config_cls(**driver.options)
    except TypeError as e:
        raise ValueError(
            f"Model {model.name!r}: invalid keys in [model.{model.name}.driver."
            f"{driver.type}] — {e}. Allowed fields: "
            f"{[f.name for f in __import__('dataclasses').fields(spec.config_cls)]}"
        )

    # Derive paths
    auth_dir = str(pie_path.get_auth_dir())
    program_dir = str(pie_path.get_program_dir())
    log_dir = str(pie_path.get_log_dir())

    # Validate devices (universal driver field)
    device_value = list(driver.device)
    world_size = len(device_value)

    available_gpus = torch.cuda.device_count()
    for device in device_value:
        if device and device.startswith("cuda:"):
            device_idx = int(device.split(":")[1])
            if device_idx >= available_gpus:
                raise RuntimeError(
                    f"Device '{device}' is not accessible. "
                    f"Only {available_gpus} GPU(s) are visible (cuda:0 to cuda:{available_gpus - 1}). "
                    f"Check CUDA_VISIBLE_DEVICES environment variable."
                )

    # Calculate topology
    tp_degree = driver.tensor_parallel_size
    if tp_degree <= 0:
        tp_degree = world_size

    group_topology = worker.calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    console.print("[dim]Starting runtime...[/dim]")
    console.print(
        f"[dim]  driver={driver.type}, {world_size} devices, "
        f"{num_groups} group(s), TP={tp_degree}[/dim]"
    )

    # Spawn workers
    mp.set_start_method("spawn", force=True)
    master_port = 29500 + random.randint(0, 1000)

    # Pack only the keys workers actually consume via `RuntimeConfig.from_args`.
    # The model name and admission policy go directly to the Rust ModelConfig
    # (further below); `tensor_parallel_size` is passed explicitly to the
    # worker as `tp_degree`. Per-driver knobs travel separately in
    # `driver_options_dict`.
    model_config_dict = {
        "hf_repo": model.hf_repo,
        "activation_dtype": driver.activation_dtype,
        "random_seed": driver.random_seed,
        "telemetry_enabled": config.telemetry.enabled,
        "telemetry_endpoint": config.telemetry.endpoint,
        "telemetry_service_name": config.telemetry.service_name,
    }

    driver_options_dict = asdict(driver_options)

    spawn_ctx = mp.get_context("spawn")
    ready_queue = spawn_ctx.Queue()

    ctx = mp.spawn(
        worker.worker_main,
        args=(
            world_size,
            device_value,
            master_port,
            model_config_dict,
            driver_options_dict,
            group_topology,
            ready_queue,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
        daemon=True,
    )

    # Collect ready signals (each leader sends a DriverCapabilities;
    # followers send None).
    from pie.capabilities import DriverCapabilities

    connected_ranks: set[int] = set()
    server_names_by_group: dict[int, str] = {}
    capabilities_by_group: dict[int, DriverCapabilities] = {}
    start_wait = time.time()

    while len(connected_ranks) < world_size:
        for p in ctx.processes:
            if not p.is_alive() and p.exitcode != 0:
                raise RuntimeError(
                    f"Worker process {p.pid} died with exit code {p.exitcode}"
                )

        if time.time() - start_wait > timeout:
            ready_queue.close()
            ready_queue.join_thread()
            raise TimeoutError(f"Timed out waiting for {world_size} workers")

        try:
            rank, server_name, payload = ready_queue.get(timeout=0.2)
            connected_ranks.add(rank)
            if server_name is not None:
                if not isinstance(payload, DriverCapabilities):
                    raise RuntimeError(
                        f"Worker {rank} sent unexpected ready payload "
                        f"{type(payload).__name__}; expected DriverCapabilities."
                    )
                for gid, group in enumerate(group_topology):
                    if rank in group:
                        server_names_by_group[gid] = server_name
                        capabilities_by_group[gid] = payload
                        break
            console.print(f"[dim]  Worker {rank} ready ({len(connected_ranks)}/{world_size})[/dim]")
        except queue.Empty:
            continue

    ready_queue.close()
    ready_queue.join_thread()

    # Every group must have a leader that reported capabilities; otherwise
    # we have no source of truth for KV page count, kv_page_size, etc.
    missing_groups = [gid for gid in range(num_groups) if gid not in capabilities_by_group]
    if missing_groups:
        raise RuntimeError(
            f"No DriverCapabilities received from groups {missing_groups}. "
            "Each group leader must publish capabilities before bootstrap."
        )

    # Surface what the driver actually negotiated. Useful when the user
    # asked for kv_page_size=8 and the driver resolved to 16, etc.
    group0_caps = capabilities_by_group[0]
    console.print(
        f"[dim]  Driver capacities: "
        f"arch={group0_caps.arch_name}, "
        f"kv_page_size={group0_caps.kv_page_size}, "
        f"total_pages={group0_caps.total_pages}, "
        f"swap_pool={group0_caps.swap_pool_size}, "
        f"vocab={group0_caps.vocab_size}, "
        f"max_model_len={group0_caps.max_model_len}, "
        f"dtype={group0_caps.activation_dtype}[/dim]"
    )

    # Build Rust config from the per-group capabilities.
    py_devices = []
    for gid in range(num_groups):
        caps = capabilities_by_group[gid]
        py_devices.append(
            pie_runtime.DeviceConfig(
                hostname=server_names_by_group[gid],
                total_pages=caps.total_pages,
                max_batch_tokens=caps.max_batch_tokens,
                max_batch_size=caps.max_batch_size,
                cpu_pages=caps.swap_pool_size,
            )
        )

    py_model = pie_runtime.ModelConfig(
        name=model.name,                              # the [model.X] table key
        arch_name=group0_caps.arch_name,
        kv_page_size=group0_caps.kv_page_size,        # driver's resolved value
        tokenizer_path=str(Path(group0_caps.snapshot_dir) / "tokenizer.json"),
        devices=py_devices,
        scheduler=pie_runtime.SchedulerConfig(
            request_timeout_secs=model.scheduler.request_timeout_secs,
            policy=model.scheduler.policy,
        ),
        default_token_budget=model.default_token_budget,
        default_endowment_pages=model.default_endowment_pages,
        oversubscription_factor=model.oversubscription_factor,
    )

    rust_config = pie_runtime.Config(
        host=config.server.host,
        port=config.server.port,
        verbose=config.server.verbose,
        registry=config.server.registry,
        auth_enabled=config.auth.enabled,
        auth_dir=auth_dir,
        program_dir=program_dir,
        log_dir=log_dir,
        telemetry_enabled=config.telemetry.enabled,
        telemetry_endpoint=config.telemetry.endpoint,
        telemetry_service_name=config.telemetry.service_name,
        runtime=pie_runtime.RuntimeConfig(
            worker_threads=config.runtime.worker_threads,
        ),
        models=[py_model],
        allow_filesystem=config.server.allow_filesystem,
        max_concurrent_processes=config.server.max_concurrent_processes,
        python_snapshot=config.server.python_snapshot,
    )

    runtime_handle = pie_runtime.bootstrap(rust_config)

    console.print(
        "[green]✓[/green] Runtime started. [dim]Press Ctrl+C to stop[/dim]"
    )

    return runtime_handle, [ctx]




def _check(driver_processes: list) -> bool:
    """Check if all driver processes are still alive."""
    for ctx in driver_processes:
        for p in ctx.processes:
            if not p.is_alive() and p.exitcode != 0:
                log.error("Driver process exited unexpectedly (exit code %s)", p.exitcode)
                return False
    return True


def _terminate(
    server_handle: Any | None,
    driver_processes: list,
) -> None:
    """Terminate the runtime and driver processes."""

    if server_handle is not None:
        try:
            if server_handle.is_running():
                server_handle.shutdown()
        except Exception:
            pass

    time.sleep(1.0)

    for ctx in driver_processes:
        try:
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            for p in ctx.processes:
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
            ctx.join(timeout=1)
        except Exception:
            pass

    # Best-effort cleanup of the shmem region. The worker normally unlinks on
    # its own SIGTERM handler, but SIGKILL fallback skips it; the parent has
    # the same name available.
    import ctypes as _ctypes
    try:
        _librt = _ctypes.CDLL("librt.so.1")
        _librt.shm_unlink.argtypes = [_ctypes.c_char_p]
        _librt.shm_unlink(b"/pie_shmem")
    except Exception:
        pass
