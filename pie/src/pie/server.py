"""Pie server lifecycle.

Manages the full lifecycle: spawn workers → bootstrap Rust runtime →
run workload → shut down.  The ``Server`` async context manager is the
sole public API.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import queue
import random
import socket
import time
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
            models=[ModelConfig(
                name="default",
                hf_repo="Qwen/Qwen3-0.6B",
                driver=DriverConfig(type="native", device=["cuda:0"]),
            )],
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
    """Spawn per-model workers, collect ready signals, bootstrap the Rust runtime.

    Each `[[model]]` entry gets its own worker pool with its own master_port
    and its own typed driver config. Devices are guaranteed disjoint across
    models by `_check_devices_disjoint` at config-load time.
    """
    import importlib

    from pie import _runtime as pie_runtime
    from pie import path as pie_path
    from pie.drivers import resolve_driver
    from pie.capabilities import DriverCapabilities
    import torch
    import torch.multiprocessing as mp

    if not config.models:
        raise ValueError("No [[model]] sections configured.")

    # Derive paths shared across models
    auth_dir = str(pie_path.get_auth_dir())
    program_dir = str(pie_path.get_program_dir())
    log_dir = str(pie_path.get_log_dir())

    # GPU visibility check (cheap, do once)
    available_gpus = torch.cuda.device_count()

    mp.set_start_method("spawn", force=True)
    spawn_ctx = mp.get_context("spawn")

    # Each model needs a distinct master_port for its torch.distributed group.
    # Space them by 100 to leave room for high-TP-degree groups.
    base_master_port = 29500 + random.randint(0, 1000)

    console.print("[dim]Starting runtime...[/dim]")
    console.print(f"[dim]  {len(config.models)} model(s) to launch[/dim]")

    py_models: list = []
    all_worker_pools: list = []

    for model_idx, model in enumerate(config.models):
        py_model, ctx = _spawn_model_workers(
            model=model,
            model_idx=model_idx,
            master_port=base_master_port + model_idx * 100,
            available_gpus=available_gpus,
            telemetry=config.telemetry,
            spawn_ctx=spawn_ctx,
            console=console,
            timeout=timeout,
            pie_runtime=pie_runtime,
        )
        py_models.append(py_model)
        all_worker_pools.append(ctx)

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
            wasm_max_instances=config.runtime.wasm_max_instances,
            wasm_max_memory_mb=config.runtime.wasm_max_memory_mb,
            wasm_warm_memory_mb=config.runtime.wasm_warm_memory_mb,
            wasm_warm_slots=config.runtime.wasm_warm_slots,
            allow_fs=config.runtime.allow_fs,
            fs_scratch_dir=config.runtime.fs_scratch_dir,
            allow_network=config.runtime.allow_network,
            network_allowed_hosts=config.runtime.network_allowed_hosts,
            max_upload_mb=config.runtime.max_upload_mb,
        ),
        models=py_models,
        max_concurrent_processes=config.server.max_concurrent_processes,
        python_snapshot=config.server.python_snapshot,
    )

    runtime_handle = pie_runtime.bootstrap(rust_config)

    console.print(
        "[green]✓[/green] Runtime started. [dim]Press Ctrl+C to stop[/dim]"
    )

    return runtime_handle, all_worker_pools


def _spawn_model_workers(
    *,
    model,                  # pie.config.ModelConfig
    model_idx: int,
    master_port: int,
    available_gpus: int,
    telemetry,              # pie.config.TelemetryConfig
    spawn_ctx,
    console: Console,
    timeout: float,
    pie_runtime,
):
    """Spawn one model's worker pool, gather capabilities, build pyo3 ModelConfig."""
    import importlib

    from pie.drivers import resolve_driver
    from pie.capabilities import DriverCapabilities
    import torch.multiprocessing as mp

    driver = model.driver

    # Resolve the driver via the registry — validates `type` and gives us
    # the worker module path + typed config class.
    spec = resolve_driver(driver.type)
    worker = importlib.import_module(spec.worker_module)

    # Build the typed driver config from [model.driver.options].
    try:
        driver_options = spec.config_cls(**driver.options)
    except TypeError as e:
        allowed = [f.name for f in dataclasses.fields(spec.config_cls)]
        raise ValueError(
            f"Model {model.name!r}: invalid keys in [model.driver.options] — {e}. "
            f"Allowed fields: {allowed}"
        )

    # Validate devices (universal driver field)
    device_value = list(driver.device)
    world_size = len(device_value)
    for dev in device_value:
        if dev and dev.startswith("cuda:"):
            dev_idx = int(dev.split(":")[1])
            if dev_idx >= available_gpus:
                raise RuntimeError(
                    f"Model {model.name!r}: device {dev!r} is not accessible. "
                    f"Only {available_gpus} GPU(s) are visible "
                    f"(cuda:0 to cuda:{available_gpus - 1}). "
                    f"Check CUDA_VISIBLE_DEVICES."
                )

    # Calculate topology
    tp_degree = driver.tensor_parallel_size
    if tp_degree <= 0:
        tp_degree = world_size
    group_topology = worker.calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    console.print(
        f"[dim]  [{model.name}] driver={driver.type}, "
        f"{world_size} device(s), {num_groups} group(s), TP={tp_degree}[/dim]"
    )

    # Pack the per-worker config dicts. Universal driver fields and
    # telemetry get re-flattened here; per-driver knobs travel in
    # driver_options_dict.
    model_config_dict = {
        "hf_repo": model.hf_repo,
        "activation_dtype": driver.activation_dtype,
        "random_seed": driver.random_seed,
        "telemetry_enabled": telemetry.enabled,
        "telemetry_endpoint": telemetry.endpoint,
        "telemetry_service_name": telemetry.service_name,
    }
    driver_options_dict = asdict(driver_options)

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
    connected_ranks: set[int] = set()
    server_names_by_group: dict[int, str] = {}
    capabilities_by_group: dict[int, DriverCapabilities] = {}
    start_wait = time.time()

    while len(connected_ranks) < world_size:
        for p in ctx.processes:
            if not p.is_alive() and p.exitcode != 0:
                raise RuntimeError(
                    f"Model {model.name!r}: worker process {p.pid} died "
                    f"with exit code {p.exitcode}"
                )

        if time.time() - start_wait > timeout:
            ready_queue.close()
            ready_queue.join_thread()
            raise TimeoutError(
                f"Model {model.name!r}: timed out waiting for {world_size} workers"
            )

        try:
            rank, server_name, payload = ready_queue.get(timeout=0.2)
            connected_ranks.add(rank)
            if server_name is not None:
                if not isinstance(payload, DriverCapabilities):
                    raise RuntimeError(
                        f"Model {model.name!r}: worker {rank} sent unexpected "
                        f"ready payload {type(payload).__name__}; expected "
                        f"DriverCapabilities."
                    )
                for gid, group in enumerate(group_topology):
                    if rank in group:
                        server_names_by_group[gid] = server_name
                        capabilities_by_group[gid] = payload
                        break
            console.print(
                f"[dim]    [{model.name}] worker {rank} ready "
                f"({len(connected_ranks)}/{world_size})[/dim]"
            )
        except queue.Empty:
            continue

    ready_queue.close()
    ready_queue.join_thread()

    missing_groups = [gid for gid in range(num_groups) if gid not in capabilities_by_group]
    if missing_groups:
        raise RuntimeError(
            f"Model {model.name!r}: no DriverCapabilities received from groups "
            f"{missing_groups}. Each group leader must publish capabilities."
        )

    group0_caps = capabilities_by_group[0]
    console.print(
        f"[dim]    [{model.name}] capacities: "
        f"arch={group0_caps.arch_name}, "
        f"kv_page_size={group0_caps.kv_page_size}, "
        f"total_pages={group0_caps.total_pages}, "
        f"swap_pool={group0_caps.swap_pool_size}, "
        f"dtype={group0_caps.activation_dtype}[/dim]"
    )

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
        name=model.name,
        arch_name=group0_caps.arch_name,
        kv_page_size=group0_caps.kv_page_size,
        tokenizer_path=str(Path(group0_caps.snapshot_dir) / "tokenizer.json"),
        devices=py_devices,
        scheduler=pie_runtime.SchedulerConfig(
            batch_policy=model.scheduler.batch_policy,
            request_timeout_secs=model.scheduler.request_timeout_secs,
            default_token_limit=model.scheduler.default_token_limit,
            default_endowment_pages=model.scheduler.default_endowment_pages,
            admission_oversubscription_factor=
                model.scheduler.admission_oversubscription_factor,
            restore_pause_at_utilization=
                model.scheduler.restore_pause_at_utilization,
        ),
    )

    return py_model, ctx




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
