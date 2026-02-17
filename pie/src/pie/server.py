"""Pie server lifecycle.

Manages the full lifecycle: spawn workers → bootstrap Rust runtime →
run workload → shut down.  The ``Server`` async context manager is the
sole public API.
"""

from __future__ import annotations

import asyncio
import copy
import logging
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


# -- Public API ---------------------------------------------------------------


class Server:
    """Async context manager that owns a Pie runtime.

    Usage::

        from pie.server import Server
        from pie.config import Config, ModelConfig, AuthConfig

        cfg = Config(
            auth=AuthConfig(enabled=False),
            models=[ModelConfig(hf_repo="Qwen/Qwen3-0.6B")],
        )
        async with Server(cfg) as server:
            client = await server.connect()
            await client.install_program(wasm_path, manifest_path)
            await client.launch_daemon("my-inferlet@0.1.0", 8080)
    """

    def __init__(self, config: Config):
        self._config = copy.copy(config)
        # Auto-assign a free port if not specified.
        if self._config.port == 0:
            self._config.port = _find_free_port()

        # Filled during __aenter__
        self._handle: Any = None
        self._workers: list = []
        self._clients: list[Any] = []

    async def __aenter__(self) -> Server:
        console = Console(quiet=True)

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
        return f"ws://{self._config.host}:{self._config.port}"

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
    timeout: float = 300.0,
) -> tuple[Any, list]:
    """Spawn workers, collect ready signals, bootstrap the Rust runtime."""
    from pie import _runtime as pie_runtime
    from pie import path as pie_path
    from pie_backend import worker
    import torch
    import torch.multiprocessing as mp

    model = config.primary_model

    # Derive paths
    auth_dir = str(pie_path.get_auth_dir())
    program_dir = str(pie_path.get_program_dir())
    log_dir = str(pie_path.get_log_dir())

    # Validate devices
    device_value = model.device if isinstance(model.device, list) else [model.device]
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
    tp_degree = model.tensor_parallel_size
    if tp_degree is None:
        tp_degree = world_size
        console.print(
            f"[yellow]![/yellow] tensor_parallel_size not set, defaulting to {tp_degree} (use all GPUs)"
        )

    group_topology = worker.calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    console.print("[dim]Starting runtime...[/dim]")
    console.print(f"[dim]  {world_size} devices, {num_groups} groups (TP={tp_degree})[/dim]")

    # Spawn workers
    mp.set_start_method("spawn", force=True)
    master_port = 29500 + random.randint(0, 1000)

    # Build model_config dict for worker (worker_main still expects a dict)
    model_config_dict = asdict(model)
    model_config_dict.pop("name", None)
    model_config_dict.update({
        "telemetry_enabled": config.telemetry.enabled,
        "telemetry_endpoint": config.telemetry.endpoint,
        "telemetry_service_name": config.telemetry.service_name,
    })

    spawn_ctx = mp.get_context("spawn")
    ready_queue = spawn_ctx.Queue()

    ctx = mp.spawn(
        worker.worker_main,
        args=(
            world_size,
            device_value,
            master_port,
            model_config_dict,
            group_topology,
            ready_queue,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
        daemon=True,
    )

    # Collect ready signals
    connected_ranks: set[int] = set()
    server_names_by_group: dict[int, str] = {}
    device_metadata_by_group: dict[int, dict] = {}
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
            rank, server_name, metadata = ready_queue.get(timeout=0.2)
            connected_ranks.add(rank)
            if server_name is not None:
                for gid, group in enumerate(group_topology):
                    if rank in group:
                        server_names_by_group[gid] = server_name
                        device_metadata_by_group[gid] = metadata or {}
                        break
            console.print(f"[dim]  Worker {rank} ready ({len(connected_ranks)}/{world_size})[/dim]")
        except queue.Empty:
            continue

    ready_queue.close()
    ready_queue.join_thread()

    # Build Rust config and bootstrap
    py_devices = []
    for gid in range(num_groups):
        meta = device_metadata_by_group.get(gid, {})
        py_devices.append(
            pie_runtime.DeviceConfig(
                hostname=server_names_by_group[gid],
                total_pages=meta.get("total_pages", 0),
                max_batch_tokens=meta.get("max_batch_tokens", 10240),
                max_batch_size=meta.get("max_batch_size", 128),
            )
        )

    group0_meta = device_metadata_by_group.get(0, {})

    py_model = pie_runtime.ModelConfig(
        name=model.hf_repo,
        arch_name=group0_meta.get("arch_name", "dummy"),
        kv_page_size=model.kv_page_size,
        tokenizer_path=str(Path(group0_meta.get("snapshot_dir", "")) / "tokenizer.json"),
        devices=py_devices,
        scheduler=pie_runtime.SchedulerConfig(
            max_in_flight_batches=4,
            request_timeout_secs=120,
            max_wait_ms=50,
            min_batch_for_optimization=8,
        ),
    )

    rust_config = pie_runtime.Config(
        host=config.host,
        port=config.port,
        verbose=config.verbose,
        registry=config.registry,
        auth_enabled=config.auth.enabled,
        auth_dir=auth_dir,
        program_dir=program_dir,
        log_dir=log_dir,
        telemetry_enabled=config.telemetry.enabled,
        telemetry_endpoint=config.telemetry.endpoint,
        telemetry_service_name=config.telemetry.service_name,
        models=[py_model],
        allow_filesystem=config.allow_filesystem,
    )

    runtime_handle = pie_runtime.bootstrap(rust_config)

    console.print(
        "[green]✓[/green] Runtime started. [dim]Press Ctrl+C to stop[/dim]"
    )

    return runtime_handle, [ctx]




def _check(backend_processes: list) -> bool:
    """Check if all backend processes are still alive."""
    for ctx in backend_processes:
        for p in ctx.processes:
            if not p.is_alive() and p.exitcode != 0:
                log.error("Backend process exited unexpectedly (exit code %s)", p.exitcode)
                return False
    return True


def _terminate(
    server_handle: Any | None,
    backend_processes: list,
) -> None:
    """Terminate the runtime and backend processes."""

    if server_handle is not None:
        try:
            if server_handle.is_running():
                server_handle.shutdown()
        except Exception:
            pass

    time.sleep(1.0)

    for ctx in backend_processes:
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
