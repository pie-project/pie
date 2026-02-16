"""Runtime lifecycle for Pie.

Handles the full lifecycle: spawn workers → bootstrap Rust runtime →
run workload → shut down. Core abstraction is the `runtime` context
manager. Public functions `serve` and `oneshot` implement the two
CLI modes on top of it.
"""

from __future__ import annotations


import queue
import sys
import time
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Generator

import typer
from rich.console import Console

from pie_cli.config.schema import Config


@contextmanager
def runtime(
    config: Config,
    console: Console | None = None,
) -> Generator[tuple[Any, list], None, None]:
    """Start the runtime, yield (handle, workers), shut down on exit.

    Usage::

        with runtime(cfg, console=console) as (handle, workers):
            # do work ...
    """
    if console is None:
        console = Console()

    server_handle = None
    backend_processes = []

    try:
        server_handle, backend_processes = _bootstrap(config, console)
        console.print()
        yield server_handle, backend_processes
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    finally:
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            _terminate(server_handle, backend_processes)
        console.print("[green]✓[/green] Shutdown complete")


def serve(
    config: Config,
    monitor: bool = False,
    console: Console | None = None,
) -> None:
    """Persistent serving mode: poll or TUI monitor."""
    with runtime(config, console=console) as (handle, workers):
        if monitor:
            from pie_cli.monitor.app import LLMMonitorApp
            from pie_cli.monitor.provider import PieMetricsProvider

            model = config.primary_model
            provider = PieMetricsProvider(
                host=config.host,
                port=config.port,
                internal_token=handle.internal_token,
                config={
                    "model": model.name or model.hf_repo,
                    "tp_size": model.tensor_parallel_size or len(model.device),
                    "max_batch": model.max_batch_tokens or 32,
                },
            )
            provider.start()
            try:
                LLMMonitorApp(provider=provider).run()
            finally:
                provider.stop()
        else:
            while True:
                if not _check(workers):
                    break
                if handle and hasattr(handle, "is_running") and not handle.is_running():
                    break
                time.sleep(1.0)


def oneshot(
    config: Config,
    program_name: str | None = None,
    arguments: list[str] | None = None,
    wasm_path: Path | None = None,
    manifest_path: Path | None = None,
    console: Console | None = None,
    force_overwrite: bool = False,
) -> None:
    """Run a single program then shut down."""
    import asyncio

    with runtime(config, console=console) as (handle, workers):
        name = program_name
        if wasm_path is not None and manifest_path is not None:
            import tomllib

            manifest = tomllib.loads(manifest_path.read_text())
            pkg_name = manifest["package"]["name"]
            version = manifest["package"]["version"]
            name = f"{pkg_name}@{version}"

        async def _run():
            nonlocal name
            from pie_client import PieClient, Event

            async with PieClient(f"ws://{config.host}:{config.port}") as client:
                await client.auth_by_token(handle.internal_token)

                # Install from local path if provided
                if wasm_path is not None and manifest_path is not None:
                    if not wasm_path.exists():
                        raise FileNotFoundError(f"Program not found: {wasm_path}")
                    if not manifest_path.exists():
                        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

                    print("Installing program (force overwrite)...")
                    await client.install_program(wasm_path, manifest_path, force_overwrite=force_overwrite)

                # Resolve bare name to name@version if needed
                if "@" not in name:
                    name = await client.resolve_version(name, config.registry)

                # Launch and stream
                print(f"Launching {name}...")
                process = await client.launch_process(
                    name,
                    arguments=arguments or [],
                    capture_outputs=True,
                )
                print(f"Process started: {process.process_id}")

                try:
                    while True:
                        event, value = await process.recv()
                        if event == Event.Stdout:
                            print(value, end="", flush=True)
                        elif event == Event.Stderr:
                            print(value, end="", file=sys.stderr, flush=True)
                        elif event == Event.Message:
                            print(f"[Message] {value}")
                        elif event == Event.Return:
                            print(value)
                            break
                        elif event == Event.Error:
                            print(f"❌ {value}", file=sys.stderr)
                            break
                        elif event == Event.File:
                            print(f"[Received file: {len(value)} bytes]")
                except Exception as e:
                    import traceback
                    print(f"[RECV ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)

        asyncio.run(_run())


# -- Internal -----------------------------------------------------------------


def _bootstrap(
    config: Config,
    console: Console,
    timeout: float = 300.0,
) -> tuple[Any, list]:
    """Spawn workers, collect ready signals, bootstrap the Rust runtime."""
    import pie_runtime
    from pie_cli import path as pie_path
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
                print(
                    f"❌ Backend process exited unexpectedly (exit code {p.exitcode})",
                    file=sys.stderr,
                )
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
