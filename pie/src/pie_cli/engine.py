"""Engine lifecycle management for Pie.

Thin orchestrator: spawns worker processes, collects ready signals,
bootstraps the Rust runtime, and manages process lifecycle (terminate/check).
"""

import sys
import time
import random
import warnings
from typing import Optional, Any




def start(
    engine_config: dict,
    model_configs: list[dict],
    timeout: float = 300.0,
    console: Optional[Any] = None,
    on_status: Optional[callable] = None,
) -> tuple["_pie.RuntimeHandle", list]:
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        model_configs: List of model configurations
        timeout: Maximum time to wait for backends to connect (seconds)
        console: Optional rich.console.Console for output
        on_status: Optional callback for status updates: (status_message: str) -> None

    Returns:
        Tuple of (RuntimeHandle, list of worker contexts)

    Raises:
        RuntimeError: If engine or backend fails to start
    """
    from pie_backend import _pie
    from pie_cli import path as pie_path
    from pie_backend import worker
    import torch
    import torch.multiprocessing as mp

    def status(msg: str):
        if on_status:
            on_status(msg)

    model_config = model_configs[0]

    # Derive paths
    auth_dir = str(pie_path.get_auth_dir())
    program_dir = str(pie_path.get_program_dir())
    log_dir = str(pie_path.get_log_dir())
    auth_config = engine_config.get("auth", {})
    telemetry_config = engine_config.get("telemetry", {})

    # — Validate devices —
    device_value = model_config.get("device")
    device_value = device_value if isinstance(device_value, list) else [device_value]
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

    # — Calculate topology —
    tp_degree = model_config.get(
        "tensor_parallel_size", engine_config.get("tensor_parallel_size")
    )
    if tp_degree is None:
        tp_degree = world_size
        if console:
            console.print(
                f"[yellow]![/yellow] tensor_parallel_size not set, defaulting to {tp_degree} (use all GPUs)"
            )

    group_topology = worker.calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    if console:
        console.print("[dim]Starting engine...[/dim]")
    status(f"Initializing multi-GPU backend ({world_size} devices)...")
    status(f"  Topology: {num_groups} groups (TP={tp_degree})")

    # — Spawn workers —
    mp.set_start_method("spawn", force=True)
    master_port = 29500 + random.randint(0, 1000)

    # Add telemetry fields to model_config for RuntimeConfig.from_args()
    model_config_with_telemetry = {
        **model_config,
        "telemetry_enabled": telemetry_config.get("enabled", False),
        "telemetry_endpoint": telemetry_config.get("endpoint", "http://localhost:4317"),
        "telemetry_service_name": telemetry_config.get("service_name", "pie"),
    }

    spawn_ctx = mp.get_context("spawn")
    ready_queue = spawn_ctx.Queue()

    ctx = mp.spawn(
        worker.worker_main,
        args=(
            world_size,
            device_value,
            master_port,
            model_config_with_telemetry,
            group_topology,
            ready_queue,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
        daemon=True,
    )

    # — Collect ready signals —
    connected_ranks = set()
    server_names_by_group = {}
    device_metadata_by_group = {}
    start_wait = time.time()

    import queue

    while len(connected_ranks) < world_size:
        # Check for dead processes
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
            status(f"  Worker {rank} ready ({len(connected_ranks)}/{world_size})")
        except queue.Empty:
            continue

    ready_queue.close()
    ready_queue.join_thread()

    # — Build Rust config and bootstrap —
    py_devices = []
    for gid in range(num_groups):
        meta = device_metadata_by_group.get(gid, {})
        py_devices.append(
            _pie.DeviceConfig(
                hostname=server_names_by_group[gid],
                total_pages=meta.get("total_pages", 0),
                max_batch_tokens=meta.get("max_batch_tokens", 10240),
                max_batch_size=meta.get("max_batch_size", 128),
            )
        )

    # Use group 0's metadata for model-level info (all groups load the same model)
    group0_meta = device_metadata_by_group.get(0, {})

    py_model = _pie.ModelConfig(
        name=model_config.get("hf_repo", "unknown"),
        chat_template=group0_meta.get("chat_template", ""),
        stop_tokens=group0_meta.get("stop_tokens", []),
        kv_page_size=model_config.get("kv_page_size", 16),
        tokenizer_path="",
        devices=py_devices,
        scheduler=_pie.SchedulerConfig(
            max_in_flight_batches=4,
            request_timeout_secs=120,
            max_wait_ms=50,
            min_batch_for_optimization=8,
        ),
    )

    config = _pie.Config(
        host=engine_config.get("host", "127.0.0.1"),
        port=engine_config.get("port", 8080),
        verbose=engine_config.get("verbose", False),
        registry=engine_config.get("registry", "https://registry.pie-project.org/"),
        auth_enabled=auth_config.get("enabled", False),
        auth_dir=auth_dir,
        program_dir=program_dir,
        log_dir=log_dir,
        telemetry_enabled=telemetry_config.get("enabled", False),
        telemetry_endpoint=telemetry_config.get("endpoint", "http://localhost:4317"),
        telemetry_service_name=telemetry_config.get("service_name", "pie"),
        models=[py_model],
    )

    runtime_handle = _pie.bootstrap(config)

    if console:
        console.print(
            "[green]✓[/green] Engine running. [dim]Press Ctrl+C to stop[/dim]"
        )

    return runtime_handle, [ctx]


# =============================================================================
# Process Lifecycle
# =============================================================================


def check(
    backend_processes: list, on_error: Optional[callable] = None
) -> bool:
    """Check if all backend processes are still alive.

    Handles SpawnContext objects from mp.spawn.

    Returns:
        True if all processes are alive, False if any have died
    """
    for ctx in backend_processes:
        if hasattr(ctx, "processes"):
            for p in ctx.processes:
                if not p.is_alive():
                    exitcode = p.exitcode
                    if exitcode != 0:
                        msg = f"Backend process exited unexpectedly (exit code {exitcode})"
                        if on_error:
                            on_error(msg)
                        else:
                            print(f"❌ {msg}", file=sys.stderr)
                        return False
    return True


def terminate(
    server_handle: Optional[Any],
    backend_processes: list,
    on_message: Optional[callable] = None,
) -> None:
    """Terminate the engine and backend processes.

    Args:
        server_handle: The RuntimeHandle (or None if already shut down)
        backend_processes: List of SpawnContext objects
        on_message: Optional callback for status messages
    """
    # Suppress semaphore leak warnings during shutdown
    warnings.filterwarnings(
        "ignore", message=".*leaked semaphore.*", category=UserWarning
    )

    def log(msg: str):
        if on_message:
            on_message(msg)

    # 1. Shut down the server first (sends shutdown signal to workers via IPC)
    if server_handle is not None:
        try:
            if server_handle.is_running():
                server_handle.shutdown()
        except Exception as e:
            log(f"Error shutting down engine: {e}")

    # 2. Give workers time to shut down gracefully
    time.sleep(1.0)

    # 3. Terminate worker processes
    for ctx in backend_processes:
        if hasattr(ctx, "processes"):
            try:
                for p in ctx.processes:
                    if p.is_alive():
                        p.terminate()
                for p in ctx.processes:
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()
                ctx.join(timeout=1)
            except Exception as e:
                log(f"Error terminating SpawnContext: {e}")
