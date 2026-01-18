"""Engine and backend management for Pie.

This module handles the lifecycle of the Pie engine and backend services.
"""

import sys
import time
import os
import signal
import subprocess
import random
import asyncio
import warnings
from pathlib import Path


from typing import Optional, Any
import queue  # For Queue type hint logic if needed, but Queue is from MP


class EngineError(Exception):
    """Exception raised for engine/backend errors."""

    pass


class FfiWorkerHandle:
    """Wrapper for FFI worker thread to look like a process for cleanup."""

    def __init__(self, thread, stop_event):
        self.thread = thread
        self.stop_event = stop_event
        self.pid = -1  # Pseudo-PID

    def terminate(self):
        self.stop_event.set()

    def kill(self):
        """Simulate kill by ensuring stop event is set (threads cannot be SIGKILLed)."""
        self.stop_event.set()

    def join(self, timeout=None):
        self.thread.join(timeout=timeout)

    def is_alive(self):
        return self.thread.is_alive()


def start_engine_and_backend(
    engine_config: dict,
    model_configs: list[dict],
    timeout: float = 300.0,
    console: Optional[Any] = None,
    on_status: Optional[callable] = None,
    on_message: Optional[callable] = None,
) -> tuple["_pie.ServerHandle", list]:
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        model_configs: List of model configurations
        timeout: Maximum time to wait for backends to connect (seconds)
        console: Optional rich.console.Console for output
        on_status: Optional callback for status updates: (status_message: str) -> None
        on_message: Optional callback for log messages: (level: str, message: str) -> None

    Returns:
        Tuple of (ServerHandle, list of backend processes - empty for FFI mode)

    Raises:
        EngineError: If engine or backend fails to start
    """
    """Start the Pie engine and all configured backend services.

    Args:
        engine_config: Engine configuration dict
        model_configs: List of model configurations
        timeout: Maximum time to wait for backends to connect (seconds)
        console: Optional rich.console.Console for output
        on_status: Optional callback for status updates: (status_message: str) -> None
        on_message: Optional callback for log messages: (level: str, message: str) -> None

    Returns:
        Tuple of (ServerHandle, list of backend processes - empty for FFI mode)

    Raises:
        EngineError: If engine or backend fails to start
    """

    def status_update(msg: str):
        if on_status:
            on_status(msg)

    def log_message(level: str, msg: str):
        if on_message:
            on_message(level, msg)

    from . import path as pie_path
    from . import _pie
    import torch

    # Load authorized users if auth is enabled
    authorized_users_path = None
    if engine_config.get("enable_auth", True):
        auth_path = pie_path.get_authorized_users_path()
        if auth_path.exists():
            authorized_users_path = str(auth_path)

    # Create server config
    # Get telemetry config from engine_config (loaded from [telemetry] section)
    telemetry_config = engine_config.get("telemetry", {})
    server_config = _pie.ServerConfig(
        host=engine_config.get("host", "127.0.0.1"),
        port=engine_config.get("port", 8080),
        enable_auth=engine_config.get("enable_auth", True),
        cache_dir=engine_config.get("cache_dir"),
        verbose=engine_config.get("verbose", False),
        log_dir=engine_config.get("log_dir"),
        registry=engine_config.get("registry", "https://registry.pie-project.org/"),
        telemetry_enabled=telemetry_config.get("enabled", False),
        telemetry_endpoint=telemetry_config.get("endpoint", "http://localhost:4317"),
        telemetry_service_name=telemetry_config.get("service_name", "pie-runtime"),
    )

    # FFI MODE: Queue-based communication for high throughput
    if console is not None:
        console.print("[dim]Starting engine...[/dim]")

    try:
        # Currently supports single-model configurations
        if len(model_configs) > 1:
            raise EngineError(
                "Currently only single-model configurations are supported"
            )

        model_config = model_configs[0]

        # Detect multi-GPU configuration
        device_value = model_config.get("device")
        world_size = len(device_value) if isinstance(device_value, list) else 1

        # Validate that all configured devices are accessible
        devices_to_validate = (
            device_value if isinstance(device_value, list) else [device_value]
        )
        available_gpus = torch.cuda.device_count()

        for device in devices_to_validate:
            if device and device.startswith("cuda:"):
                device_idx = int(device.split(":")[1])
                if device_idx >= available_gpus:
                    raise EngineError(
                        f"Device '{device}' is not accessible. "
                        f"Only {available_gpus} GPU(s) are visible (cuda:0 to cuda:{available_gpus - 1}). "
                        f"Check CUDA_VISIBLE_DEVICES environment variable."
                    )

        if world_size > 1:
            # Multi-GPU FFI mode: spawn worker processes
            backend_processes = _start_multi_gpu_ffi_backend(
                engine_config,
                model_config,
                server_config,
                authorized_users_path,
                device_value,
                world_size,
                console,
                status_update,
                timeout,
            )
            server_handle = backend_processes.pop(0)  # First element is server_handle
        else:
            # Single-GPU mode: use same IPC architecture as multi-GPU with world_size=1
            status_update("Initializing single-GPU backend...")

            # Treat as multi-GPU with 1 device for unified code path
            backend_processes = _start_multi_gpu_ffi_backend(
                engine_config,
                model_config,
                server_config,
                authorized_users_path,
                [device_value] if isinstance(device_value, str) else device_value,
                1,  # world_size = 1
                console,
                status_update,
                timeout,
            )
            server_handle = backend_processes.pop(0)  # First element is server_handle

    except Exception as e:
        raise EngineError(f"Failed to initialize backend: {e}") from e

    # Final success message
    if console is not None:
        console.print(
            "[green]✓[/green] Engine running. [dim]Press Ctrl+C to stop[/dim]"
        )

    return server_handle, backend_processes


def _build_backend_config(
    engine_config: dict, model_config: dict, internal_token: str
) -> dict:
    """Build the configuration dict for RuntimeConfig.from_args().

    Args:
        engine_config: Engine configuration dict (host, port) - not used for RuntimeConfig
        model_config: Model configuration dict
        internal_token: Internal authentication token - not used for RuntimeConfig

    Returns:
        Configuration dict suitable for RuntimeConfig.from_args(**kwargs)
    """
    # Note: host, port, internal_auth_token are for pycrust server registration,
    # not RuntimeConfig. They're excluded here because in FFI mode we don't
    # need to register with the engine - we call Python directly.

    # Handle device field - can be string or list
    # RuntimeConfig.from_args expects: device (str) OR devices (list[str])
    device_value = model_config.get("device")
    device_key = None
    if device_value is not None:
        if isinstance(device_value, list):
            device_key = "devices"
        else:
            device_key = "device"

    config = {
        "hf_repo": model_config.get("hf_repo"),
        "cache_dir": model_config.get("cache_dir"),
        "kv_page_size": model_config.get("kv_page_size", 16),
        "max_dist_size": model_config.get("max_dist_size", 64),
        "max_num_embeds": model_config.get("max_num_embeds", 128),
        "max_batch_tokens": model_config.get("max_batch_tokens", 10240),
        "max_num_adapters": model_config.get("max_num_adapters", 48),
        "max_adapter_rank": model_config.get("max_adapter_rank", 8),
        "adapter_path": model_config.get("adapter_path"),
        "gpu_mem_utilization": model_config.get("gpu_mem_utilization", 0.9),
        "max_batch_size": model_config.get("max_batch_size", 128),
        "activation_dtype": model_config.get("activation_dtype", "bfloat16"),
        "weight_dtype": model_config.get("weight_dtype"),
        "telemetry_enabled": engine_config.get("telemetry", {}).get("enabled", False),
        "telemetry_endpoint": engine_config.get("telemetry", {}).get(
            "endpoint", "http://localhost:4317"
        ),
        "telemetry_service_name": engine_config.get("telemetry", {}).get(
            "service_name", "pie"
        ),
        "random_seed": model_config.get("random_seed", 42),
        "use_cuda_graphs": model_config.get("use_cuda_graphs", False),
        "dummy_mode": model_config.get("dummy_mode", False),
    }

    # Add device with correct key
    if device_key is not None:
        config[device_key] = device_value

    # Remove None values to use from_args() defaults
    return {k: v for k, v in config.items() if v is not None}


# =============================================================================
# Multi-GPU FFI Mode Helpers
# =============================================================================


def _init_distributed(rank: int, world_size: int, master_port: int, device: str):
    """Initialize torch.distributed for a given rank.

    Sets up CUDA device and process group using FileStore for rendezvous.
    """
    import datetime
    import torch
    import torch.distributed as dist

    # Set CUDA device
    torch.cuda.set_device(device)

    # Suppress harmless warnings
    warnings.filterwarnings(
        "ignore", message=".*barrier.*device under current context.*"
    )

    # Use FileStore for more robust rendezvous (avoids port conflicts)
    store_path = f"/tmp/pie_dist_store_{master_port}"
    store = dist.FileStore(store_path, world_size)
    timeout = datetime.timedelta(seconds=300)

    # Initialize process group with NCCL
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Extract device index for device_id parameter
    device_id = None
    if device.startswith("cuda:"):
        device_id = torch.device(device)

    dist.init_process_group(
        backend,
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        device_id=device_id,
    )


def _setup_process_groups(rank: int, group_topology: list[list[int]]) -> dict:
    """Create ProcessGroups for each execution group (Rank 0 + Group Workers)."""
    import torch.distributed as dist

    pg_map = {}
    for i, group_ranks in enumerate(group_topology):
        # Comm group includes Rank 0 (Controller) + Group Workers
        comm_ranks = sorted(list(set([0] + group_ranks)))

        # Create group (Collective: all ranks/participants must call this)
        # Note: Depending on backend, might need global participation.
        # For safety in NCCL, everyone calls new_group for all groups.
        pg = dist.new_group(comm_ranks)

        pg_map[i] = pg

    return pg_map


def _setup_compute_process_groups(rank: int, group_topology: list[list[int]]) -> dict:
    """Create ProcessGroups for Tensor Parallel computation (TP ranks only)."""
    import torch.distributed as dist

    pg_map = {}
    for i, group_ranks in enumerate(group_topology):
        # Compute group includes ONLY the TP workers for this group
        # Sort to ensure consistent ordering
        comm_ranks = sorted(list(set(group_ranks)))

        # Create group
        pg = dist.new_group(comm_ranks)

        pg_map[i] = pg

    return pg_map


def _create_runtime(
    config_dict: dict,
    devices: list[str],
    rank: int,
    world_size: int,
    group_topology: list[list[int]],
    result_queue: Any | None = None,
    result_queues: list | None = None,  # All queues (for Rank 0)
    pg_map: dict | None = None,
    compute_pg_map: dict | None = None,
):
    """Create a Runtime instance for the given rank."""

    from pie_worker.config import RuntimeConfig
    from pie_worker.runtime import Runtime

    # Remove device/devices from config to avoid duplicate argument
    filtered_config = {
        k: v for k, v in config_dict.items() if k not in ("device", "devices")
    }

    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=devices,
        rank=rank,
        world_size=world_size,
    )

    # Determine my group ID
    my_group_id = 0
    for i, group in enumerate(group_topology):
        if rank in group:
            my_group_id = i
            break

    rt = Runtime(
        config,
        log_queue=None,
        group_id=my_group_id,
        result_queue=result_queue,
        result_queues=result_queues,
        process_groups=pg_map,
        compute_process_groups=compute_pg_map,
        group_topology=group_topology,
    )
    return rt


# =============================================================================
# Multi-GPU FFI Mode Entry Points
# =============================================================================


def _start_multi_gpu_ffi_backend(
    engine_config: dict,
    model_config: dict,
    server_config,
    authorized_users_path: str | None,
    devices: list[str],
    world_size: int,
    console,
    status_update: callable,
    timeout: float,
) -> list:
    """Start multi-GPU backend with Coordinator + All Workers architecture.

    Main Process (Coordinator):
        - Only runs the Rust server
        - Does NOT participate in torch.distributed
        - Does NOT load any model

    Worker Processes (0..world_size-1):
        - All run identical code
        - Each participates in torch.distributed
        - Each loads model on its assigned GPU
        - Each connects to IPC server

    Returns:
        List where first element is ServerHandle, rest are worker processes
    """

    status_update(f"Initializing multi-GPU backend ({world_size} devices)...")

    import torch.multiprocessing as mp
    from . import _pie
    from . import path as pie_path

    # Use 'spawn' context for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    # Generate master port
    master_port = 29500 + random.randint(0, 1000)

    # Build config dict for all ranks
    full_config = _build_backend_config(
        engine_config, model_config, authorized_users_path
    )

    # Determine Tensor Parallel degree and topology
    tp_degree = model_config.get(
        "tensor_parallel_size", engine_config.get("tensor_parallel_size")
    )
    if tp_degree is None:
        # Default to world_size (TP across all devices) to avoid OOM by default
        # If users want DP, they should explicitly set tensor_parallel_size=1
        tp_degree = world_size
        if console:
            console.print(
                f"[yellow]![/yellow] tensor_parallel_size not set, defaulting to {tp_degree} (use all GPUs)"
            )

    group_topology = _calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    status_update(f"  Topology: {num_groups} groups (TP={tp_degree})")

    # Phase 1: Start server and get IPC server names for ALL groups
    # Server starts immediately, workers will connect later
    partial_handle, ipc_server_names = _pie.start_server_phase1(
        server_config, authorized_users_path, num_groups
    )

    # Create ready queue for workers to signal when they've connected to IPC
    spawn_ctx = mp.get_context("spawn")
    ready_queue = spawn_ctx.Queue()

    # Spawn ALL worker processes (ranks 0..world_size-1)
    # All workers run identical code - just with different rank/device
    ctx = mp.spawn(
        _ipc_worker_process,
        args=(
            world_size,
            devices,
            master_port,
            full_config,
            group_topology,
            ipc_server_names,
            ready_queue,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
        daemon=True,  # Workers die automatically when main process exits
    )

    # Wait for ALL workers to signal they've connected to IPC
    # Each worker sends its rank when ready
    # Wait for ALL workers to signal they've connected to IPC
    # Monitor processes while waiting to catch early exits (e.g. OOM)
    connected_ranks = set()
    start_wait = time.time()

    # We need to wait for world_size ranks
    while len(connected_ranks) < world_size:
        # 1. Check if processes are alive to catch early exits (e.g. OOM)
        for p in ctx.processes:
            if not p.is_alive():
                exitcode = p.exitcode
                if exitcode != 0:
                    raise RuntimeError(
                        f"Worker process {p.pid} died unexpectedly with exit code {exitcode}"
                    )

        # 2. Check for timeout
        if time.time() - start_wait > timeout:
            # Clean up
            ready_queue.close()
            ready_queue.join_thread()
            raise TimeoutError(f"Timed out waiting for {world_size} workers to connect")

        # 3. Try access queue
        try:
            # Use non-blocking get
            rank = ready_queue.get(timeout=0.2)
            connected_ranks.add(rank)
            if console:
                status_update(
                    f"  Worker {rank} ready ({len(connected_ranks)}/{world_size})"
                )
        except queue.Empty:
            continue

    # Clean up the ready_queue to prevent semaphore leak
    ready_queue.close()
    ready_queue.join_thread()

    # Phase 2: Complete initialization (blocks until handshake succeeds)
    # All workers are now connected via IPC
    server_handle = partial_handle.complete()

    # Return server handle and worker context
    return [server_handle, ctx]


def _calculate_topology(world_size: int, tp_degree: int) -> list[list[int]]:
    """Calculate process groups based on world size and TP degree.

    Returns:
        List of groups, where each group is a list of ranks.
        Example: world_size=4, tp=2 -> [[0, 1], [2, 3]]
    """
    if world_size % tp_degree != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by TP degree ({tp_degree})"
        )

    num_groups = world_size // tp_degree
    topology = []
    for g in range(num_groups):
        group_ranks = list(range(g * tp_degree, (g + 1) * tp_degree))
        topology.append(group_ranks)

    return topology


def _ipc_worker_process(
    local_rank: int,  # mp.spawn passes 0-indexed local rank (= actual rank for us)
    world_size: int,
    devices: list[str],
    master_port: int,
    config_dict: dict,
    group_topology: list[list[int]],
    ipc_server_names: list[str],
    ready_queue,  # Queue to signal when connected
):
    """Worker process for Coordinator + All Workers architecture.

    All workers run this identical code path. Each worker:
    1. Initializes torch.distributed
    2. Loads model on its assigned GPU
    3. Connects to the IPC server for its group
    4. Signals ready via ready_queue
    5. Runs the IPC worker loop

    Args:
        local_rank: Rank of this worker (0 to world_size-1)
        world_size: Total number of workers
        devices: List of device strings
        master_port: Port for torch.distributed rendezvous
        config_dict: Runtime configuration
        group_topology: List of groups, each containing ranks
        ipc_server_names: IPC server names for each group
        ready_queue: Queue to signal when ready
    """
    from pie import _pie
    from pie_worker.runtime import Runtime
    from pie_worker.config import RuntimeConfig
    import torch.distributed as dist

    rank = local_rank  # With nprocs=world_size, local_rank IS the actual rank

    # Determine my group and TP rank within it
    my_group_id = 0
    tp_rank = 0
    for i, group in enumerate(group_topology):
        if rank in group:
            my_group_id = i
            tp_rank = group.index(rank)  # My position within the TP group
            break

    tp_degree = len(group_topology[my_group_id])  # Number of GPUs in my TP group

    # Initialize distributed
    _init_distributed(rank, world_size, master_port, devices[rank])

    # Setup process groups (collective ops - all ranks must participate)
    # Capture the mappings to pass to Runtime
    pg_map = _setup_process_groups(rank, group_topology)
    compute_pg_map = _setup_compute_process_groups(rank, group_topology)

    # Create runtime config
    # For TP>1: each worker needs ALL devices in its TP group so devices[tp_rank] works
    # Get the device list for this TP group (e.g., ["cuda:0", "cuda:1"] for group 0)
    my_group_ranks = group_topology[my_group_id]
    group_devices = [devices[r] for r in my_group_ranks]

    filtered_config = {
        k: v for k, v in config_dict.items() if k not in ("device", "devices")
    }
    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=group_devices,  # All devices in this TP group
        rank=tp_rank,  # Position within TP group
        world_size=tp_degree,  # Size of TP group
        tensor_parallel_size=tp_degree,  # Ensure sharding is enabled!
    )

    # Create runtime (loads model on this GPU)
    # Pass process groups for TP communication
    runtime = Runtime(
        config,
        group_id=my_group_id,
        process_groups=pg_map,
        compute_process_groups=compute_pg_map,
        group_topology=group_topology,
    )

    # Sync all workers before connecting to server
    dist.barrier()

    # Check if I'm a group leader (first rank in my TP group)
    is_group_leader = tp_rank == 0

    try:
        if is_group_leader:
            # Group leader: connect to IPC and handle requests
            server_name = ipc_server_names[my_group_id]
            ipc_queue = _pie.FfiIpcQueue.connect(server_name, my_group_id)

            # Signal that we're connected and ready
            ready_queue.put(rank)

            # Run IPC worker loop (handles requests from Rust server)
            _run_ipc_worker_loop(ipc_queue, runtime)
        else:
            # Non-leader: signal ready, then run worker loop waiting for commands from leader
            ready_queue.put(rank)
            runtime.worker_loop()
    finally:
        # Cleanup - ensure process group is destroyed even on termination
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_ipc_worker_loop(ipc_queue, runtime):
    """Run the IPC worker loop for a group leader.

    This is similar to the FFI worker loop in server.py, but uses IPC for
    cross-process communication. This allows each group leader to have its
    own Python GIL, eliminating GIL contention between groups.

    Args:
        ipc_queue: FfiIpcQueue instance connected to Rust
        runtime: Runtime instance to dispatch calls to
    """
    import msgpack
    from pie_worker.server import (
        STATUS_OK,
        STATUS_METHOD_NOT_FOUND,
        STATUS_INTERNAL_ERROR,
    )

    # Method dispatch table
    methods = {
        "handshake": runtime.handshake_rpc,
        "query": runtime.query_rpc,
        "fire_batch": runtime.fire_batch,
        "embed_image": runtime.embed_image_rpc,
        "initialize_adapter": runtime.initialize_adapter_rpc,
        "update_adapter": runtime.update_adapter_rpc,
        "upload_adapter": runtime.upload_adapter_rpc,
        "download_adapter": runtime.download_adapter_rpc,
    }

    shutdown_requested = False
    poll_timeout_ms = 100
    parent_pid = os.getppid()
    check_parent_every = 10  # Check parent alive every N poll cycles
    poll_count = 0

    try:
        while not shutdown_requested:
            # Check if parent process is still alive periodically
            poll_count += 1
            if poll_count % check_parent_every == 0:
                try:
                    os.kill(parent_pid, 0)  # Doesn't kill, just checks existence
                except OSError:
                    # Parent process has exited, we should exit too
                    break

            try:
                # Poll the IPC queue (releases GIL while waiting)
                request = ipc_queue.poll_blocking(poll_timeout_ms)
                if request is None:
                    continue  # Timeout, try again
            except Exception:
                # IPC queue may be closed when server shuts down
                break

            request_id, method, payload = request

            try:
                # Unpack args
                args = msgpack.unpackb(payload)

                # Check for shutdown
                if method == "shutdown":
                    shutdown_requested = True
                    response = msgpack.packb(None)
                    ipc_queue.respond(request_id, response)
                    continue

                # Get handler
                fn = methods.get(method)
                if fn is None:
                    response = msgpack.packb(f"Method not found: {method}")
                    ipc_queue.respond(request_id, response)
                    continue

                # Call handler
                if isinstance(args, dict):
                    result = fn(**args)
                elif isinstance(args, (list, tuple)):
                    result = fn(*args)
                else:
                    result = fn(args)

                # Pack and respond
                response = msgpack.packb(result)
                ipc_queue.respond(request_id, response)

            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                print(f"[IPC Worker Error] {method}: {e}\n{tb}")
                response = msgpack.packb(str(e))
                ipc_queue.respond(request_id, response)
    finally:
        # Ensure cleanup when loop stops
        runtime.shutdown()


def _ipc_group_worker(
    server_name: str,
    group_id: int,
    config_dict: dict,
    device: str,
):
    """IPC worker process for symmetric all-IPC architecture.

    This runs in a separate subprocess with its own GIL.
    It loads the model on the specified device and connects to the IPC server.

    Args:
        server_name: IPC server name to connect to
        group_id: Group ID for this worker
        config_dict: Runtime configuration
        device: Device string (e.g., "cuda:0")
    """
    from pie import _pie
    from pie_worker.runtime import Runtime
    from pie_worker.config import RuntimeConfig

    # Create runtime config for this group
    filtered_config = {
        k: v for k, v in config_dict.items() if k not in ("device", "devices")
    }

    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=[device],
        rank=0,  # Local rank in this process
        world_size=1,  # Single GPU in this process
    )

    # Create runtime (loads model on this GPU)
    runtime = Runtime(config, group_id=group_id)

    # Connect to IPC and run worker loop
    ipc_queue = _pie.FfiIpcQueue.connect(server_name, group_id)
    _run_ipc_worker_loop(ipc_queue, runtime)


def wait_for_backends(
    server_handle: "_pie.ServerHandle",
    expected_count: int,
    timeout: float,
    backend_processes: list,
) -> bool:
    """Wait for the expected number of backends to register with the engine.

    Args:
        server_handle: The server handle to query
        expected_count: Number of backends we expect to connect
        timeout: Maximum time to wait in seconds
        backend_processes: List of backend processes to check for early exit

    Returns:
        True if all backends connected, False if timeout
    """
    start_time = time.time()
    poll_interval = 0.5  # seconds

    while time.time() - start_time < timeout:
        # Check if the engine is still running
        if not server_handle.is_running():
            print("❌ Engine stopped unexpectedly", file=sys.stderr)
            return False

        # Check if any backend process has died
        if not check_backend_processes(backend_processes):
            return False

        # Check registered models
        models = server_handle.registered_models()
        if len(models) >= expected_count:
            return True

        time.sleep(poll_interval)

    return False


def check_backend_processes(
    backend_processes: list, on_error: Optional[callable] = None
) -> bool:
    """Check if all backend processes are still alive.

    Args:
        backend_processes: List of backend processes to check
        on_error: Optional callback for error messages: (message: str) -> None

    Returns:
        True if all processes are alive, False if any have died
    """

    all_alive = True
    for process in backend_processes:
        # In FFI mode, backend_processes may contain dispatcher functions (not processes)
        # Skip anything that's not a process
        if not hasattr(process, "pid") or not hasattr(process, "is_alive"):
            continue

        is_dead = False
        return_code = None
        stderr = ""

        if isinstance(process, subprocess.Popen):
            if process.poll() is not None:
                is_dead = True
                return_code = process.returncode
                stderr = process.stderr.read().decode() if process.stderr else ""
        else:
            # Assume multiprocessing.Process
            if not process.is_alive():
                is_dead = True
                return_code = process.exitcode
                # Check for SpawnContext
                if hasattr(process, "processes"):
                    # For SpawnContext, is_alive checks if any process is alive
                    # Context is "dead" if all processes are dead
                    pass

        if is_dead:
            all_alive = False
            error_msg = f"Backend process exited unexpectedly (exit code {return_code})"
            if stderr:
                error_msg += f" stderr: {stderr[:500]}"
            if on_error:
                on_error(error_msg)
            else:
                print(f"❌ {error_msg}", file=sys.stderr)

    return all_alive


def terminate_engine_and_backend(
    server_handle: "_pie.ServerHandle | None",
    backend_processes: list,
    on_message: Optional[callable] = None,
) -> None:
    """Terminate the engine and backend processes.

    Args:
        server_handle: The server handle (or None if already shut down)
        backend_processes: List of backend subprocess.Popen objects
        on_message: Optional callback for status messages: (message: str) -> None
    """
    # Suppress semaphore leak warning during shutdown (cosmetic, happens when workers are killed)
    warnings.filterwarnings(
        "ignore", message=".*leaked semaphore.*", category=UserWarning
    )

    from pie_worker import utils as pie_utils

    def log(msg: str):
        if on_message:
            on_message(msg)
        else:
            # sys imported globally
            pass
            # print(f"[Manager] {msg}", file=sys.stderr)

    # 1. Shut down the server FIRST - this sends shutdown signal to workers via IPC
    if server_handle is not None:
        try:
            if server_handle.is_running():
                server_handle.shutdown()
        except Exception as e:
            log(f"Error shutting down engine: {e}")

    # 2. Give workers time to shut down gracefully after receiving IPC shutdown
    time.sleep(1.0)

    # 3. Broadcast STOP signal via control channel (legacy, may not be in use)
    try:
        if pie_utils._control_channel is not None:
            pie_utils._control_channel.send("STOP")
            time.sleep(0.5)
    except ImportError:
        pass
    except Exception as e:
        log(f"Error sending STOP signal: {e}")

    for process in backend_processes:
        # Check for SpawnContext (multiprocessing spawn context)
        # It doesn't have a pid attribute directly, but has .processes and .join
        if hasattr(process, "join") and hasattr(process, "processes"):
            try:
                # Terminate all worker processes in the spawn context
                for p in process.processes:
                    if p.is_alive():
                        p.terminate()
                # Wait briefly for termination
                for p in process.processes:
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()  # Force kill if still alive
                # Finally join the context itself
                process.join(timeout=1)
            except Exception as e:
                log(f"Error terminating SpawnContext: {e}")
            continue

        # In FFI mode, backend_processes may contain dispatcher functions (not processes)
        # Skip anything that's not a process or context
        if not hasattr(process, "pid"):
            continue

        is_running = False
        pid = process.pid

        if isinstance(process, subprocess.Popen):
            is_running = process.poll() is None
        else:
            is_running = process.is_alive()

        if is_running:
            try:
                if isinstance(process, subprocess.Popen):
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)
                else:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        raise subprocess.TimeoutExpired(cmd=str(pid), timeout=5)
            except subprocess.TimeoutExpired:
                log(f"Force killing process {pid}")
                process.kill()
            except Exception as e:
                log(f"Error terminating process {pid}: {e}")

    # Finalize control channel queues if they exist
    # This prevents "leaked semaphore" warnings from multiprocessing.resource_tracker
    try:
        if pie_utils._control_channel is not None:
            pie_utils._control_channel.cleanup()
            pie_utils._control_channel = None

    except ImportError:
        pass  # pie_worker might not be installed or importable
    except Exception as e:
        log(f"Error cleaning up control channel: {e}")


def run_interactive_shell(engine_config: dict, internal_token: str) -> None:
    """Run the interactive shell session.

    Args:
        engine_config: Engine configuration dict
        internal_token: Internal authentication token
    """
    # Simple readline-based shell
    try:
        import readline
    except ImportError:
        pass  # readline not available on all platforms

    from . import path as pie_path

    # Load history
    history_path = pie_path.get_shell_history_path()
    try:
        if history_path.exists():
            readline.read_history_file(str(history_path))
    except (OSError, NameError):
        pass

    client_config = {
        "host": engine_config.get("host", "127.0.0.1"),
        "port": engine_config.get("port", 8080),
        "internal_auth_token": internal_token,
    }

    print("Available commands:")
    print("  run <path> [ARGS]... - Run a .wasm inferlet with optional arguments")
    print("  stat                 - Query the backend statistics")
    print("  exit                 - Exit the Pie session")
    print("  help                 - Show this help message")

    while True:
        try:
            line = input("pie> ")
        except EOFError:
            print("Exiting...")
            break
        except KeyboardInterrupt:
            print("\n(To exit, type 'exit' or press Ctrl-D)")
            continue

        line = line.strip()
        if not line:
            continue

        parts = line.split()
        command = parts[0]
        args = parts[1:]

        if command == "exit":
            print("Exiting...")
            break
        elif command == "help":
            print("Available commands:")
            print("  run <path> [ARGS]... - Run a .wasm inferlet")
            print("  stat                 - Query backend statistics")
            print("  exit                 - Exit the session")
            print("  help                 - Show this help message")
        elif command == "run":
            if not args:
                print("Usage: run <inferlet_path> [ARGS]...")
                continue
            inferlet_path = Path(args[0]).expanduser()
            inferlet_args = args[1:]
            try:
                submit_inferlet_and_wait(client_config, inferlet_path, inferlet_args)
            except Exception as e:
                print(f"Error running inferlet: {e}")
        elif command == "stat":
            print("(stat command not yet implemented)")
        else:
            print(f"Unknown command: '{command}'. Type 'help' for a list of commands.")

    # Save history
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(history_path))
    except (OSError, NameError):
        pass


def submit_inferlet_and_wait(
    client_config: dict,
    inferlet_path: Path,
    arguments: list[str],
    server_handle: "_pie.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Submit an inferlet to the engine and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_path: Path to the .wasm inferlet file
        arguments: Arguments to pass to the inferlet
        server_handle: Optional server handle for process monitoring
        backend_processes: Optional list of backend processes to monitor
    """
    asyncio.run(
        _submit_inferlet_async(
            client_config,
            inferlet_path,
            arguments,
            server_handle,
            backend_processes,
            on_event,
        )
    )


async def _submit_inferlet_async(
    client_config: dict,
    inferlet_path: Path,
    arguments: list[str],
    server_handle: "_pie.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Async implementation of submit_inferlet_and_wait."""

    import blake3
    from pie_client import PieClient, Event

    def emit(event_type: str, msg: str):
        if on_event:
            on_event(event_type, msg)
        else:
            print(msg)

    # Check inferlet exists
    if not inferlet_path.exists():
        raise FileNotFoundError(f"Inferlet not found: {inferlet_path}")

    # Read and hash the inferlet
    inferlet_blob = inferlet_path.read_bytes()
    program_hash = blake3.blake3(inferlet_blob).hexdigest()
    emit("info", f"Inferlet hash: {program_hash}")

    # Build the WebSocket URI
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    internal_token = client_config.get("internal_auth_token")
    server_uri = f"ws://{host}:{port}"

    # Start monitoring task if processes provided
    monitor_task = None
    if backend_processes:
        monitor_task = asyncio.create_task(
            _monitor_processes_task(server_handle, backend_processes)
        )

    try:
        async with PieClient(server_uri) as client:
            # Authenticate with internal token
            await client.internal_authenticate(internal_token)

            # Check if program already exists, upload if not
            if not await client.program_exists(program_hash):
                emit("info", "Uploading inferlet...")
                await client.upload_program(inferlet_blob)
            else:
                emit("info", "Inferlet already cached on server.")

            # Launch the instance
            emit("info", f"Launching {inferlet_path.name}...")
            instance = await client.launch_instance(
                program_hash=program_hash,
                arguments=arguments,
                detached=False,
            )
            emit("info", f"Instance launched: {instance.instance_id}")

            # Stream events until completion
            while True:
                recv_task = asyncio.create_task(instance.recv())
                tasks = [recv_task]
                if monitor_task:
                    tasks.append(monitor_task)

                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                if monitor_task in done:
                    monitor_task.result()

                event, message = recv_task.result()

                if event == Event.Stdout:
                    print(message, end="", flush=True)
                elif event == Event.Stderr:
                    print(message, end="", file=sys.stderr, flush=True)
                elif event == Event.Message:
                    emit("message", f"[Message] {message}")
                elif event == Event.Completed:
                    emit("completed", f"{message}")
                    break
                elif event == Event.Aborted:
                    emit("aborted", f"⚠️ Instance aborted: {message}")
                    break
                elif event == Event.Exception:
                    emit("exception", f"❌ Instance exception: {message}")
                    break
                elif event == Event.ServerError:
                    emit("error", f"❌ Server error: {message}")
                    break
                elif event == Event.OutOfResources:
                    emit("error", f"❌ Out of resources: {message}")
                    break
                elif event == Event.Blob:
                    emit("blob", f"[Received blob: {len(message)} bytes]")
                else:
                    emit("unknown", f"[Unknown event {event}]: {message}")

    finally:
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass


async def _monitor_processes_task(
    server_handle: "_pie.ServerHandle | None",
    backend_processes: list | None,
):
    """Async task to monitor backend processes."""
    import asyncio

    if not backend_processes:
        return

    while True:
        if not check_backend_processes(backend_processes):
            raise RuntimeError("Backend process died")

        if server_handle and hasattr(server_handle, "is_running"):
            if not server_handle.is_running():
                raise RuntimeError("Engine process died")

        await asyncio.sleep(1.0)


def submit_inferlet_from_registry_and_wait(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle: "_pie.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Submit an inferlet from the registry and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_name: Inferlet name (e.g., "std/text-completion@0.1.0")
        arguments: Arguments to pass to the inferlet
        server_handle: Optional server handle for process monitoring
        backend_processes: Optional list of backend processes to monitor
        on_event: Optional callback for events: (event_type: str, message: str) -> None
    """
    import asyncio

    asyncio.run(
        _submit_inferlet_from_registry_async(
            client_config,
            inferlet_name,
            arguments,
            server_handle,
            backend_processes,
            on_event,
        )
    )


async def _submit_inferlet_from_registry_async(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle: "_pie.ServerHandle | None" = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Async implementation of submit_inferlet_from_registry_and_wait."""
    import asyncio
    from pie_client import PieClient, Event

    def emit(event_type: str, msg: str):
        if on_event:
            on_event(event_type, msg)
        else:
            print(msg)

    # Build the WebSocket URI
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    internal_token = client_config.get("internal_auth_token")
    server_uri = f"ws://{host}:{port}"

    # Start monitoring task if processes provided
    monitor_task = None
    if backend_processes:
        monitor_task = asyncio.create_task(
            _monitor_processes_task(server_handle, backend_processes)
        )

    try:
        async with PieClient(server_uri) as client:
            # Authenticate with internal token
            await client.internal_authenticate(internal_token)

            # Launch the instance from registry
            instance = await client.launch_instance_from_registry(
                inferlet=inferlet_name,
                arguments=arguments,
                detached=False,
            )

            # Stream events until completion
            while True:
                recv_task = asyncio.create_task(instance.recv())
                tasks = [recv_task]
                if monitor_task:
                    tasks.append(monitor_task)

                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                if monitor_task in done:
                    monitor_task.result()

                event, message = recv_task.result()

                if event == Event.Stdout:
                    print(message, end="", flush=True)
                elif event == Event.Stderr:
                    print(message, end="", file=sys.stderr, flush=True)
                elif event == Event.Message:
                    emit("message", f"[Message] {message}")
                elif event == Event.Completed:
                    emit("completed", f"{message}")
                    break
                elif event == Event.Aborted:
                    emit("aborted", f"⚠️ Instance aborted: {message}")
                    break
                elif event == Event.Exception:
                    emit("exception", f"❌ Instance exception: {message}")
                    break
                elif event == Event.ServerError:
                    emit("error", f"❌ Server error: {message}")
                    break
                elif event == Event.OutOfResources:
                    emit("error", f"❌ Out of resources: {message}")
                    break
                elif event == Event.Blob:
                    emit("blob", f"[Received blob: {len(message)} bytes]")
                else:
                    emit("unknown", f"[Unknown event {event}]: {message}")
    except Exception:
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
        raise
