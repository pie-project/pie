"""Worker process for Pie multi-GPU backend.

This module contains everything that runs inside a spawned child process:
topology calculation, torch.distributed initialization, process group setup,
and the worker entry point that creates the Backend and RPC server.
"""

import warnings


# =============================================================================
# Topology
# =============================================================================


def calculate_topology(world_size: int, tp_degree: int) -> list[list[int]]:
    """Calculate process group topology from world size and TP degree.

    Args:
        world_size: Total number of worker processes
        tp_degree: Tensor parallel degree (GPUs per model replica)

    Returns:
        List of groups, each a list of ranks.
        Example: world_size=4, tp=2 → [[0, 1], [2, 3]]

    Raises:
        ValueError: If world_size is not divisible by tp_degree
    """
    if world_size % tp_degree != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by TP degree ({tp_degree})"
        )

    num_groups = world_size // tp_degree
    return [
        list(range(g * tp_degree, (g + 1) * tp_degree))
        for g in range(num_groups)
    ]


# =============================================================================
# Distributed Initialization
# =============================================================================


def _init_distributed(rank: int, world_size: int, master_port: int, device: str):
    """Initialize torch.distributed for a given rank.

    Sets up CUDA device and process group using FileStore for rendezvous.
    """
    import datetime
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(device)

    # Suppress harmless barrier warnings
    warnings.filterwarnings(
        "ignore", message=".*barrier.*device under current context.*"
    )

    # FileStore for robust rendezvous (avoids port conflicts)
    store = dist.FileStore(f"/tmp/pie_dist_store_{master_port}", world_size)
    timeout = datetime.timedelta(seconds=300)

    backend = "nccl" if torch.cuda.is_available() else "gloo"

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


def _setup_process_groups(group_topology: list[list[int]]) -> dict:
    """Create ProcessGroups for each execution group (Rank 0 + Group Workers)."""
    import torch.distributed as dist

    pg_map = {}
    for i, group_ranks in enumerate(group_topology):
        comm_ranks = sorted(set([0] + group_ranks))
        pg_map[i] = dist.new_group(comm_ranks)
    return pg_map


def _setup_compute_process_groups(group_topology: list[list[int]]) -> dict:
    """Create ProcessGroups for Tensor Parallel computation (TP ranks only)."""
    import torch.distributed as dist

    pg_map = {}
    for i, group_ranks in enumerate(group_topology):
        comm_ranks = sorted(set(group_ranks))
        pg_map[i] = dist.new_group(comm_ranks)
    return pg_map


# =============================================================================
# Worker Entry Point
# =============================================================================


def worker_main(
    local_rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,
    model_config: dict,
    group_topology: list[list[int]],
    ready_queue,
):
    """Worker process entry point for mp.spawn.

    Each worker:
    1. Computes its group membership and TP rank
    2. Initializes torch.distributed
    3. Sets up process groups
    4. Creates RuntimeConfig + Backend
    5. Group leaders: create RpcServer, report server_name via ready_queue
       Non-leaders: report ready, run worker_loop

    Args:
        local_rank: Rank of this worker (0 to world_size-1)
        world_size: Total number of workers
        devices: List of device strings (one per rank)
        master_port: Port for torch.distributed rendezvous
        model_config: Model configuration dict (passed to RuntimeConfig.from_args)
        group_topology: List of groups, each containing ranks
        ready_queue: Queue to signal readiness: (rank, server_name|None, metadata|None)
    """
    import pie_runtime
    from pie_backend.backend import Backend
    from pie_backend.config import RuntimeConfig
    from pie_backend.server import poll_rpc_server
    import torch
    import torch.distributed as dist
    import threading

    rank = local_rank

    # — Determine group membership —
    my_group_id = 0
    tp_rank = 0
    for i, group in enumerate(group_topology):
        if rank in group:
            my_group_id = i
            tp_rank = group.index(rank)
            break

    tp_degree = len(group_topology[my_group_id])

    # — Initialize distributed —
    if world_size > 1:
        _init_distributed(rank, world_size, master_port, devices[rank])
    else:
        device_str = devices[rank]
        if device_str.startswith("cuda"):
            torch.cuda.set_device(device_str)

    # — Setup process groups —
    if world_size > 1:
        pg_map = _setup_process_groups(group_topology)
        compute_pg_map = _setup_compute_process_groups(group_topology)
    else:
        pg_map = {}
        compute_pg_map = {}

    # — Create runtime config —
    group_devices = [devices[r] for r in group_topology[my_group_id]]

    # Pass model_config directly — RuntimeConfig.from_args() owns all defaults.
    # Filter out device/devices keys since we pass them explicitly.
    filtered_config = {
        k: v for k, v in model_config.items()
        if k not in ("device", "devices", "scheduler")
    }

    config = RuntimeConfig.from_args(
        **filtered_config,
        devices=group_devices,
        rank=tp_rank,
        world_size=tp_degree,
        tensor_parallel_size=tp_degree,
    )

    # — Create backend (loads model on this GPU) —
    runtime = Backend(
        config,
        group_id=my_group_id,
        process_groups=pg_map,
        compute_process_groups=compute_pg_map,
        group_topology=group_topology,
    )

    # Sync all workers before signaling ready
    if dist.is_initialized():
        dist.barrier()

    is_group_leader = tp_rank == 0

    try:
        if is_group_leader:
            # Create RPC server for Rust to connect to
            server = pie_runtime.RpcServer.create()
            server_name = server.server_name()

            chat_template_info = runtime.get_chat_template()
            metadata = {
                "total_pages": getattr(runtime, "total_pages", 0),
                "max_batch_tokens": getattr(runtime, "max_batch_tokens", 10240),
                "max_batch_size": getattr(runtime, "max_batch_size", 128),
                "chat_template": chat_template_info.get("template_content", ""),
                "stop_tokens": chat_template_info.get("stop_tokens", []),
            }

            ready_queue.put((rank, server_name, metadata))

            # Run RPC loop — Rust connects as client
            stop_event = threading.Event()
            poll_rpc_server(server, runtime, stop_event)
        else:
            ready_queue.put((rank, None, None))
            runtime.worker_loop()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
