"""Worker process for Pie multi-GPU backend.

This module contains everything that runs inside a spawned child process:
topology calculation, torch.distributed initialization, process group setup,
and the two worker roles:
- Group leaders run the RPC loop (receive from Rust, broadcast to TP peers)
- Followers wait for broadcasts and run inference
"""

from __future__ import annotations

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
    4. Creates RuntimeConfig + Engine
    5. Group leaders: create RpcServer, report server_name via ready_queue
       Non-leaders: report ready, run follower loop

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
    from pie_backend.engine import Engine
    from pie_backend.config import RuntimeConfig
    from pie_backend.model import get_chat_template
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

    # — Load engine (loads model on this GPU) —
    compute_pg = compute_pg_map.get(my_group_id)
    engine = Engine.load(config, compute_process_group=compute_pg)

    # Sync all workers before signaling ready
    if dist.is_initialized():
        dist.barrier()

    is_group_leader = tp_rank == 0

    try:
        if is_group_leader:
            # Create RPC server for Rust to connect to
            server = pie_runtime.RpcServer.create()
            server_name = server.server_name()

            chat_template_info = engine.chat_template()
            metadata = {
                "total_pages": getattr(config, "max_num_kv_pages", 0),
                "max_batch_tokens": getattr(config, "max_batch_tokens", 10240),
                "max_batch_size": getattr(config, "max_batch_size", 128),
                "chat_template": chat_template_info.get("template_content", ""),
                "stop_tokens": chat_template_info.get("stop_tokens", []),
            }

            ready_queue.put((rank, server_name, metadata))

            # Run RPC loop — Rust connects as client
            stop_event = threading.Event()
            _leader_loop(
                engine=engine,
                server=server,
                stop_event=stop_event,
                compute_pg=compute_pg,
                group_topology=group_topology,
                group_id=my_group_id,
            )
        else:
            ready_queue.put((rank, None, None))
            _follower_loop(
                engine=engine,
                compute_pg=compute_pg,
                leader_rank=group_topology[my_group_id][0],
                group_id=my_group_id,
                config=config,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# Leader Loop (absorbs server.py + fire_batch from backend.py)
# =============================================================================

# RPC status codes (must match Rust)
_STATUS_OK = 0
_STATUS_METHOD_NOT_FOUND = 1
_STATUS_INVALID_PARAMS = 2
_STATUS_INTERNAL_ERROR = 3


def _leader_loop(
    engine,
    server,
    stop_event,
    compute_pg,
    group_topology: list[list[int]],
    group_id: int,
    poll_timeout_ms: int = 100,
) -> None:
    """RPC loop for group leaders.

    Polls the RPC server for requests from Rust, dispatches to handlers.
    Handles fire_batch by building Batch, broadcasting to TP followers,
    running engine.fire_batch, and packaging responses.
    """
    import time
    import msgpack
    import torch.distributed as dist
    from pie_backend import utils, message
    from pie_backend.batching import Batch
    from pie_backend.latency import StepTiming, LatencyStats

    config = engine.config
    latency_stats = LatencyStats(enabled=config.telemetry_enabled)

    def _handle_query(**kwargs) -> dict:
        req = message.QueryRequest(**kwargs)
        value = engine.query(req.query)
        return {"value": value}

    def _handle_embed_image(**kwargs) -> None:
        # TODO: implement image embedding
        pass

    def _handle_init_adapter(**kwargs) -> None:
        req = message.InitializeAdapterRequest(**kwargs)
        args = {
            "adapter_ptr": req.adapter_ptr,
            "rank": req.rank,
            "alpha": req.alpha,
            "population_size": req.population_size,
            "mu_fraction": req.mu_fraction,
            "initial_sigma": req.initial_sigma,
        }
        # Broadcast to TP followers
        if config.world_size > 1 and compute_pg is not None:
            leader_global_rank = group_topology[group_id][0]
            utils.broadcast_struct(
                {"type": "INIT_ADAPTER", "kwargs": args},
                src=leader_global_rank,
                device=config.device,
                group=compute_pg,
            )
        engine.init_adapter(**args)

    def _handle_update_adapter(**kwargs) -> None:
        req = message.UpdateAdapterRequest(**kwargs)
        args = {
            "adapter_ptr": req.adapter_ptr,
            "scores": req.scores,
            "seeds": req.seeds,
            "max_sigma": req.max_sigma,
        }
        if config.world_size > 1 and compute_pg is not None:
            leader_global_rank = group_topology[group_id][0]
            utils.broadcast_struct(
                {"type": "UPDATE_ADAPTER", "kwargs": args},
                src=leader_global_rank,
                device=config.device,
                group=compute_pg,
            )
        engine.update_adapter(**args)

    def _handle_load_adapter(**kwargs) -> None:
        req = message.LoadAdapterRequest(**kwargs)
        data = req.adapter_data
        if isinstance(data, list):
            data = bytes(data)
        args = {"adapter_ptr": req.adapter_ptr, "name": req.name, "data": data}
        if config.world_size > 1 and compute_pg is not None:
            leader_global_rank = group_topology[group_id][0]
            utils.broadcast_struct(
                {"type": "LOAD_ADAPTER", "kwargs": args},
                src=leader_global_rank,
                device=config.device,
                group=compute_pg,
            )
        engine.load_adapter(**args)

    def _handle_save_adapter(**kwargs) -> None:
        req = message.SaveAdapterRequest(**kwargs)
        args = {"adapter_ptr": req.adapter_ptr, "name": req.name}
        if config.world_size > 1 and compute_pg is not None:
            leader_global_rank = group_topology[group_id][0]
            utils.broadcast_struct(
                {"type": "SAVE_ADAPTER", "kwargs": args},
                src=leader_global_rank,
                device=config.device,
                group=compute_pg,
            )
        engine.save_adapter(**args)

    def _handle_fire_batch(**kwargs) -> dict:
        t_start = time.perf_counter()

        # Build batch
        t0 = time.perf_counter()
        batch = Batch(
            kwargs,
            config.kv_page_size,
            config.max_dist_size,
            engine.adapters,
            vocab_size=getattr(
                engine.model_config,
                "num_vocabs",
                getattr(engine.model_config, "vocab_size", 128000),
            ),
        )
        build_timing = batch.timing
        t_build_batch = time.perf_counter() - t0

        device = config.device

        # Create GPU tensors
        t0 = time.perf_counter()
        inputs = batch.get_model_inputs(device)
        t_get_inputs = time.perf_counter() - t0

        t0 = time.perf_counter()
        sampling_metadata = batch.get_sampling_metadata(
            device, config.activation_dtype
        )
        t_get_sampling_meta = time.perf_counter() - t0

        # Broadcast to TP followers if multi-GPU
        t0 = time.perf_counter()
        should_broadcast = (
            config.world_size > 1
            and config.rank == 0
            and compute_pg is not None
        )
        if should_broadcast:
            leader_global_rank = group_topology[group_id][0]
            utils.broadcast_struct(
                {
                    "type": "STEP",
                    "inputs": inputs,
                    "sampling_metadata": sampling_metadata,
                },
                src=leader_global_rank,
                device=device,
                group=compute_pg,
                group_id=group_id,
            )
        t_broadcast = time.perf_counter() - t0

        # TP barrier
        if config.world_size > 1 and compute_pg is not None:
            if dist.get_world_size(group=compute_pg) > 1:
                dist.barrier(group=compute_pg)

        # Execute inference
        t0 = time.perf_counter()
        sampling_results = engine.fire_batch(inputs, sampling_metadata)
        t_inference = time.perf_counter() - t0

        # Package responses
        t0 = time.perf_counter()
        responses = batch.create_responses(sampling_results)
        results = [
            {"tokens": resp.tokens, "dists": resp.dists}
            for resp in responses
        ]
        t_create_responses = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        # Record latency stats
        latency_stats.record_span(
            StepTiming(
                build_batch=t_build_batch,
                get_inputs=t_get_inputs,
                get_sampling_meta=t_get_sampling_meta,
                broadcast=t_broadcast,
                inference=t_inference,
                create_responses=t_create_responses,
                total=t_total,
                decode_u32=build_timing["decode_u32"],
                mask_loop=build_timing["mask_loop"],
                brle_decode=build_timing["brle_decode"],
                sampler_loop=build_timing["sampler_loop"],
            ),
            traceparent=kwargs.get("trace_context"),
        )

        return {"results": results}

    # Method dispatch table
    methods = {
        "query": _handle_query,
        "fire_batch": _handle_fire_batch,
        "embed_image": _handle_embed_image,
        "initialize_adapter": _handle_init_adapter,
        "update_adapter": _handle_update_adapter,
        "load_adapter": _handle_load_adapter,
        "save_adapter": _handle_save_adapter,
    }

    try:
        while not stop_event.is_set():
            # Poll the RPC server (releases GIL while waiting)
            request = server.poll_blocking(poll_timeout_ms)
            if request is None:
                continue  # Timeout, try again

            request_id, method, payload = request

            try:
                args = msgpack.unpackb(payload)

                fn = methods.get(method)
                if fn is None:
                    response = msgpack.packb(f"Method not found: {method}")
                    server.respond(request_id, response)
                    continue

                # Call handler
                if isinstance(args, dict):
                    result = fn(**args)
                elif isinstance(args, (list, tuple)):
                    result = fn(*args)
                else:
                    result = fn(args)

                response = msgpack.packb(result)
                server.respond(request_id, response)

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[RPC Server Error] {method}: {e}\n{tb}")
                response = msgpack.packb(str(e))
                server.respond(request_id, response)
    finally:
        # Shutdown: broadcast STOP to followers
        print("[RPC Worker] Shutting down...")
        if config.world_size > 1 and config.rank == 0 and compute_pg is not None:
            try:
                leader_global_rank = group_topology[group_id][0]
                utils.broadcast_struct(
                    "STOP",
                    src=leader_global_rank,
                    device=config.device,
                    group=compute_pg,
                )
            except Exception:
                pass


# =============================================================================
# Follower Loop (absorbs Backend.worker_loop)
# =============================================================================


def _follower_loop(
    engine,
    compute_pg,
    leader_rank: int,
    group_id: int,
    config=None,
    result_queue=None,
) -> None:
    """Broadcast loop for TP followers.

    Waits for control messages from the group leader and executes
    inference steps or adapter operations.
    """
    import signal
    import torch
    import torch.distributed as dist
    from pie_backend import utils

    device = config.device if config else "cuda:0"
    is_group_leader_of_secondary = (result_queue is not None)

    shutdown_requested = False

    def sigterm_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    while not shutdown_requested:
        # Receive control message from leader
        try:
            msg = utils.broadcast_struct(
                None, src=leader_rank, device=device, group=compute_pg
            )
        except Exception:
            break

        if shutdown_requested:
            break

        if msg == "STOP":
            break

        if isinstance(msg, dict):
            msg_type = msg.get("type")

            if msg_type == "STEP":
                inputs = msg["inputs"]
                sampling_metadata = msg["sampling_metadata"]
                try:
                    # TP barrier
                    if config and config.world_size > 1 and compute_pg is not None:
                        if dist.get_world_size(group=compute_pg) > 1:
                            dist.barrier(group=compute_pg)

                    result = engine.fire_batch(inputs, sampling_metadata)

                    # If secondary group leader, push result to queue
                    if is_group_leader_of_secondary:
                        result_queue.put(result)
                except Exception as e:
                    print(f"Worker {config.rank if config else '?'} fire_batch error: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if is_group_leader_of_secondary:
                        result_queue.put(None)

            elif msg_type == "INIT_ADAPTER":
                engine.init_adapter(**msg["kwargs"])

            elif msg_type == "UPDATE_ADAPTER":
                engine.update_adapter(**msg["kwargs"])

            elif msg_type == "LOAD_ADAPTER":
                engine.load_adapter(**msg["kwargs"])

            elif msg_type == "SAVE_ADAPTER":
                engine.save_adapter(**msg["kwargs"])
