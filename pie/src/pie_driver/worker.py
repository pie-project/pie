"""Worker process for Pie multi-GPU driver.

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


def run_worker(
    *,
    local_rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,
    model_config: dict,
    group_topology: list[list[int]],
    ready_queue,
    build_engine,
    runtime_config_extras: dict | None = None,
    config_cls=None,
):
    """Generic worker body shared across drivers (native / vllm / dummy).

    Owns the universal lifecycle: tqdm lock, torch.distributed init, group
    setup, RuntimeConfig assembly, ready-queue handshake, and leader/follower
    dispatch. Driver-specific work happens inside `build_engine`:

        engine = build_engine(runtime_config, compute_pg)

    `runtime_config_extras` is merged into the kwargs passed to
    `<config_cls>.from_args` after `model_config`. Drivers whose knobs live
    on the runtime config subclass (native, dummy → NativeRuntimeConfig) pass
    their `driver_config` dict here. Drivers that hold knobs in their own
    typed config (vllm) pass nothing and use the universal `RuntimeConfig`.

    `config_cls` selects the dataclass: `RuntimeConfig` for vllm,
    `NativeRuntimeConfig` for native/dummy. Defaults to `RuntimeConfig`.

    CUDA cleanup is wrapped around the body so leaks don't survive a
    crashed worker.
    """
    import gc
    import threading
    import inspect

    import torch
    import torch.distributed as dist
    from tqdm import tqdm

    from pie import _runtime as pie_runtime
    from pie_driver.config import RuntimeConfig

    if config_cls is None:
        config_cls = RuntimeConfig

    # Workers only need thread-safety for tqdm, not the default
    # multiprocessing.RLock (creates a POSIX semaphore that leaks when the
    # worker is terminated).
    tqdm.set_lock(threading.RLock())

    rank = local_rank
    try:
        # Determine group membership
        my_group_id = 0
        tp_rank = 0
        for i, group in enumerate(group_topology):
            if rank in group:
                my_group_id = i
                tp_rank = group.index(rank)
                break
        tp_degree = len(group_topology[my_group_id])

        # Distributed init (TP > 1 only; single-rank skips for latency)
        if world_size > 1:
            _init_distributed(rank, world_size, master_port, devices[rank])
            # Side-effect: also creates leader-inclusive groups used by
            # downstream broadcast paths.
            _setup_process_groups(group_topology)
            compute_pg_map = _setup_compute_process_groups(group_topology)
        else:
            device_str = devices[rank]
            if device_str.startswith("cuda"):
                torch.cuda.set_device(device_str)
            compute_pg_map = {}

        # Build the runtime config (lean RuntimeConfig for vllm,
        # NativeRuntimeConfig for native/dummy). Universal kwargs come from
        # `model_config`; driver-specific kwargs from `runtime_config_extras`.
        # `device`/`devices`/`tensor_parallel_size` are passed explicitly
        # below (per-rank values), so they should never come through
        # `merged_source`.
        valid_keys = set(inspect.signature(config_cls.from_args).parameters.keys())
        merged_source = model_config | (runtime_config_extras or {})
        merged = {
            k: v for k, v in merged_source.items()
            if k in valid_keys and v is not None
        }
        group_devices = [devices[r] for r in group_topology[my_group_id]]
        config = config_cls.from_args(
            **merged,
            devices=group_devices,
            rank=tp_rank,
            tensor_parallel_size=tp_degree,
        )

        compute_pg = compute_pg_map.get(my_group_id)
        engine = build_engine(config, compute_pg)

        if dist.is_initialized():
            dist.barrier()

        is_group_leader = tp_rank == 0
        try:
            if is_group_leader:
                server = pie_runtime.RpcServer.create()
                server_name = server.server_name()
                ready_queue.put((rank, server_name, engine.capabilities()))

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
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        gc.collect()


def worker_main(
    local_rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,
    model_config: dict,
    driver_config: dict,
    group_topology: list[list[int]],
    ready_queue,
):
    """Worker entry point — `native` driver.

    `driver_config` is `NativeDriverConfig` as a dict; native's knobs live
    on `NativeRuntimeConfig` (a subclass of the universal `RuntimeConfig`),
    so we forward the dict to `run_worker` as `runtime_config_extras`.
    """
    from pie_driver.config import NativeRuntimeConfig
    from pie_driver.engine import Engine

    run_worker(
        local_rank=local_rank,
        world_size=world_size,
        devices=devices,
        master_port=master_port,
        model_config=model_config,
        group_topology=group_topology,
        ready_queue=ready_queue,
        build_engine=lambda cfg, pg: Engine.load(cfg, compute_process_group=pg),
        runtime_config_extras=driver_config,
        config_cls=NativeRuntimeConfig,
    )


# =============================================================================
# Leader Loop
# =============================================================================

# RPC status codes (must match Rust)
_STATUS_OK = 0
_STATUS_METHOD_NOT_FOUND = 1
_STATUS_INVALID_PARAMS = 2
_STATUS_INTERNAL_ERROR = 3


def _populate_next_drafts(batch, sampling_results: dict, engine) -> None:
    """Ask the engine for next-iteration drafts for spec-output requests.

    Looks for `engine.spec_step(sessions)` on the engine — drivers that
    don't implement it (native flashinfer, vllm) are no-ops here. For each
    request that asked for `output_speculative_tokens(true)`, builds a
    `(session_id, accepted_tokens)` pair and hands it to `spec_step`,
    which observes the just-accepted tokens (extending per-session
    history) and proposes a draft continuation. The result is stuffed
    into `sampling_results['spec_tokens']` / `['spec_positions']`, and
    `Batch.create_responses` packs it into TokensWithSpeculation.

    Session ID: the request's first physical KV page ID (stable for an
    active context across iterations). Eviction would invalidate it, but
    a new context just gets a new session.
    """
    step = getattr(engine, "spec_step", None)
    if step is None:
        return

    num_requests = len(batch.request_output_counts)
    output_flags = batch.output_spec_flags
    spec_accepted_all = sampling_results.get("spec_accepted_tokens", None)
    final_tokens = sampling_results.get("tokens", [])

    sessions: list[tuple[int, list[int]]] = []
    next_draft_base: list[int] = []  # one entry per `sessions` entry
    spec_request_idx: list[int] = []

    cursor = 0  # walks over inferlet sampler slots in final_tokens
    for i in range(num_requests):
        num_outputs = int(batch.request_output_counts[i])
        if spec_accepted_all is not None and spec_accepted_all[i] is not None:
            accepted = list(spec_accepted_all[i])
        else:
            accepted = []
            for k in range(cursor, cursor + num_outputs):
                if int(batch.sampler_types[k]) != 0:
                    accepted.append(int(final_tokens[k]))
        cursor += num_outputs

        if not output_flags[i] or not accepted:
            continue

        # Stable per-context session id: the runtime's ContextId, carried
        # in `BatchedForwardPassRequest.context_ids`. ContextId stays valid
        # across swap + restore; the first KV page id (older fallback) does
        # not, so we prefer it. The fallback covers runtimes that haven't
        # populated context_ids yet.
        if batch.context_ids:
            session_id = int(batch.context_ids[i])
        else:
            page_start = int(batch.kv_page_indptr[i])
            page_end = int(batch.kv_page_indptr[i + 1])
            if page_end == page_start:
                continue
            session_id = int(batch.kv_page_indices[page_start])
        last_pending_pos = int(batch.position_ids[batch.qo_indptr[i + 1] - 1])

        sessions.append((session_id, accepted))
        next_draft_base.append(last_pending_pos + len(accepted))
        spec_request_idx.append(i)

    if not sessions:
        return

    drafts_per_session = step(sessions)

    spec_tokens_per_req: list[list[int] | None] = [None] * num_requests
    spec_positions_per_req: list[list[int] | None] = [None] * num_requests
    for s_idx, req_i in enumerate(spec_request_idx):
        chain = drafts_per_session[s_idx]
        if not chain:
            continue
        base = next_draft_base[s_idx]
        spec_tokens_per_req[req_i] = chain
        spec_positions_per_req[req_i] = [base + 1 + k for k in range(len(chain))]

    sampling_results["spec_tokens"] = spec_tokens_per_req
    sampling_results["spec_positions"] = spec_positions_per_req


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
    from pie_driver import utils, message
    from pie_driver.batching import Batch
    from pie_driver.latency import StepTiming, LatencyStats

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

    # `kv_page_size` and `max_dist_size` are NativeRuntimeConfig-only after
    # the universal/native config split. Fall back to the engine's
    # capability handshake (which every driver implements) for kv_page_size,
    # and to a sensible default for max_dist_size which only matters for
    # distribution-mode sampling responses.
    _kv_page_size = getattr(config, "kv_page_size", None)
    if _kv_page_size is None:
        _kv_page_size = engine.capabilities().kv_page_size
    _max_dist_size = getattr(config, "max_dist_size", 32)

    def _run_fire_batch_inner(kwargs):
        """Run the model + verify-drafts + populate-drafts steps and return
        `(sampling_results, batch, timings)` where `timings` is a dict ready
        to feed StepTiming.

        Used by both the dict-returning `_handle_fire_batch` (cold-path
        compatibility) and the shmem fast path that encodes directly.
        """
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        batch = Batch(
            kwargs,
            _kv_page_size,
            _max_dist_size,
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

        t0 = time.perf_counter()
        if batch.has_speculative_inputs:
            inputs = batch.get_spec_expanded_model_inputs(device)
        else:
            inputs = batch.get_model_inputs(device)
        t_get_inputs = time.perf_counter() - t0

        t0 = time.perf_counter()
        if batch.has_speculative_inputs:
            sampling_metadata = batch.get_spec_expanded_sampling_metadata(
                device, config.activation_dtype
            )
        else:
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

        if config.world_size > 1 and compute_pg is not None:
            if dist.get_world_size(group=compute_pg) > 1:
                dist.barrier(group=compute_pg)

        t0 = time.perf_counter()
        sampling_results = engine.fire_batch(inputs, sampling_metadata)
        t_inference = time.perf_counter() - t0

        if batch.has_speculative_inputs:
            batch.verify_drafts(sampling_results)
        _populate_next_drafts(batch, sampling_results, engine)

        timings = {
            "t_start": t_start,
            "build_batch": t_build_batch,
            "get_inputs": t_get_inputs,
            "get_sampling_meta": t_get_sampling_meta,
            "broadcast": t_broadcast,
            "inference": t_inference,
            "build_timing": build_timing,
            "traceparent": kwargs.get("trace_context"),
        }
        return sampling_results, batch, timings

    def _record_step_timing(timings, t_create_responses):
        latency_stats.record_span(
            StepTiming(
                build_batch=timings["build_batch"],
                get_inputs=timings["get_inputs"],
                get_sampling_meta=timings["get_sampling_meta"],
                broadcast=timings["broadcast"],
                inference=timings["inference"],
                create_responses=t_create_responses,
                total=time.perf_counter() - timings["t_start"],
                decode_u32=timings["build_timing"]["decode_u32"],
                mask_loop=timings["build_timing"]["mask_loop"],
                brle_decode=timings["build_timing"]["brle_decode"],
                sampler_loop=timings["build_timing"]["sampler_loop"],
            ),
            traceparent=timings["traceparent"],
        )

    def _handle_fire_batch(**kwargs) -> dict:
        """Cold-path entry: returns a dict (msgpack-serializable). Used by
        the legacy ipc-channel transport. The shmem fast path bypasses this
        and encodes directly into the response buffer.
        """
        sampling_results, batch, timings = _run_fire_batch_inner(kwargs)
        t0 = time.perf_counter()
        responses = batch.create_responses(sampling_results)
        results = [
            {
                "tokens": resp.tokens,
                "dists": resp.dists,
                "logits": getattr(resp, "logits", []) or [],
                "logprobs": getattr(resp, "logprobs", []) or [],
                "entropies": getattr(resp, "entropies", []) or [],
                "spec_tokens": getattr(resp, "spec_tokens", []) or [],
                "spec_positions": getattr(resp, "spec_positions", []) or [],
            }
            for resp in responses
        ]
        t_create_responses = time.perf_counter() - t0
        _record_step_timing(timings, t_create_responses)
        return {"results": results}

    def _handle_copy_d2h(**kwargs) -> None:
        """D2H: copy GPU KV pages to pinned CPU buffers (vectorized per layer).

        NOTE: tensor[idx].copy_(src) with advanced (fancy) indexing is a NO-OP
        in PyTorch — it creates a temporary copy and .copy_() writes to the
        temporary, which is then discarded. We use index_copy_ instead, which
        correctly scatter-writes to the original tensor.
        """
        import torch
        gpu_kv = engine.kv_cache_at_layer
        host_kv = engine.kv_cache_at_layer_host
        phys_ids = kwargs["phys_ids"]
        slots = kwargs["slots"]
        max_gpu = gpu_kv[0].shape[0]
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for p in phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"swap_out: GPU phys_id {p} out of bounds [0, {max_gpu})")
        for s in slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"swap_out: CPU slot {s} out of bounds [0, {max_cpu})")
        src = torch.tensor(phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        dst = torch.tensor(slots, dtype=torch.long)
        for layer_idx in range(len(gpu_kv)):
            host_kv[layer_idx].index_copy_(0, dst, gpu_kv[layer_idx][src].cpu())
        #torch.cuda.synchronize() -> we don't need this

    def _handle_copy_h2d(**kwargs) -> None:
        """H2D: copy pinned CPU buffers back to GPU KV pages (vectorized per layer).

        NOTE: Same as swap_out — must use index_copy_ to avoid the fancy
        indexing no-op bug with tensor[idx].copy_().
        """
        import torch
        gpu_kv = engine.kv_cache_at_layer
        host_kv = engine.kv_cache_at_layer_host
        phys_ids = kwargs["phys_ids"]
        slots = kwargs["slots"]
        max_gpu = gpu_kv[0].shape[0]
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for p in phys_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"swap_in: GPU phys_id {p} out of bounds [0, {max_gpu})")
        for s in slots:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"swap_in: CPU slot {s} out of bounds [0, {max_cpu})")
        dst = torch.tensor(phys_ids, dtype=torch.long, device=gpu_kv[0].device)
        src = torch.tensor(slots, dtype=torch.long)
        for layer_idx in range(len(gpu_kv)):
            gpu_kv[layer_idx].index_copy_(0, dst, host_kv[layer_idx][src].to(gpu_kv[layer_idx].device))
        #torch.cuda.synchronize() -> we don't need this

    def _handle_copy_d2d(**kwargs) -> None:
        """D2D: copy GPU KV pages to other GPU KV pages.

        Args (via kwargs):
            src_phys_ids: list[int] — source GPU physical page IDs
            dst_phys_ids: list[int] — destination GPU physical page IDs
        """
        import torch
        gpu_kv = engine.kv_cache_at_layer
        src_ids = kwargs["src_phys_ids"]
        dst_ids = kwargs["dst_phys_ids"]
        max_gpu = gpu_kv[0].shape[0]
        for p in src_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"copy_d2d: src phys_id {p} out of bounds [0, {max_gpu})")
        for p in dst_ids:
            if p < 0 or p >= max_gpu:
                raise ValueError(f"copy_d2d: dst phys_id {p} out of bounds [0, {max_gpu})")
        src = torch.tensor(src_ids, dtype=torch.long, device=gpu_kv[0].device)
        dst = torch.tensor(dst_ids, dtype=torch.long, device=gpu_kv[0].device)
        for layer_idx in range(len(gpu_kv)):
            gpu_kv[layer_idx].index_copy_(0, dst, gpu_kv[layer_idx][src])
        #torch.cuda.synchronize() -> we don't need this

    def _handle_copy_h2h(**kwargs) -> None:
        """H2H: copy pinned CPU pages to other CPU pages (no GPU involved)."""
        import torch
        host_kv = engine.kv_cache_at_layer_host
        src_ids = kwargs["src_slots"]
        dst_ids = kwargs["dst_slots"]
        max_cpu = host_kv[0].shape[0] if host_kv else 0
        for s in src_ids:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"copy_h2h: src slot {s} out of bounds [0, {max_cpu})")
        for s in dst_ids:
            if s < 0 or s >= max_cpu:
                raise ValueError(f"copy_h2h: dst slot {s} out of bounds [0, {max_cpu})")
        src = torch.tensor(src_ids, dtype=torch.long)
        dst = torch.tensor(dst_ids, dtype=torch.long)
        for layer_idx in range(len(host_kv)):
            host_kv[layer_idx].index_copy_(0, dst, host_kv[layer_idx][src])

    # Method dispatch table
    methods = {
        "ping": lambda **kw: {"ok": True},
        "query": _handle_query,
        "fire_batch": _handle_fire_batch,
        "embed_image": _handle_embed_image,
        "initialize_adapter": _handle_init_adapter,
        "update_adapter": _handle_update_adapter,
        "load_adapter": _handle_load_adapter,
        "save_adapter": _handle_save_adapter,
        "swap_out_pages": _handle_copy_d2h,
        "swap_in_pages": _handle_copy_h2d,
        "copy_d2h": _handle_copy_d2h,
        "copy_h2d": _handle_copy_h2d,
        "copy_d2d": _handle_copy_d2d,
        "copy_h2h": _handle_copy_h2h,
    }

    import os as _os, sys as _sys, time as _time

    # ---- Shmem fast-path setup -----------------------------------------------
    # `fire_batch` always runs over shared memory with the zero-copy schema;
    # all other RPCs (adapter ops, page copies, query/ping) use the legacy
    # ipc-channel transport on `server`.
    SHMEM_NAME = "/pie_shmem"
    SHMEM_SLOTS = 8
    SHMEM_REQ_BUF = 4 * 1024 * 1024
    SHMEM_RESP_BUF = 1 * 1024 * 1024
    SHMEM_BUSY_US = 10_000

    from pie_driver import shmem_ipc as _shm
    from pie_driver import shmem_schema as _shm_schema
    import threading as _threading
    _shmem_server = _shm.ShmemServer(
        SHMEM_NAME, num_slots=SHMEM_SLOTS,
        req_buf=SHMEM_REQ_BUF, resp_buf=SHMEM_RESP_BUF,
    )
    print(f"[shmem] Worker created shmem region '{SHMEM_NAME}' "
          f"({SHMEM_SLOTS} slots) — fast path enabled.")

    import signal as _signal

    def _shmem_loop():
        while not stop_event.is_set():
            req = _shmem_server.busy_poll_any_slot(SHMEM_BUSY_US)
            if req is None:
                continue
            req_id, method_tag, payload_len, slot, send_walltime_us = req

            # Zero-copy view of the request payload bytes (flat schema).
            view = _shmem_server.request_payload_view(slot, payload_len)
            args = _shm_schema.parse_request(memoryview(view))

            try:
                sampling_results, batch, timings = _run_fire_batch_inner(args)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[Shmem Worker Error] fire_batch: {e}\n{tb}")
                response = msgpack.packb(str(e))
                _shmem_server.respond(slot, response, 0)
                continue
            t_after_handler = _time.perf_counter()

            # Direct encode: bypass Batch.create_responses + dict comp on
            # the fast path; falls back to msgpack only when the batch
            # has special samplers / spec output.
            resp_view = _shmem_server.respond_view(slot)
            resp_len = _shm_schema.write_response_v2(
                sampling_results, batch, resp_view, msgpack
            )
            _shmem_server.commit_respond(slot, resp_len, 0)

            _record_step_timing(timings, _time.perf_counter() - t_after_handler)

    _shmem_thread = _threading.Thread(target=_shmem_loop, name="shmem-fire-batch", daemon=True)
    _shmem_thread.start()

    # Best-effort cleanup on signal-induced exit (mp.spawn daemon=True sends
    # SIGTERM on parent exit, which bypasses atexit).
    def _shmem_signal_cleanup(signum, frame):
        try:
            _shmem_server.close()
        except Exception:
            pass
        _os._exit(0)
    for _sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(_sig, _shmem_signal_cleanup)
        except Exception:
            pass

    try:
        # Cold-path RPC loop: adapter ops, page copies, query/ping. Busy-spin
        # for 10 ms before blocking — these calls are infrequent, so the
        # busy window is mostly there to keep latency low when they do fire.
        while not stop_event.is_set():
            req = server.poll_busy(SHMEM_BUSY_US, poll_timeout_ms)
            if req is None:
                continue
            request_id, method, payload, send_walltime_us = req

            try:
                args = msgpack.unpackb(payload)
                fn = methods.get(method)
                if fn is None:
                    response = msgpack.packb(f"Method not found: {method}")
                    server.respond(request_id, response, 0)
                    continue

                if isinstance(args, dict):
                    result = fn(**args)
                elif isinstance(args, (list, tuple)):
                    result = fn(*args)
                else:
                    result = fn(args)

                response = msgpack.packb(result)
                server.respond(request_id, response, 0)

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[RPC Server Error] {method}: {e}\n{tb}")
                response = msgpack.packb(str(e))
                server.respond(request_id, response)
    finally:
        # Shutdown: stop shmem thread and broadcast STOP to followers
        try:
            _shmem_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            _shmem_server.close()
        except Exception:
            pass
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
# Follower Loop
# =============================================================================


def _follower_loop(
    engine,
    compute_pg,
    leader_rank: int,
    group_id: int,
    config=None,
) -> None:
    """Broadcast loop for TP followers.

    Waits for control messages from the group leader and executes
    inference steps or adapter operations.
    """
    import signal
    import torch
    import torch.distributed as dist
    from pie_driver import utils

    device = config.device if config else "cuda:0"

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

                    engine.fire_batch(inputs, sampling_metadata)
                except Exception as e:
                    print(f"Worker {config.rank if config else '?'} fire_batch error: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            elif msg_type == "INIT_ADAPTER":
                engine.init_adapter(**msg["kwargs"])

            elif msg_type == "UPDATE_ADAPTER":
                engine.update_adapter(**msg["kwargs"])

            elif msg_type == "LOAD_ADAPTER":
                engine.load_adapter(**msg["kwargs"])

            elif msg_type == "SAVE_ADAPTER":
                engine.save_adapter(**msg["kwargs"])
