"""Worker process for Pie multi-GPU driver.

This module contains everything that runs inside a spawned child process:
topology calculation and the two worker roles:
  - Group leaders run the RPC loop (receive from Rust, broadcast to TP peers)
  - Followers wait for broadcasts and run inference

Bridge is torch-free: every operation that touches `torch.distributed`,
`torch.cuda`, or per-flavor KV tensors is dispatched through a
flavor-supplied `runtime_ops` module (see `pie_driver_dev.utils` for the
contract) or through methods on `engine` (see `engine.kv_copy_*`).
"""

from __future__ import annotations


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
    group_id_base: int,
    build_engine,
    runtime_ops,
    runtime_config_extras: dict | None = None,
    config_cls=None,
):
    """Generic worker body shared across drivers (native / vllm / sglang).

    Owns the universal lifecycle: tqdm lock, ready-queue handshake, and
    leader/follower dispatch. Anything that imports torch is dispatched
    through:

      * `runtime_ops` — a module-like object with
        `init_distributed(tp_rank, tp_degree, group_id, master_port, device)`,
        `set_device(device)`, `barrier()`, `broadcast_struct(data, src,
        device)`, `cleanup_runtime()`, and `cleanup_distributed()`. Each
        flavor wheel ships its own copy (see `pie_driver_dev.utils`).

      * `engine.kv_copy_{d2h,h2d,d2d,h2h}` — per-flavor KV-swap methods.

    Driver-specific engine work happens inside `build_engine`:

        engine = build_engine(runtime_config)

    The default PG ends up being the TP group itself (each DP replica brings
    up its own torch.distributed world via FileStore on a per-group path —
    matches sglang's native multi-process DP architecture). Collectives in
    driver code call `dist.all_reduce(t)` / `dist.broadcast(t, src=0)` etc.
    with no explicit group; they hit the default PG, which is the TP group.

    `runtime_config_extras` is merged into the kwargs passed to
    `<config_cls>.from_args` after `model_config`. Drivers whose knobs live
    on the runtime config subclass (native, dummy → NativeRuntimeConfig) pass
    their `driver_config` dict here. Drivers that hold knobs in their own
    typed config (vllm) pass nothing and use the universal `RuntimeConfig`.

    `config_cls` selects the dataclass: `RuntimeConfig` for vllm,
    `NativeRuntimeConfig` (downstream) for native/dummy. Defaults to
    `RuntimeConfig`.

    CUDA cleanup is wrapped around the body so leaks don't survive a
    crashed worker.
    """
    import gc
    import threading
    import inspect
    import os

    from tqdm import tqdm

    from .config import RuntimeConfig

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
        global_group_id = group_id_base + my_group_id
        tp_degree = len(group_topology[my_group_id])
        group_devices = [devices[r] for r in group_topology[my_group_id]]

        # Per-replica MASTER_PORT so DP replicas on the same host don't
        # collide on downstream nccl ports (sglang derives `nccl_port =
        # MASTER_PORT + 1`; with the default 29500 every replica would try
        # to bind 29501). Stride by 10 leaves room for sglang's auxiliary
        # ports without crossing into the next replica.
        os.environ["MASTER_PORT"] = str(master_port + my_group_id * 10)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

        configure_dist_env = getattr(
            runtime_ops, "configure_distributed_environment", None
        )
        if configure_dist_env is not None:
            configure_dist_env(tp_degree=tp_degree, devices=group_devices)

        # Distributed init: per-TP-group, NOT global. Default PG ends up
        # being the TP group itself. Skipped entirely when this group has
        # only one rank (no collectives to do).
        if tp_degree > 1:
            runtime_ops.init_distributed(
                tp_rank, tp_degree, my_group_id, master_port, devices[rank]
            )
        else:
            runtime_ops.set_device(devices[rank])

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
        config = config_cls.from_args(
            **merged,
            devices=group_devices,
            rank=tp_rank,
            tensor_parallel_size=tp_degree,
        )

        engine = build_engine(config)

        runtime_ops.barrier()

        is_group_leader = tp_rank == 0
        try:
            if is_group_leader:
                # No more RpcServer — all method dispatch goes through
                # the unified shmem channel inside `_leader_loop`. The
                # ready-queue payload's second element is unused but
                # kept for launcher-side schema compatibility.
                #
                # The ready signal is deferred until INSIDE `_leader_loop`,
                # after the `ShmemServer` is created, to close the race
                # where the runtime would attach to a not-yet-existing
                # shmem region and see zero magic. We pass a callable so
                # the leader loop can fire it at the right moment.
                _shmem_spin_budget_us = int(model_config.get("shmem_spin_budget_us", 1000))
                caps = engine.capabilities()
                def _signal_ready():
                    ready_queue.put((rank, "", caps))

                stop_event = threading.Event()
                _leader_loop(
                    engine=engine,
                    stop_event=stop_event,
                    group_id=global_group_id,
                    runtime_ops=runtime_ops,
                    shmem_spin_budget_us=_shmem_spin_budget_us,
                    on_ready=_signal_ready,
                )
            else:
                ready_queue.put((rank, None, None))
                _follower_loop(
                    engine=engine,
                    config=config,
                    runtime_ops=runtime_ops,
                )
        finally:
            runtime_ops.cleanup_distributed()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        runtime_ops.cleanup_runtime()
        gc.collect()


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
    stop_event,
    group_id: int,
    runtime_ops,
    shmem_spin_budget_us: int = 1000,
    on_ready=None,
) -> None:
    """Unified shmem dispatch loop for group leaders.

    Polls the per-driver shmem ring for method-tagged requests; the
    method_tag in each slot header selects between forward, page-copy,
    and adapter-load handlers. All transport flows through the same
    `DriverChannel` — there is no separate cold-path RPC.
    """
    import time
    from .batching import Batch
    from .latency import StepTiming, LatencyStats

    config = engine.config
    latency_stats = LatencyStats(enabled=config.telemetry_enabled)

    # Some runtime configs carry the resolved page size directly. Fall back
    # to the engine's capability handshake (which every driver implements)
    # and to a sensible default for max_dist_size, which only matters for
    # distribution-mode sampling responses.
    _kv_page_size = getattr(config, "kv_page_size", None)
    if _kv_page_size is None:
        _kv_page_size = engine.capabilities().kv_page_size
    _max_dist_size = getattr(config, "max_dist_size", 32)

    def _run_fire_batch_inner(kwargs):
        """Run the model + verify-drafts + populate-drafts steps and return
        `(sampling_results, batch, timings)` where `timings` is a dict ready
        to feed StepTiming.
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
        inputs = engine.build_model_inputs(batch)
        t_get_inputs = time.perf_counter() - t0

        t0 = time.perf_counter()
        sampling_metadata = engine.build_sampling_metadata(batch)
        t_get_sampling_meta = time.perf_counter() - t0

        # Broadcast to TP followers if multi-GPU. The default PG IS the
        # TP group after the per-replica distributed init, so leader =
        # local rank 0 and no `group=` argument is needed.
        t0 = time.perf_counter()
        should_broadcast = config.world_size > 1 and config.rank == 0
        if should_broadcast:
            runtime_ops.broadcast_struct(
                {
                    "type": "STEP",
                    "inputs": inputs,
                    "sampling_metadata": sampling_metadata,
                },
                src=0,
                device=device,
            )
        t_broadcast = time.perf_counter() - t0

        if config.world_size > 1:
            runtime_ops.barrier()

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

    import os as _os, sys as _sys, time as _time

    # ---- Shmem channel setup -------------------------------------------------
    # All driver RPCs (forward batches, page copies, adapter ops) flow
    # through the same method-tagged shmem ring — see the unified
    # `DriverChannel` in `runtime/src/driver/shmem.rs`.
    #
    # Per-DP-replica shmem region. POSIX shm names are global per-host, so a
    # hardcoded "/pie_shmem" used to collide silently across DP replicas —
    # one replica's leader would create the region, the other would either
    # fail or attach to the same backing store, mixing requests across
    # replicas. The runtime mirrors this naming in `device.rs`
    # (`shmem_name(device_idx)` → `/pie_shmem_g{idx}`); device_idx in Rust
    # equals group_id in Python because pie/server.py builds DeviceConfig
    # in group order.
    SHMEM_NAME = f"/pie_shmem_g{group_id}"
    SHMEM_SLOTS = 8
    SHMEM_REQ_BUF = 4 * 1024 * 1024
    # Sized to hold a full-vocab `Distribution` probe / `Sampler::Dist`
    # payload (vocab × 8 bytes ≈ 2.6 MiB on 150K-vocab models) plus
    # per-request overhead and the spec-mode multi-slot tail. The Rust
    # client picks this up from the shmem header at attach time, so it's
    # a purely driver-side knob.
    SHMEM_RESP_BUF = 8 * 1024 * 1024
    from . import shmem_ipc as _shm
    from . import shmem_schema as _shm_schema
    import pie_bridge as _pb
    import threading as _threading
    # `spin_budget_us` controls how long poll_blocking busy-spins
    # before parking on the global `req_wake` atomic (cross-process
    # futex / WaitOnAddress / __ulock_wait). See `ShmemServer::create`
    # in pie_bridge.
    _shmem_server = _shm.ShmemServer(
        SHMEM_NAME, num_slots=SHMEM_SLOTS,
        req_buf=SHMEM_REQ_BUF, resp_buf=SHMEM_RESP_BUF,
        spin_budget_us=shmem_spin_budget_us,
    )
    _response_builder = _shm_schema.ResponseBuilder()
    print(f"[shmem] Worker created shmem region '{SHMEM_NAME}' "
          f"({SHMEM_SLOTS} slots) — fast path enabled.")

    # Now that the shmem region exists with its header populated, it's
    # safe for the launcher to emit the ready handshake to the runtime.
    # Without this ordering the runtime would race shmem creation and
    # either see "No such file" or a zero-magic header.
    if on_ready is not None:
        on_ready()

    import signal as _signal

    from . import methods as _methods

    _METHOD_FORWARD = _methods.FORWARD
    _METHOD_COPY_D2H = _methods.COPY_D2H
    _METHOD_COPY_H2D = _methods.COPY_H2D
    _METHOD_COPY_D2D = _methods.COPY_D2D
    _METHOD_COPY_H2H = _methods.COPY_H2H
    _METHOD_LOAD_ADAPTER = _methods.LOAD_ADAPTER
    _METHOD_SAVE_ADAPTER = _methods.SAVE_ADAPTER
    _METHOD_ZO_INITIALIZE_ADAPTER = _methods.ZO_INITIALIZE_ADAPTER
    _METHOD_ZO_UPDATE_ADAPTER = _methods.ZO_UPDATE_ADAPTER
    _METHOD_HEALTH = _methods.HEALTH

    def _handle_copy_v2(method_tag: int, srcs: list, dsts: list) -> int:
        try:
            if method_tag == _METHOD_COPY_D2H:
                engine.kv_copy_d2h(srcs, dsts)
            elif method_tag == _METHOD_COPY_H2D:
                engine.kv_copy_h2d(srcs, dsts)
            elif method_tag == _METHOD_COPY_D2D:
                engine.kv_copy_d2d(srcs, dsts)
            elif method_tag == _METHOD_COPY_H2H:
                engine.kv_copy_h2h(srcs, dsts)
            return 0
        except Exception as e:
            print(f"[pie_driver_dev] copy method={method_tag} failed: {e}")
            return 5

    def _handle_load_adapter_v2(adapter_id: int, path: str) -> int:
        try:
            if config.world_size > 1:
                runtime_ops.broadcast_struct(
                    {"type": "LOAD_ADAPTER",
                     "kwargs": {"adapter_ptr": adapter_id, "name": path, "data": b""}},
                    src=0,
                    device=config.device,
                )
            engine.load_adapter(adapter_ptr=adapter_id, name=path, data=b"")
            return 0
        except Exception as e:
            print(f"[pie_driver_dev] load_adapter failed: {e}")
            return 5

    def _shmem_loop():
        # `torch.cuda.set_device()` is thread-local. Pin this dispatch
        # thread to the same device the engine loaded onto.
        try:
            runtime_ops.set_device(config.device)
        except Exception:
            pass
        while not stop_event.is_set():
            # poll_blocking is a millisecond timeout; `spin_budget_us`
            # (in the ShmemServer constructor) controls how long the
            # busy-spin window is before the cross-process park kicks
            # in. A 50 ms timeout gives a responsive stop_event check.
            lease = _shmem_server.poll_blocking(50)
            if lease is None:
                continue
            payload = lease.payload  # bytes (copy from shmem slot)
            try:
                method_tag = _shm_schema.peek_method_tag(payload)
            except Exception as e:
                print(f"[pie_driver_dev] malformed request: {e}")
                lease.commit_status(2)
                continue

            if method_tag == _METHOD_FORWARD:
                args = _shm_schema.parse_request(payload)
                driver_id = int(args.get("driver_id", 0))
                try:
                    sampling_results, batch, timings = _run_fire_batch_inner(args)
                except Exception:
                    import traceback
                    _sys.stderr.write(
                        f"[shmem worker pid={_os.getpid()}] fire_batch raised; aborting:\n"
                    )
                    traceback.print_exc(file=_sys.stderr)
                    _sys.stderr.flush()
                    _os._exit(1)
                t_after_handler = _time.perf_counter()
                out_bytes = _response_builder.build_from_batch(
                    sampling_results, batch, driver_id
                )
                lease.commit(out_bytes)
                _record_step_timing(timings, _time.perf_counter() - t_after_handler)
            elif method_tag in (_METHOD_COPY_D2H, _METHOD_COPY_H2D,
                                _METHOD_COPY_D2D, _METHOD_COPY_H2H):
                cr = _pb.Frame.parse(payload).payload.as_copy()
                if cr is None:
                    lease.commit_status(2)
                    continue
                status = _handle_copy_v2(method_tag, list(cr.srcs), list(cr.dsts))
                lease.commit_status(status)
            elif method_tag == _METHOD_LOAD_ADAPTER:
                ar = _pb.Frame.parse(payload).payload.as_adapter()
                if ar is None:
                    lease.commit_status(2)
                    continue
                path = ar.path or ""
                status = _handle_load_adapter_v2(int(ar.adapter_id), path)
                lease.commit_status(status)
            elif method_tag in (_METHOD_SAVE_ADAPTER,
                                _METHOD_ZO_INITIALIZE_ADAPTER,
                                _METHOD_ZO_UPDATE_ADAPTER):
                # No-op stubs — adapter persistence and zeroth-order
                # training aren't implemented in any Python driver yet.
                lease.commit_status(0)
            elif method_tag == _METHOD_HEALTH:
                lease.commit_status(0)
            else:
                # Unknown method — bad_method status.
                lease.commit_status(2)

    _shmem_thread = _threading.Thread(target=_shmem_loop, name="shmem-fire-batch", daemon=True)
    _shmem_thread.start()

    def _stop_shmem():
        # Quiesce the busy-poll thread, THEN unmap the region. Closing the
        # mmap before the thread exits leaves it dereferencing freed memory
        # inside _u64_at_slot.
        try:
            stop_event.set()
        except Exception:
            pass
        try:
            _shmem_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            # New ShmemServer auto-releases on drop; stop() signals any
            # internal poll loops to exit.
            _shmem_server.stop()
        except Exception:
            pass

    # mp.spawn daemon=True sends SIGTERM on parent exit, which bypasses atexit.
    def _shmem_signal_cleanup(signum, frame):
        _stop_shmem()
        _os._exit(0)
    for _sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _signal.signal(_sig, _shmem_signal_cleanup)
        except Exception:
            pass

    try:
        # The shmem thread is the only dispatch loop — block until stop.
        stop_event.wait()
    finally:
        _stop_shmem()
        print("[pie_driver_dev] Shutting down...")
        if config.world_size > 1 and config.rank == 0:
            try:
                runtime_ops.broadcast_struct(
                    "STOP",
                    src=0,
                    device=config.device,
                )
            except Exception:
                pass


# =============================================================================
# Follower Loop
# =============================================================================


def _follower_loop(
    engine,
    config,
    runtime_ops,
) -> None:
    """Broadcast loop for TP followers.

    Waits for control messages from the group leader and executes
    inference steps or adapter operations.
    """
    import signal

    device = config.device

    shutdown_requested = False

    def sigterm_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    while not shutdown_requested:
        # Receive control message from leader (rank 0 of the TP group,
        # which is the only group in this worker's distributed world).
        try:
            msg = runtime_ops.broadcast_struct(None, src=0, device=device)
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
                    # TP barrier — default PG is the TP group post-init.
                    if config.world_size > 1:
                        runtime_ops.barrier()

                    engine.fire_batch(inputs, sampling_metadata)
                except Exception as e:
                    print(f"Worker {config.rank} fire_batch error: {e}")

            elif msg_type == "INIT_ADAPTER":
                engine.init_adapter(**msg["kwargs"])

            elif msg_type == "UPDATE_ADAPTER":
                engine.update_adapter(**msg["kwargs"])

            elif msg_type == "LOAD_ADAPTER":
                engine.load_adapter(**msg["kwargs"])

            elif msg_type == "SAVE_ADAPTER":
                engine.save_adapter(**msg["kwargs"])
