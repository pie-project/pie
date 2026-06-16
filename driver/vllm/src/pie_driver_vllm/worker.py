"""Worker entry point for the vllm driver.

Delegates the universal lifecycle (distributed init, group setup, ready-queue
handshake, leader/follower dispatch) to `._bridge.worker.run_worker`, then
plugs in vllm-specific engine construction.
"""

from __future__ import annotations


# Re-export for parity with other workers (server.py imports this).
from ._bridge.worker import calculate_topology  # noqa: F401


def worker_main(
    local_rank: int,
    world_size: int,
    devices: list[str],
    master_port: int,
    model_config: dict,
    driver_config: dict,
    group_topology: list[list[int]],
    group_id_base: int,
    ready_queue,
):
    """Worker entry point — `vllm` driver.

    `driver_config` is `VllmDriverConfig` as a dict; vllm's own knobs live
    on the typed dataclass and never leak into pie's `RuntimeConfig`.
    """
    import os

    # Match vLLM's multiprocessing executor behavior. Without this clamp each
    # embedded TP rank can fan out to hundreds of CPU threads during weight
    # load/postprocess, which makes large hybrid checkpoints spend minutes in
    # loader CPU contention before Pie ever reaches the ready handshake.
    if "OMP_NUM_THREADS" not in os.environ:
        try:
            import torch

            if torch.get_num_threads() > 1:
                os.environ["OMP_NUM_THREADS"] = "1"
                torch.set_num_threads(1)
        except Exception:
            pass

    # Only isolate vLLM's cache when Pie is launching multiple independent
    # DP replicas. For a single TP group, keep vLLM's normal cache root so
    # compile/kernel artifacts are shared with standalone vLLM benchmarks.
    if len(group_topology) > 1:
        safe_device = devices[local_rank].replace(":", "_")
        os.environ.setdefault(
            "VLLM_CACHE_ROOT",
            f"/tmp/pie_vllm_cache/pid_{os.getpid()}_{safe_device}",
        )

    from ._bridge.worker import run_worker
    from . import utils as runtime_ops
    from .config import VllmDriverConfig
    from .engine import VllmEngine

    vllm_cfg = VllmDriverConfig(**driver_config)

    run_worker(
        local_rank=local_rank,
        world_size=world_size,
        devices=devices,
        master_port=master_port,
        model_config=model_config,
        group_topology=group_topology,
        group_id_base=group_id_base,
        ready_queue=ready_queue,
        build_engine=lambda cfg: VllmEngine.load(cfg, vllm_cfg),
        runtime_ops=runtime_ops,
    )
