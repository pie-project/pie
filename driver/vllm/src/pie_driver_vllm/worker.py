"""Worker entry point for the vllm driver.

Delegates the universal lifecycle (distributed init, group setup, ready-queue
handshake, leader/follower dispatch) to `pie_driver_dev.worker.run_worker`, then
plugs in vllm-specific engine construction.
"""

from __future__ import annotations


# Re-export for parity with other workers (server.py imports this).
from pie_driver_dev.worker import calculate_topology  # noqa: F401


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
    from pie_driver_dev.worker import run_worker
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
    )
