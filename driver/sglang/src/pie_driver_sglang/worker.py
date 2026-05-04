"""Worker entry point for the sglang driver.

Delegates the universal lifecycle (distributed init, group setup, ready-queue
handshake, leader/follower dispatch) to `pie_driver_dev.worker.run_worker`, then
plugs in sglang-specific engine construction.
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
    ready_queue,
):
    """Worker entry point — `sglang` driver.

    `driver_config` is `SGLangDriverConfig` as a dict; sglang-native knobs
    live on the typed dataclass and never leak into pie's `RuntimeConfig`.
    """
    from pie_driver_dev.worker import run_worker
    from .config import SGLangDriverConfig
    from .engine import SGLangEngine

    sgl_cfg = SGLangDriverConfig(**driver_config)

    run_worker(
        local_rank=local_rank,
        world_size=world_size,
        devices=devices,
        master_port=master_port,
        model_config=model_config,
        group_topology=group_topology,
        ready_queue=ready_queue,
        build_engine=lambda cfg: SGLangEngine.load(cfg, sgl_cfg),
        # `cpu_mem_budget_in_gb` is the only universal pie knob carried on
        # SGLangDriverConfig; forward it so RuntimeConfig.swap_budget_bytes
        # gets sized correctly.
        runtime_config_extras={"cpu_mem_budget_in_gb": sgl_cfg.cpu_mem_budget_in_gb},
    )
