"""Worker entry point for the `dummy` driver.

Reuses pie_driver's engine machinery with `dummy_mode=True` to skip real
weight loading. Useful for tests and scheduler/runtime smoke checks.
"""

from __future__ import annotations


# Re-export for parity with other workers (server.py imports this).
from pie_driver.worker import calculate_topology  # noqa: F401


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
    """Worker entry point — `dummy` driver."""
    from pie_driver.config import NativeRuntimeConfig
    from pie_driver.engine import Engine
    from pie_driver.worker import run_worker

    run_worker(
        local_rank=local_rank,
        world_size=world_size,
        devices=devices,
        master_port=master_port,
        model_config=model_config,
        group_topology=group_topology,
        ready_queue=ready_queue,
        build_engine=lambda cfg: Engine.load(cfg),
        # Dummy uses NativeRuntimeConfig with `dummy_mode=True` so Engine.load
        # takes its dummy branch (no real weight load, returns random tokens).
        runtime_config_extras={**driver_config, "dummy_mode": True},
        config_cls=NativeRuntimeConfig,
    )
