"""Worker entry point for the dev (native flashinfer) driver.

Delegates the universal lifecycle (distributed init, group setup,
ready-queue handshake, leader/follower dispatch) to
`._bridge.worker.run_worker`, then plugs in the dev driver's
flashinfer-based `Engine`.
"""

from __future__ import annotations


# Re-export for parity with other workers (the standalone launcher calls
# `worker.calculate_topology` before spawning).
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
    """Worker entry point — `native` driver.

    `driver_config` is `NativeDriverConfig` as a dict; native's knobs live
    on `NativeRuntimeConfig` (a subclass of the universal `RuntimeConfig`),
    so we forward the dict to `run_worker` as `runtime_config_extras`.
    """
    from ._bridge.worker import run_worker
    from pie_driver_dev import utils as runtime_ops
    from pie_driver_dev.config import NativeRuntimeConfig
    from pie_driver_dev.engine import Engine

    run_worker(
        local_rank=local_rank,
        world_size=world_size,
        devices=devices,
        master_port=master_port,
        model_config=model_config,
        group_topology=group_topology,
        group_id_base=group_id_base,
        ready_queue=ready_queue,
        build_engine=lambda cfg: Engine.load(cfg),
        runtime_ops=runtime_ops,
        runtime_config_extras=driver_config,
        config_cls=NativeRuntimeConfig,
    )
