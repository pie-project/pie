"""Worker entry point for the TensorRT-LLM driver."""

from __future__ import annotations

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
    from ._bridge.worker import run_worker
    from . import utils as runtime_ops
    from .config import TensorRTLLMDriverConfig, TensorRTLLMRuntimeConfig
    from .engine import TensorRTLLMEngine

    trt_cfg = TensorRTLLMDriverConfig(**driver_config)

    run_worker(
        local_rank=local_rank,
        world_size=world_size,
        devices=devices,
        master_port=master_port,
        model_config=model_config,
        group_topology=group_topology,
        group_id_base=group_id_base,
        ready_queue=ready_queue,
        build_engine=lambda cfg: TensorRTLLMEngine.load(cfg, trt_cfg),
        runtime_ops=runtime_ops,
        config_cls=TensorRTLLMRuntimeConfig,
    )
