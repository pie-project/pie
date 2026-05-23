"""Torch-bound runtime helpers for the TensorRT-LLM subprocess driver."""

from __future__ import annotations

from typing import Any

import torch


def broadcast_struct(
    data: Any,
    src: int = 0,
    device=None,
    group=None,
    group_id: int | None = None,
) -> Any:
    import torch.distributed as dist

    if isinstance(device, str):
        device = torch.device(device)

    rank = dist.get_rank()
    is_sender = rank == src
    tensors: list = []

    def separate(obj):
        if isinstance(obj, torch.Tensor):
            tensors.append(obj)
            return {
                "__TENSOR__": len(tensors) - 1,
                "shape": obj.shape,
                "dtype": obj.dtype,
            }
        if isinstance(obj, dict):
            return {k: separate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [separate(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(separate(v) for v in obj)
        return obj

    metadata = separate(data) if is_sender else None
    meta_list = [metadata]
    dist.broadcast_object_list(meta_list, src=src, group=group)
    metadata = meta_list[0]

    if not is_sender:
        tensor_specs: dict = {}

        def find_specs(obj):
            if isinstance(obj, dict) and "__TENSOR__" in obj:
                tensor_specs[obj["__TENSOR__"]] = (obj["shape"], obj["dtype"])
            elif isinstance(obj, dict):
                for v in obj.values():
                    find_specs(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    find_specs(v)

        find_specs(metadata)
        tensors = [None] * len(tensor_specs)
        for idx, (shape, dtype) in tensor_specs.items():
            tensors[idx] = torch.empty(shape, dtype=dtype, device=device)

    for t in tensors:
        if is_sender:
            t = t.contiguous()
        dist.broadcast(t, src=src, group=group)

    def reconstruct(obj):
        if isinstance(obj, dict) and "__TENSOR__" in obj:
            return tensors[obj["__TENSOR__"]]
        if isinstance(obj, dict):
            return {k: reconstruct(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [reconstruct(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(reconstruct(v) for v in obj)
        return obj

    return reconstruct(metadata)


def init_distributed(
    tp_local_rank: int,
    tp_degree: int,
    group_id: int,
    master_port: int,
    device: str,
) -> None:
    """Initialize torch.distributed for Pie's TP worker topology.

    TensorRT-LLM support is currently single-rank per model replica. The Rust
    config validator rejects tensor_parallel_size > 1 before launch; this is
    kept for defensive parity with the shared worker contract.
    """
    if tp_degree != 1:
        raise NotImplementedError(
            "pie_driver_tensorrt_llm currently supports tensor_parallel_size = 1. "
            "Use data-parallel devices or a TensorRT-LLM serving endpoint for TP."
        )
    set_device(device)


def set_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.set_device(device)


def cleanup_runtime() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def validate_cuda_devices(devices: list[str]) -> None:
    available_gpus = torch.cuda.device_count()
    for dev in devices:
        if dev and dev.startswith("cuda:"):
            idx = int(dev.split(":")[1])
            if idx >= available_gpus:
                raise RuntimeError(
                    f"device {dev!r} not accessible - only {available_gpus} GPU(s) "
                    f"visible (cuda:0..cuda:{available_gpus - 1}). "
                    f"Check CUDA_VISIBLE_DEVICES."
                )
