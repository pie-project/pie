"""Torch-bound runtime helpers for the native driver.

Bridge ships a flavor-agnostic `._bridge.utils` that holds only
torch-free helpers (`is_apple_silicon`, `terminate`). Anything that
touches `torch.distributed` or `torch.cuda` lives here so the bridge
stays torch-free.
"""

from __future__ import annotations

from typing import Any

import os
import psutil
import torch


# CPU process group cache for GLOO-based metadata broadcasts. Populated
# by the worker when it stands up a CPU companion group; None means
# "use the default PG (NCCL)" inside `broadcast_struct`.
_cpu_group = None


def devices_use_system_topology(
    devices: list[str], tp_degree: int | None = None
) -> bool:
    """Return True when any TP GPU pair only meets at NVML SYSTEM level."""
    if tp_degree is None:
        tp_degree = len(devices)
    if tp_degree <= 1:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False

    try:
        import pynvml

        indices: list[int] = []
        for device in devices[:tp_degree]:
            dev = torch.device(device)
            if dev.type != "cuda":
                return False
            idx = 0 if dev.index is None else int(dev.index)
            if idx >= torch.cuda.device_count():
                return False
            indices.append(idx)

        if len(set(indices)) < 2:
            return False

        pynvml.nvmlInit()
        handles = []
        for idx in indices:
            prop = torch.cuda.get_device_properties(idx)
            bus_id = (
                f"{prop.pci_domain_id:08x}:"
                f"{prop.pci_bus_id:02x}:"
                f"{prop.pci_device_id:02x}.0"
            )
            handles.append(pynvml.nvmlDeviceGetHandleByPciBusId(bus_id.encode()))

        system_level = pynvml.NVML_TOPOLOGY_SYSTEM
        for i in range(len(handles)):
            for j in range(i + 1, len(handles)):
                level = pynvml.nvmlDeviceGetTopologyCommonAncestor(
                    handles[i], handles[j],
                )
                if level >= system_level:
                    return True
    except Exception:
        return False
    finally:
        try:
            pynvml.nvmlShutdown()  # type: ignore[name-defined]
        except Exception:
            pass

    return False


def configure_distributed_environment(tp_degree: int, devices: list[str]) -> bool:
    """Set distributed env defaults before torch.distributed is initialized."""
    system_topology = devices_use_system_topology(devices, tp_degree)
    if system_topology:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    return system_topology


def get_device_sm() -> int:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def get_available_memory(devices: list, rank: int = 0) -> int:
    """Free bytes on the local rank's device, min-reduced across the TP group."""
    device = devices[rank]
    if isinstance(device, str):
        device = torch.device(device)

    is_cuda = device.type == "cuda"
    is_cpu = device.type in ("cpu", "mps")

    # Clear cache on the CUDA device for a better measurement
    if is_cuda and torch.cuda.is_available():
        with torch.cuda.device(device.index):
            torch.cuda.empty_cache()

    total_free_bytes: int | list = []

    if is_cuda:
        if get_device_sm() in (87, 110, 121):  # Orin, Thor, Spark
            total_free_bytes = psutil.virtual_memory().available
        else:
            total_free_bytes, _ = torch.cuda.mem_get_info(device)

    if is_cpu:
        total_free_bytes = psutil.virtual_memory().available

    if not total_free_bytes:
        raise RuntimeError("No supported devices found in the devices list")

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and len(devices) > 1
    ):
        tensor = torch.tensor(total_free_bytes, dtype=torch.int64)
        if is_cuda:
            tensor = tensor.to(device)

        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        total_free_bytes = int(tensor.item())

    return total_free_bytes


def broadcast_struct(
    data: Any,
    src: int = 0,
    device=None,
    group=None,
    group_id: int | None = None,
) -> Any:
    """Broadcast a structure of data with embedded tensors efficiently.

    Metadata is broadcast via GLOO (CPU), tensors via NCCL (GPU).
    """
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
        elif isinstance(obj, dict):
            return {k: separate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [separate(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(separate(v) for v in obj)
        else:
            return obj

    # 1. Prepare metadata on source
    metadata = None
    if is_sender:
        metadata = separate(data)

    # 2. Broadcast metadata via GLOO
    meta_list = [metadata]
    dist.broadcast_object_list(meta_list, src=src, group=group or _cpu_group)
    metadata = meta_list[0]

    # 3. Prepare tensors for broadcast (receiver allocates buffers)
    if not is_sender:
        tensor_specs: dict = {}

        def find_specs(obj):
            if isinstance(obj, dict) and "__TENSOR__" in obj:
                tensor_specs[obj["__TENSOR__"]] = (obj["shape"], obj["dtype"])
            elif isinstance(obj, dict):
                for v in obj.values():
                    find_specs(v)
            elif isinstance(obj, list):
                for v in obj:
                    find_specs(v)
            elif isinstance(obj, tuple):
                for v in obj:
                    find_specs(v)

        find_specs(metadata)
        tensors = [None] * len(tensor_specs)
        for idx, (shape, dtype) in tensor_specs.items():
            tensors[idx] = torch.empty(shape, dtype=dtype, device=device)

    # 4. Broadcast tensors via NCCL
    for t in tensors:
        if is_sender:
            t = t.contiguous()
        dist.broadcast(t, src=src, group=group)

    # 5. Reconstruct
    def reconstruct(obj):
        if isinstance(obj, dict) and "__TENSOR__" in obj:
            return tensors[obj["__TENSOR__"]]
        elif isinstance(obj, dict):
            return {k: reconstruct(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [reconstruct(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(reconstruct(v) for v in obj)
        else:
            return obj

    return reconstruct(metadata)


def init_distributed(
    tp_local_rank: int,
    tp_degree: int,
    group_id: int,
    master_port: int,
    device: str,
) -> None:
    """Initialize torch.distributed for one TP group.

    Each DP replica brings up its own torch.distributed *world* whose size
    is just `tp_degree`. The default process group thus equals the TP
    group — no global PG, no subgroups. Cross-replica coordination happens
    at the Rust runtime / RPC layer (request routing + KV swap framing),
    not via torch.distributed.

    The FileStore path is namespaced by `(master_port, group_id)` so each
    TP group rendezvous independently when multiple groups share a host.
    """
    import datetime
    import warnings
    import torch.distributed as dist

    torch.cuda.set_device(device)

    warnings.filterwarnings(
        "ignore", message=".*barrier.*device under current context.*"
    )

    store = dist.FileStore(
        f"/tmp/pie_dist_store_{master_port}_g{group_id}", tp_degree
    )
    timeout = datetime.timedelta(seconds=300)

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    device_id = None
    if device.startswith("cuda:"):
        device_id = torch.device(device)

    dist.init_process_group(
        backend,
        store=store,
        rank=tp_local_rank,
        world_size=tp_degree,
        timeout=timeout,
        device_id=device_id,
    )

    global _cpu_group
    if backend == "nccl" and tp_degree > 1:
        _cpu_group = dist.new_group(
            ranks=list(range(tp_degree)),
            backend="gloo",
            timeout=timeout,
        )


def set_device(device: str) -> None:
    """Pin the current thread to `device` (no-op for non-CUDA strings)."""
    if device.startswith("cuda"):
        torch.cuda.set_device(device)


def cleanup_runtime() -> None:
    """Best-effort post-worker CUDA cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def cleanup_distributed() -> None:
    import torch.distributed as dist

    global _cpu_group
    _cpu_group = None
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def validate_cuda_devices(devices: list[str]) -> None:
    """Raise if any `cuda:N` device in `devices` exceeds the visible GPU count."""
    available_gpus = torch.cuda.device_count()
    for dev in devices:
        if dev and dev.startswith("cuda:"):
            idx = int(dev.split(":")[1])
            if idx >= available_gpus:
                raise RuntimeError(
                    f"device {dev!r} not accessible — only {available_gpus} GPU(s) "
                    f"visible (cuda:0..cuda:{available_gpus - 1}). "
                    f"Check CUDA_VISIBLE_DEVICES."
                )
