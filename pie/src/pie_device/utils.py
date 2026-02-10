import os
import platform
import sys
import traceback
from pathlib import Path
import psutil
from typing import Any


import torch


# CPU process group for GLOO-based metadata broadcasts
_cpu_group = None


def is_apple_silicon() -> bool:
    """Check if running on macOS with Apple Silicon (M1/M2/M3/M4).

    Returns:
        True if running on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.processor() == "arm"


def resolve_cache_dir(cache_dir: str | None) -> str:
    """Resolve the cache directory using CLI arg > env var > default.

    - Windows: Uses %LOCALAPPDATA%/pie
    - Unix (Linux, macOS, etc.): Uses ~/.cache/pie for Docker compatibility
    """
    if cache_dir:
        return cache_dir

    if "PIE_HOME" in os.environ:
        return os.environ["PIE_HOME"]

    # Platform-specific cache directory (matches C++ backend in utils.hpp)
    if sys.platform == "win32":
        # Windows: Use LOCALAPPDATA for cache (standard on Windows)
        local_appdata = os.environ.get("LOCALAPPDATA")
        if not local_appdata:
            raise RuntimeError(
                "Could not determine cache directory. "
                "Please set %LOCALAPPDATA% or specify --cache-dir"
            )
        return str(Path(local_appdata) / "pie")
    else:
        # Unix (Linux, macOS): Use ~/.cache for Docker volume mount compatibility
        home = Path.home()
        return str(home / ".cache" / "pie")


def resolve_adapter_path(adapter_path: str | None) -> str:
    """Resolve the adapter storage path using CLI arg > env var > default.

    Default path is ~/.pie/adapters/
    """
    if adapter_path:
        return os.path.expanduser(adapter_path)

    if "PIE_ADAPTER_PATH" in os.environ:
        return os.path.expanduser(os.environ["PIE_ADAPTER_PATH"])

    # Default: ~/.pie/adapters/
    home = Path.home()
    return str(home / ".pie" / "adapters")


def terminate(msg: str) -> None:
    """Terminate the program with a message."""
    print(f"\n[!!!] {msg} Terminating.", file=sys.stderr)
    traceback.print_exc()
    os._exit(1)


def get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def get_available_memory(devices: list[torch.device], rank: int = 0) -> int:
    device = devices[rank]

    is_cuda = device.type == "cuda"
    is_cpu = device.type in ("cpu", "mps")

    # Clear cache on all CUDA devices to get a better measurement
    if is_cuda and torch.cuda.is_available():
        with torch.cuda.device(device.index):
            torch.cuda.empty_cache()

    total_free_bytes = []

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
    device: torch.device | None = None,
    group: "torch.distributed.ProcessGroup | None" = None,
    group_id: int | None = None,
) -> Any:
    """
    Broadcast a structure of data with embedded tensors efficiently.

    Metadata is broadcast via GLOO (CPU), tensors via NCCL (GPU).
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    is_sender = rank == src
    tensors = []

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
        tensor_specs = {}

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
