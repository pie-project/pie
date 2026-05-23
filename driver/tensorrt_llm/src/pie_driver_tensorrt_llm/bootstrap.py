"""Runtime bootstrap helpers for TensorRT-LLM wheel libraries."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


_BOOTSTRAP_ENV = "PIE_TRTLLM_LD_LIBRARY_PATH_BOOTSTRAPPED"

_LIBRARY_RELPATHS = (
    "nvidia/cu13/lib",
    "nvidia/cublas/lib",
    "nvidia/cuda_runtime/lib",
    "nvidia/cuda_nvrtc/lib",
    "nvidia/cudnn/lib",
    "nvidia/cufft/lib",
    "nvidia/curand/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/cusparselt/lib",
    "nvidia/nccl/lib",
    "nvidia/nvjitlink/lib",
    "nvidia/nvshmem/lib",
    "nvidia/nvtx/lib",
    "tensorrt_libs",
    "tensorrt_llm/libs",
)


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in [*site.getsitepackages(), site.getusersitepackages()]:
        if raw:
            path = Path(raw)
            if path not in roots:
                roots.append(path)
    return roots


def cuda_library_dirs() -> list[str]:
    dirs: list[str] = []
    for root in _site_roots():
        for rel in _LIBRARY_RELPATHS:
            path = root / rel
            if path.is_dir():
                value = str(path)
                if value not in dirs:
                    dirs.append(value)
    return dirs


def ensure_cuda_library_path(*, module: str) -> None:
    """Re-exec this module once with NVIDIA wheel library paths exported.

    TensorRT-LLM 1.2.x imports native extensions that depend on CUDA 13
    libraries from NVIDIA Python wheels. Those directories are not always on
    the dynamic linker's default search path, so the subprocess driver exports
    them before importing `tensorrt_llm`.
    """

    if os.name != "posix":
        return

    env = os.environ.copy()
    venv_bin = Path(sys.prefix) / "bin"
    if venv_bin.is_dir():
        current_path = env.get("PATH", "")
        path_parts = [p for p in current_path.split(":") if p]
        venv_bin_s = str(venv_bin)
        if venv_bin_s not in path_parts:
            env["PATH"] = ":".join([venv_bin_s, *path_parts])
            os.environ["PATH"] = env["PATH"]

    dirs = cuda_library_dirs()
    if not dirs:
        return

    current = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    missing = [p for p in dirs if p not in parts]
    if not missing:
        return

    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        return

    env[_BOOTSTRAP_ENV] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([*missing, *parts])
    os.execvpe(sys.executable, [sys.executable, "-m", module, *sys.argv[1:]], env)
