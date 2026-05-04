"""pie_kernels: unified kernel interface.

Dispatches attention, RoPE, KV-cache, MoE, and related ops to either the
upstream `flashinfer` package (CUDA) or `pie_kernels.metal` (MPS) based on
device availability. Callers should write device-agnostic code:

    import pie_kernels as ops
    wrapper = ops.BatchAttentionWithAttentionSinkWrapper(...)

Symbol resolution goes through the selected backend via ``__getattr__``, so the
full backend API surface is available without per-symbol re-exports. CUDA-only
symbols (e.g. ``flashinfer.fp4_quantization``) are not routed through this
module — import them directly from ``flashinfer`` inside code paths that are
already CUDA-guarded.

Submodules ``pie_kernels.sampling`` and ``pie_kernels.rand_mv`` provide
explicit dispatchers for those APIs.
"""

from __future__ import annotations

import torch

if torch.backends.mps.is_available():
    from . import metal as _backend
else:
    import flashinfer as _backend  # type: ignore[import-not-found]


def __getattr__(name: str):
    try:
        return getattr(_backend, name)
    except AttributeError:
        raise AttributeError(
            f"module 'pie_kernels' has no attribute {name!r} "
            f"(backend: {_backend.__name__})"
        ) from None
