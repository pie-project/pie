"""Single re-export point for every vllm internal `pie_driver_vllm` touches.

Pie depends on a handful of vllm symbols that aren't part of vllm's stable
public API — `FlashInferImpl`, `AttentionLayerBase`, `get_attention_context`,
the V1 metadata classes, etc. When vllm bumps and one of these moves or
gets renamed, the fix should be a single edit here, not a sweep across
every callsite.

**Pinned vllm:** see `pie/pyproject.toml [tool.uv.sources] vllm = { path = ..., editable = true }`.
The currently-supported revision is whatever lives at that path. Bumping it
is a deliberate operation: update the path/commit, run the integration
tests under each verified attention backend, and update this file's
exports if any symbol moved.

Importers should `from ._vllm_compat import X` rather than reaching into
vllm directly. Keeps the surface visible at one location and prevents
import-chain mistakes (some submodules pull in heavy deps; we hide those
behind named re-exports).
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Engine / config surface
# ----------------------------------------------------------------------------

from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher


# ----------------------------------------------------------------------------
# Model-executor internals (per-layer attention machinery)
# ----------------------------------------------------------------------------

from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.utils import extract_layer_index


# ----------------------------------------------------------------------------
# V1 attention (backends + metadata + KV binding)
# ----------------------------------------------------------------------------

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.utils import bind_kv_cache


# Per-backend impl classes — imported lazily because vllm's per-backend
# modules pull in heavy deps (flashinfer kernels, fa3 wheels, etc.). Most
# call sites need just one impl, so we expose a getter that imports on
# demand. The getter is in `__all__`; the class itself is *not* re-exported
# at module import time, by design.
def get_flashinfer_impl():
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    return FlashInferImpl


__all__ = [
    "AttentionLayerBase",
    "BatchDescriptor",
    "CUDAGraphMode",
    "CommonAttentionMetadata",
    "CudagraphDispatcher",
    "EngineArgs",
    "bind_kv_cache",
    "ensure_model_parallel_initialized",
    "extract_layer_index",
    "get_attention_context",
    "get_flashinfer_impl",
    "get_forward_context",
    "get_model_loader",
    "graph_capture",
    "init_distributed_environment",
    "set_cudagraph_capturing_enabled",
    "set_current_vllm_config",
    "set_forward_context",
]
