"""Helpers for grafting CUDA conditional nodes onto PyTorch CUDAGraphs.

PyTorch's CUDAGraph captures a static sequence of CUDA ops. To express
"branch on a host-driven flag" inside the captured graph — without
combinatorially exploding the number of separately-captured graphs — we
use CUDA's conditional graph nodes (driver API, CUDA 12.4+).

The integration pattern, end to end:

    import torch
    from pie_kernels.cond_graph import (
        get_set_cond_module, add_if_conditional, capture_into,
        get_current_context,
    )

    g = torch.cuda.CUDAGraph(keep_graph=True)
    stream = torch.cuda.Stream()
    flag = torch.zeros(1, device='cuda:0', dtype=torch.int32)

    with torch.cuda.stream(stream):
        g.capture_begin()
        out.copy_(A)            # base ops
        g.capture_end()

    raw_graph = g.raw_cuda_graph()
    ctx = get_current_context()
    body = add_if_conditional(raw_graph, ctx, stream, flag)

    with capture_into(stream, body):
        out.add_(B)             # body ops, fired only when flag != 0

    g.instantiate()
    flag.fill_(1); g.replay()    # out becomes A + B
    flag.fill_(0); g.replay()    # out stays A

The set-conditional helper kernel is a tiny `__global__` that reads one
int from device memory and calls `cudaGraphSetConditional`. It's
JIT-compiled once on first import via `torch.utils.cpp_extension.load_inline`
and cached.

Notes
-----
- Conditional handles must be created **after** the graph exists
  (`cuGraphConditionalHandleCreate(graph, ctx, default, flags)`).
- `node_params.conditional.phGraph_out[0]` is populated by
  `cuGraphAddNode`; `cond_params.phGraph_out` (the input copy) is **not**
  populated — read it back via `node_params`.
- The conditional value is sticky across replays unless created with
  `CU_GRAPH_COND_ASSIGN_DEFAULT`. We always emit a `set_cond_kernel`
  ahead of each conditional so the value is set deterministically per
  replay from a host-visible int tensor.
- `cuStreamBeginCaptureToGraph(graph, [leaf], ...)` lets us append nodes
  to an *existing* graph at a chosen leaf, so we can keep using PyTorch
  ops (instead of building kernel nodes by hand).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from cuda.bindings import driver


_CPP_SOURCES = r"""
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void launch_set_cond_inner(
    unsigned long long, const int*, cudaStream_t);

void launch_set_cond(
    uintptr_t handle_raw, uintptr_t val_ptr, uintptr_t stream
) {
    launch_set_cond_inner(
        handle_raw, (const int*)val_ptr, (cudaStream_t)stream
    );
}
"""

_CUDA_SOURCES = r"""
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void set_cond_from_mem(
    unsigned long long handle_raw, const int* val_ptr
) {
    cudaGraphConditionalHandle h = (cudaGraphConditionalHandle)handle_raw;
    cudaGraphSetConditional(h, (unsigned int)(*val_ptr));
}

extern "C" void launch_set_cond_inner(
    unsigned long long handle_raw,
    const int* val_ptr,
    cudaStream_t stream
) {
    set_cond_from_mem<<<1, 1, 0, stream>>>(handle_raw, val_ptr);
}
"""

_mod = None


def get_set_cond_module():
    """Return the JIT-compiled set-conditional helper module.

    Compiles on first call; subsequent calls return the cached module.
    """
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(
            name="pie_cond_helper",
            cpp_sources=_CPP_SOURCES,
            cuda_sources=_CUDA_SOURCES,
            functions=["launch_set_cond"],
            verbose=False,
            with_cuda=True,
        )
    return _mod


def get_current_context():
    """Return the current CUDA driver context (PyTorch's primary ctx)."""
    err, ctx = driver.cuCtxGetCurrent()
    _check(err, "cuCtxGetCurrent")
    return ctx


def _check(err, where: str) -> None:
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{where}: {err}")


def _last_node(raw_graph):
    """Return the last node in the graph's enumeration order.

    For a linearly-built graph (capture, add, capture, ...) this is the
    most recently added leaf, which is what we want when chaining a new
    dependent node.
    """
    err, _, n = driver.cuGraphGetNodes(raw_graph, 0)
    _check(err, "cuGraphGetNodes (count)")
    err, nodes, _ = driver.cuGraphGetNodes(raw_graph, n)
    _check(err, "cuGraphGetNodes (get)")
    if not nodes:
        raise RuntimeError(
            "graph has no nodes; capture at least one op before adding a "
            "conditional"
        )
    return nodes[-1]


def add_if_conditional(
    raw_graph,
    ctx,
    stream: torch.cuda.Stream,
    flag_tensor: torch.Tensor,
    default_value: int = 0,
):
    """Append a CUDA `if` conditional to ``raw_graph`` after its current leaf.

    Builds the dependency chain::

        current_leaf  →  set_cond_kernel(handle, flag_tensor)  →  IF(handle)

    The body sub-graph is returned (empty); populate it via
    :func:`capture_into`.

    Parameters
    ----------
    raw_graph : CUgraph
        The captured PyTorch graph to mutate. Must originate from a
        :class:`torch.cuda.CUDAGraph` created with ``keep_graph=True``.
    ctx : CUcontext
        The CUDA context backing the graph. Use :func:`get_current_context`.
    stream : torch.cuda.Stream
        Stream used to capture the set-conditional kernel into the parent
        graph. Reusing the stream that captured the base graph is fine.
    flag_tensor : torch.Tensor
        1-element ``int32`` tensor on the same device. Write 0 to skip
        the body, non-zero to execute it. Read each replay by the
        set-cond kernel.
    default_value : int
        Initial conditional value at handle creation; used until the
        first set-cond kernel runs (this almost never matters because we
        always run set-cond ahead of the conditional).

    Returns
    -------
    body_graph : CUgraph
        Empty sub-graph to be populated via :func:`capture_into`.
    """
    if flag_tensor.numel() != 1 or flag_tensor.dtype != torch.int32:
        raise ValueError(
            "flag_tensor must be a 1-element int32 cuda tensor"
        )

    err, handle = driver.cuGraphConditionalHandleCreate(
        raw_graph, ctx, default_value, 0
    )
    _check(err, "cuGraphConditionalHandleCreate")

    leaf = _last_node(raw_graph)

    # Capture the set-cond kernel onto the parent at `leaf` via stream
    # capture. This makes set_cond a child of `leaf`.
    (err,) = driver.cuStreamBeginCaptureToGraph(
        stream.cuda_stream,
        raw_graph,
        [leaf],
        None,
        1,
        driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )
    _check(err, "cuStreamBeginCaptureToGraph (set_cond)")

    mod = get_set_cond_module()
    with torch.cuda.stream(stream):
        mod.launch_set_cond(
            int(handle), flag_tensor.data_ptr(), stream.cuda_stream
        )

    err, _ = driver.cuStreamEndCapture(stream.cuda_stream)
    _check(err, "cuStreamEndCapture (set_cond)")

    set_cond_node = _last_node(raw_graph)

    # Add the IF conditional, dependent on set_cond_node so set_cond runs
    # before the conditional is evaluated (NOT relying on implicit DAG
    # ordering).
    cond_params = driver.CUDA_CONDITIONAL_NODE_PARAMS()
    cond_params.handle = handle
    cond_params.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
    cond_params.size = 1
    cond_params.ctx = ctx

    node_params = driver.CUgraphNodeParams()
    node_params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    node_params.conditional = cond_params

    err, _cond_node = driver.cuGraphAddNode(
        raw_graph, [set_cond_node], 1, node_params
    )
    _check(err, "cuGraphAddNode (CONDITIONAL_IF)")

    # The body graph is on `node_params.conditional` (the wrapper that
    # the call mutated), NOT on `cond_params` (the unwrapped input copy).
    return node_params.conditional.phGraph_out[0]


@contextmanager
def capture_into(
    stream: torch.cuda.Stream, body_graph,
) -> Iterator[None]:
    """Context manager: capture stream ops into the given body sub-graph.

    Use the result of :func:`add_if_conditional` as ``body_graph``. Inside
    the ``with`` block, any PyTorch op issued on the stream is captured
    into the conditional's body.
    """
    (err,) = driver.cuStreamBeginCaptureToGraph(
        stream.cuda_stream,
        body_graph,
        None,
        None,
        0,
        driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )
    _check(err, "cuStreamBeginCaptureToGraph (body)")
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        err, _ = driver.cuStreamEndCapture(stream.cuda_stream)
        _check(err, "cuStreamEndCapture (body)")
