"""End-to-end test for the CUDA conditional graph helper.

Validates the integration pattern we'll use for adapter / prefill /
decode conditionals in the model:

  - Capture base op (out = A) with a PyTorch CUDAGraph
  - Graft a CUDA `if` conditional whose body is `out += B`
  - Drive the conditional with a 1-int host-visible flag tensor
  - Replay with flag=0 → out == A
  - Replay with flag=1 → out == A + B

Skips automatically when CUDA isn't available or the running CUDA
runtime is older than 12.4 (conditional graph nodes were added there).
"""

from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="conditional CUDA graphs require a CUDA device",
)


def _runtime_supports_conditionals() -> bool:
    """`cuGraphConditionalHandleCreate` was added in CUDA 12.4."""
    v = torch.version.cuda
    if v is None:
        return False
    try:
        major, minor = (int(p) for p in v.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (12, 4)


@pytest.mark.skipif(
    not _runtime_supports_conditionals(),
    reason="conditional graph nodes need CUDA 12.4+",
)
def test_if_conditional_drives_branch():
    from pie_kernels.cond_graph import (
        add_if_conditional,
        capture_into,
        get_current_context,
    )

    device = torch.device("cuda:0")
    A = torch.tensor([1.0, 1.0], device=device)
    B = torch.tensor([2.0, 2.0], device=device)
    out = torch.tensor([0.0, 0.0], device=device)
    flag = torch.zeros(1, device=device, dtype=torch.int32)

    stream = torch.cuda.Stream(device=device)
    g = torch.cuda.CUDAGraph(keep_graph=True)

    # Eager warmup so kernels are JITed before capture.
    with torch.cuda.stream(stream):
        out.copy_(A)
        out.add_(B)
        out.zero_()
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        g.capture_begin()
        out.copy_(A)
        g.capture_end()

    raw_graph = g.raw_cuda_graph()
    ctx = get_current_context()
    body = add_if_conditional(raw_graph, ctx, stream, flag)

    with capture_into(stream, body):
        out.add_(B)

    g.instantiate()

    def run(value: int) -> list[float]:
        flag.fill_(value)
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        return out.cpu().tolist()

    assert run(0) == [1.0, 1.0]
    assert run(1) == [3.0, 3.0]

    # No caching — alternating replays should yield alternating outputs.
    for _ in range(5):
        assert run(0) == [1.0, 1.0]
        assert run(1) == [3.0, 3.0]
