"""
`inferlet.eta.diffusion` — the diffusion pass's reading (`Mode`) and the
reference sampler pieces, a port of `inferlet::eta::diffusion`. The pass
itself is `ForwardPass(ForwardKind.DIFFUSION)` with `canvas(Mode.DENOISE)`
and `self_conditioning(rows, weights)`.
"""

from __future__ import annotations

from wit_world.imports import forward_diffusion as _wit_diffusion

from .value import Tensor, and_, cast, cumsum, eq, iota, le, lt, neg, reduce_sum, scatter_set, sort_desc
from .ir import Dtype

Mode = _wit_diffusion.Mode
"""`ENCODE` (causal, writes the sequence) or `DENOISE` (bidirectional over
the canvas, scratch KV)."""

__all__ = ["Mode", "linear_temperature", "entropy_bound_accept", "stable_and_confident"]


def linear_temperature(remaining: int, max_steps: int, t_max: float, t_min: float) -> float:
    """The reference schedule `t_min + (t_max - t_min) * remaining / max`,
    with `remaining` counting DOWN from `max_steps` to 1. Host arithmetic;
    it reaches the program through a control channel the host `set`s."""
    return t_min + (t_max - t_min) * (remaining / max(max_steps, 1))


def entropy_bound_accept(entropy: Tensor, bound: float) -> Tensor:
    """The entropy-bound acceptance rule over one canvas: accept the
    lowest-entropy positions while `sum(H) - max(H) <= bound` over the
    accepted set (Ben-Hamu et al., 2505.24857). `entropy` is `[n]` f32; the
    answer is `[n]` bool in canvas order."""
    n = entropy.shape[0]
    neg_sorted, order = sort_desc(neg(entropy))
    sorted_ = neg(neg_sorted)
    below = le(cumsum(sorted_) - sorted_, bound)
    none = lt(iota(n), 0)
    return scatter_set(none, order, below)


def stable_and_confident(argmax: Tensor, previous: Tensor, entropy: Tensor, threshold: float) -> Tensor:
    """The reference stopping rule for one canvas: the argmax canvas did not
    move since the previous step AND the mean per-position entropy is under
    `threshold`. `argmax`/`previous` are `[n]` i32, `entropy` `[n]` f32; the
    answer is a bool scalar."""
    n = argmax.shape[0]
    unchanged = reduce_sum(cast(eq(argmax, previous), Dtype.I32))
    stable = eq(unchanged, n)
    mean = reduce_sum(entropy) / float(n)
    return and_(stable, lt(mean, threshold))
