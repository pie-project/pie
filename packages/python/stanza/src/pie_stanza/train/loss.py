"""The losses: the built-in objectives `Trainer.forward_backward` takes by name.

Each loss is a function of the per-token log-probs `logp` [rows, T-1] and the
padded batch (its `weights` a [rows, T-1] tensor). A custom loss is any
function of the same two arguments that returns a scalar.

- `cross_entropy`: the weighted mean of -logp, over the sum of the weights.
- `importance_sampling`: -exp(logp - sampler_logp) * advantage, where the
  weights carry each token's advantage, averaged over tokens with a nonzero
  weight. It needs `batch.sampler_logp`.
- `ppo`: the same ratio clipped to [0.8, 1.2], the pessimistic of the clipped
  and unclipped terms, averaged the same way.
"""

from __future__ import annotations

import torch

PPO_CLIP = 0.2


def cross_entropy(logp: torch.Tensor, batch) -> torch.Tensor:
    """Weighted mean negative log-likelihood of the target tokens."""
    return -(logp * batch.weights).sum() / batch.weights.sum().clamp_min(1e-12)


def importance_sampling(logp: torch.Tensor, batch) -> torch.Tensor:
    """Policy gradient off samples drawn at `batch.sampler_logp`."""
    ratio = (logp - _sampled(batch)).exp()
    return -(ratio * batch.weights).sum() / _count(batch)


def ppo(logp: torch.Tensor, batch) -> torch.Tensor:
    """Clipped-ratio policy gradient off samples drawn at `batch.sampler_logp`."""
    ratio = (logp - _sampled(batch)).exp()
    a = batch.weights
    clipped = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP)
    return -torch.minimum(ratio * a, clipped * a).sum() / _count(batch)


FUNCTIONS = {"cross_entropy": cross_entropy, "importance_sampling": importance_sampling, "ppo": ppo}
"""Each built-in loss by name."""

LOSSES = frozenset(FUNCTIONS)
"""The names `forward_backward` takes for `loss`."""


def _sampled(batch) -> torch.Tensor:
    if batch.sampler_logp is None:
        raise ValueError("this loss needs batch.sampler_logp, the log-probs the samples were drawn at")
    if batch.sampler_logp.shape != batch.weights.shape:
        raise ValueError(f"batch.sampler_logp is {list(batch.sampler_logp.shape)}, the batch is {list(batch.weights.shape)}")
    return batch.sampler_logp.to(batch.weights)


def _count(batch) -> torch.Tensor:
    return (batch.weights != 0).sum().clamp_min(1)
