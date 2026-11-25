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
- `dpo`: direct preference optimization. Rows come in (preferred, rejected)
  pairs; a row's reward is its weighted log-prob gain over the reference, the
  plain arm of the same program, so the reference costs no second model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

PPO_CLIP = 0.2
"""How far `ppo` lets the ratio move from 1 before it stops pushing."""
DPO_BETA = 0.1
"""How sharply `dpo` turns a reward margin into a preference."""


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


def dpo(logp: torch.Tensor, batch) -> torch.Tensor:
    """-log sigmoid(beta (gain of the preferred row - gain of the rejected row)),
    each gain the row's weighted log-prob over `batch.ref_logp`, averaged over pairs."""
    if batch.ref_logp is None:
        raise ValueError("dpo needs batch.ref_logp, the plain arm's log-probs")
    if logp.shape[0] % 2:
        raise ValueError(f"dpo reads rows as (preferred, rejected) pairs, and this batch has {logp.shape[0]} rows")
    gain = ((logp - batch.ref_logp.to(logp)) * batch.weights).sum(-1).view(-1, 2)
    return -F.logsigmoid(DPO_BETA * (gain[:, 0] - gain[:, 1])).mean()


FUNCTIONS = {"cross_entropy": cross_entropy, "importance_sampling": importance_sampling, "ppo": ppo, "dpo": dpo}
"""Each built-in loss by name."""

LOSSES = frozenset(FUNCTIONS)
"""The names `forward_backward` takes for `loss`."""

REFERENCED = frozenset({"dpo"})
"""The built-in losses that read the plain arm's log-probs."""


def _sampled(batch) -> torch.Tensor:
    if batch.sampler_logp is None:
        raise ValueError("this loss needs batch.sampler_logp, the log-probs the samples were drawn at")
    return batch.sampler_logp


def _count(batch) -> torch.Tensor:
    return (batch.weights != 0).sum().clamp_min(1)
