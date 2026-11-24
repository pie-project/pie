"""The batch: token rows to train on, with a loss weight for every prediction.

Rows are unpadded documents. The trainer pads them at the end to one length T
with weight 0, so causal attention keeps pads out of every real position.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class Batch:
    """Rows of token ids and the loss weight of predicting each next token."""

    tokens: list[list[int]]
    """One document per row, unpadded."""
    weights: list[list[float]]
    """len(row) - 1 per row: the loss weight of predicting tokens[i+1]."""
    ref_logp: torch.Tensor | None = None
    """[rows, T-1] plain-arm log-probs, filled by `forward_backward(..., reference=True)`."""
    sampler_logp: torch.Tensor | None = None
    """[rows, T-1] log-probs the samples were drawn at, for importance_sampling and ppo."""


def pad(tokens: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows as one [rows, T] id tensor, padded at the end with 0, and the
    [rows, T-1] mask of the predictions that are real."""
    if not tokens or min(len(row) for row in tokens) < 2:
        raise ValueError("every row needs at least two tokens")
    T = max(len(row) for row in tokens)
    ids = torch.zeros(len(tokens), T, dtype=torch.long)
    real = torch.zeros(len(tokens), T - 1, dtype=torch.bool)
    for i, row in enumerate(tokens):
        ids[i, : len(row)] = torch.tensor(row)
        real[i, : len(row) - 1] = True
    return ids.to(device), real.to(device)


def padded(batch: Batch, device) -> tuple[Batch, torch.Tensor]:
    """The batch as a loss sees it, with tokens [rows, T] and weights [rows, T-1]
    as tensors on `device` (pads weigh 0), and the mask of real predictions."""
    ids, real = pad(batch.tokens, device)
    weights = torch.zeros(real.shape, dtype=torch.float32)
    for i, (row, w) in enumerate(zip(batch.tokens, batch.weights, strict=True)):
        if len(w) != len(row) - 1:
            raise ValueError(f"row {i}: {len(w)} weights for {len(row)} tokens; a row of n tokens has n - 1")
        weights[i, : len(w)] = torch.tensor(w, dtype=torch.float32)
    return replace(batch, tokens=ids, weights=weights.to(device)), real
