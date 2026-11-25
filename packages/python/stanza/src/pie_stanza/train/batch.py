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
    sampler_logp: list[list[float]] | None = None
    """len(row) - 1 per row: the log-prob the sampler drew each next token at,
    for importance_sampling and ppo (pie's, or `Trainer.logprobs`'s)."""


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
    """The batch as a loss sees it, with tokens [rows, T], and weights and
    sampler_logp [rows, T-1], as tensors on `device` (pads weigh 0), and the
    mask of real predictions."""
    ids, real = pad(batch.tokens, device)
    weights = _rows(batch.tokens, batch.weights, real.shape, "weights").to(device)
    sampled = None
    if batch.sampler_logp is not None:
        sampled = _rows(batch.tokens, batch.sampler_logp, real.shape, "sampler_logp").to(device)
    return replace(batch, tokens=ids, weights=weights, sampler_logp=sampled), real


def _rows(tokens: list[list[int]], values: list[list[float]], shape, what: str) -> torch.Tensor:
    """Per-prediction values, one list a row, as one [rows, T-1] tensor padded with 0."""
    out = torch.zeros(shape, dtype=torch.float32)
    for i, (row, v) in enumerate(zip(tokens, values, strict=True)):
        if len(v) != len(row) - 1:
            raise ValueError(f"row {i}: {len(v)} {what} for {len(row)} tokens; a row of n tokens has n - 1")
        out[i, : len(v)] = torch.tensor(v, dtype=torch.float32)
    return out
