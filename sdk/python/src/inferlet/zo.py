"""
Zeroth-order optimization interface — ``pie:zo``.

Provides evolutionary optimization for adapter weights.
"""

from __future__ import annotations

from wit_world.imports import zo as _zo

from .forward import ForwardPass
from .adapter import Adapter


def adapter_seed(forward_pass: ForwardPass, seed: int) -> None:
    """Set the adapter seed for a forward pass."""
    _zo.adapter_seed(forward_pass._handle, seed)


def initialize(
    adapter: Adapter,
    rank: int,
    alpha: float,
    population_size: int,
    mu_fraction: float,
    initial_sigma: float,
) -> None:
    """Initialize zeroth-order optimization for an adapter.

    Args:
        adapter: The adapter to optimize.
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        population_size: Number of candidates per generation.
        mu_fraction: Fraction of top candidates to keep.
        initial_sigma: Initial mutation step size.
    """
    _zo.initialize(adapter._handle, rank, alpha, population_size, mu_fraction, initial_sigma)


def update(
    adapter: Adapter,
    scores: list[float],
    seeds: list[int],
    max_sigma: float,
) -> None:
    """Update the adapter using zeroth-order optimization.

    Args:
        adapter: The adapter to update.
        scores: Fitness scores for each candidate.
        seeds: Random seeds used for each candidate.
        max_sigma: Maximum mutation step size.
    """
    _zo.update(adapter._handle, scores, seeds, max_sigma)
