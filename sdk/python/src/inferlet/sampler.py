"""
Sampler presets mapping to the WIT sampler variant.

The WIT sampler is a discriminated union:
  - multinomial(temperature, seed)
  - top-k(temperature, k)
  - top-p(temperature, p)
  - min-p(temperature, p)
  - top-k-top-p(temperature, k, p)
  - embedding
  - dist(temperature, seed)
"""

from __future__ import annotations

from wit_world.imports.inference import (
    Sampler_Dist,
    Sampler_Embedding,
    Sampler_MinP,
    Sampler_Multinomial,
    Sampler_TopK,
    Sampler_TopKTopP,
    Sampler_TopP,
)
from wit_world.imports.inference import Sampler as WitSampler


class Sampler:
    """Sampler presets.

    Usage::

        Sampler.greedy()
        Sampler.top_p(temperature=0.8, top_p=0.9)
        Sampler.top_k(temperature=0.7, top_k=40)
    """

    __slots__ = ("_variant",)

    def __init__(self, variant: WitSampler) -> None:
        self._variant = variant

    # --- Presets ---

    @classmethod
    def greedy(cls) -> Sampler:
        """Deterministic (greedy) sampling."""
        return cls(Sampler_Multinomial((0.0, 1)))

    @classmethod
    def top_p(cls, temperature: float = 0.6, top_p: float = 0.95) -> Sampler:
        """Nucleus (top-p) sampling."""
        return cls(Sampler_TopP((temperature, top_p)))

    @classmethod
    def top_k(cls, temperature: float = 0.6, top_k: int = 50) -> Sampler:
        """Top-k sampling."""
        return cls(Sampler_TopK((temperature, top_k)))

    @classmethod
    def min_p(cls, temperature: float = 0.6, min_p: float = 0.1) -> Sampler:
        """Min-p sampling."""
        return cls(Sampler_MinP((temperature, min_p)))

    @classmethod
    def top_k_top_p(
        cls,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Sampler:
        """Combined top-k + top-p sampling."""
        return cls(Sampler_TopKTopP((temperature, top_k, top_p)))

    @classmethod
    def multinomial(cls, temperature: float = 1.0, seed: int = 0) -> Sampler:
        """Plain multinomial sampling."""
        return cls(Sampler_Multinomial((temperature, seed)))

    @classmethod
    def embedding(cls) -> Sampler:
        """Embedding output mode (no sampling — returns hidden states)."""
        return cls(Sampler_Embedding())

    @classmethod
    def dist(cls, temperature: float = 1.0, seed: int = 0) -> Sampler:
        """Distribution output mode (returns full probability distribution)."""
        return cls(Sampler_Dist((temperature, seed)))

    def __repr__(self) -> str:
        return f"Sampler({self._variant})"
