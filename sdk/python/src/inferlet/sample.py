"""
Samplers and probes for forward-pass output slots.

The host's ``forward-pass.sampler`` slot folds two unrelated concerns into
a single WIT variant. This module keeps them distinct:

* :class:`Sampler` picks a token. Use as
  ``forward.sample(indices, Sampler.argmax())``.
* Probes (:class:`Logits`, :class:`Distribution`, :class:`Logprob`,
  :class:`Logprobs`, :class:`Entropy`) read shape information without
  sampling. Use as
  ``forward.probe(index, Distribution(temperature=1.0, k=32))``.

Both attach to the same forward-pass slot, so a single :class:`Forward`
can mix samplers and probes freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from wit_world.imports.inference import (
    Sampler_Dist,
    Sampler_Entropy,
    Sampler_Logprob,
    Sampler_Logprobs,
    Sampler_MinP,
    Sampler_Multinomial,
    Sampler_RawLogits,
    Sampler_TopK,
    Sampler_TopKTopP,
    Sampler_TopP,
)
from wit_world.imports.inference import Sampler as _WitSampler


# =============================================================================
# Sampler — token-producing
# =============================================================================


class Sampler:
    """Token-producing sampler.

    Construct via class methods — never directly. Each constructor picks one
    sampling strategy::

        Sampler.argmax()
        Sampler.top_p(0.6, 0.95)
        Sampler.top_k(0.7, 40)
        Sampler.min_p(0.6, 0.1)
        Sampler.top_k_top_p(0.6, 40, 0.95)
        Sampler.multinomial(1.0)
    """

    __slots__ = ("_variant",)

    def __init__(self, variant: _WitSampler) -> None:
        self._variant = variant

    @classmethod
    def argmax(cls) -> Sampler:
        """Greedy / argmax — deterministic. Recommended for grammar-constrained
        generation where most masked positions have only a handful of valid
        tokens and stochastic sampling rarely helps."""
        return cls(Sampler_TopP((0.0, 1.0)))

    @classmethod
    def top_p(cls, temperature: float = 0.6, p: float = 0.95) -> Sampler:
        """Top-p (nucleus) sampling. ``temperature = 0.0`` collapses to argmax."""
        return cls(Sampler_TopP((temperature, p)))

    @classmethod
    def top_k(cls, temperature: float = 0.6, k: int = 40) -> Sampler:
        """Top-k sampling: sample from the top ``k`` tokens by probability."""
        return cls(Sampler_TopK((temperature, k)))

    @classmethod
    def min_p(cls, temperature: float = 0.6, p: float = 0.1) -> Sampler:
        """Min-p sampling: keep tokens with ``probability >= p * max_prob``."""
        return cls(Sampler_MinP((temperature, p)))

    @classmethod
    def top_k_top_p(
        cls, temperature: float = 0.6, k: int = 40, p: float = 0.95
    ) -> Sampler:
        """Combined top-k + top-p: first restrict to top ``k``, then nucleus ``p``."""
        return cls(Sampler_TopKTopP((temperature, k, p)))

    @classmethod
    def multinomial(cls, temperature: float = 1.0, draws: int = 1) -> Sampler:
        """Plain multinomial after temperature scaling. ``draws`` is a
        per-sample multiplier (typically 1)."""
        return cls(Sampler_Multinomial((temperature, draws)))

    def __repr__(self) -> str:
        v = self._variant
        kind = type(v).__name__.removeprefix("Sampler_")
        if hasattr(v, "value"):
            return f"Sampler.{kind}({v.value})"
        return f"Sampler.{kind}()"


# =============================================================================
# Probes — distribution access (no sampling)
# =============================================================================
#
# Each probe is a frozen dataclass that doubles as both spec and runtime
# marker. ``Forward.probe(idx, X())`` consumes one of these and returns a
# handle whose accessor on :class:`Output` decodes the right shape.


@dataclass(frozen=True, slots=True)
class Logits:
    """Pre-softmax, untemperatured logits as packed native-endian f32 bytes
    (length ``vocab_size * 4``). Decode via
    ``np.frombuffer(buf, dtype=np.float32)``."""

    def _to_wit(self) -> _WitSampler:
        return Sampler_RawLogits()


@dataclass(frozen=True, slots=True)
class Distribution:
    """Top-``k`` token IDs paired with probabilities (post-softmax,
    temperature-scaled). ``k = 0`` returns the full vocabulary."""

    temperature: float = 1.0
    k: int = 0

    def _to_wit(self) -> _WitSampler:
        return Sampler_Dist((self.temperature, self.k))


@dataclass(frozen=True, slots=True)
class Logprob:
    """``log p(token | context)`` at this position, no temperature scaling.
    Returned as a length-1 logprob list — read with ``output.logprobs(h)``."""

    token: int

    def _to_wit(self) -> _WitSampler:
        return Sampler_Logprob(self.token)


@dataclass(frozen=True, slots=True)
class Logprobs:
    """``log p(t | context)`` for each ``t`` in ``tokens``, no temperature
    scaling. Returned as a length-K list in the requested order.

    Accepts any iterable for ``tokens`` — converted to a tuple for
    hashability::

        Logprobs([1, 2, 3])          # list works
        Logprobs((1, 2, 3))          # tuple works
        Logprobs(range(10))          # any iterable works
    """

    tokens: tuple[int, ...] = field(default_factory=tuple)

    def __init__(self, tokens: Iterable[int] = ()) -> None:
        # Frozen dataclass requires hashable fields, so coerce iterable
        # into a tuple. Use object.__setattr__ to bypass frozen check.
        object.__setattr__(self, "tokens", tuple(tokens))

    def _to_wit(self) -> _WitSampler:
        return Sampler_Logprobs(list(self.tokens))


@dataclass(frozen=True, slots=True)
class Entropy:
    """Shannon entropy ``H(p) = -sum(p log p)`` of the unscaled distribution."""

    def _to_wit(self) -> _WitSampler:
        return Sampler_Entropy()
