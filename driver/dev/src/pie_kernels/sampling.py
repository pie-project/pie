"""Sampling primitives — dispatches to flashinfer (CUDA) or the Metal impl."""

from __future__ import annotations

import torch

if torch.backends.mps.is_available():
    from .metal.sampling import (  # noqa: F401
        sampling_from_probs,
        top_p_sampling_from_probs,
        top_k_sampling_from_probs,
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
else:
    from flashinfer.sampling import (  # type: ignore[import-not-found]  # noqa: F401
        sampling_from_probs,
        top_p_sampling_from_probs,
        top_k_sampling_from_probs,
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
