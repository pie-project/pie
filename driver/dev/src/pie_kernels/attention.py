"""Attention wrappers — dispatches to flashinfer.attention (CUDA) or pie_kernels.metal (MPS)."""

from __future__ import annotations

import torch

if torch.backends.mps.is_available():
    from .metal import BatchAttentionWithAttentionSinkWrapper  # noqa: F401
else:
    from flashinfer.attention import (  # type: ignore[import-not-found]  # noqa: F401
        BatchAttentionWithAttentionSinkWrapper,
    )
