"""Batched randn matmul / generate — dispatches to Triton (CUDA) or Metal (MPS).

``RAND_MV_AVAILABLE`` reports whether either backend is usable; callers should
gate invocations on it rather than catching ``NotImplementedError`` at runtime.
"""

from __future__ import annotations

import torch

RAND_MV_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()

if not RAND_MV_AVAILABLE:
    def batched_randn_matmul(*args, **kwargs):
        raise RuntimeError(
            "rand_mv.batched_randn_matmul requires CUDA or MPS; "
            "neither backend is available on this platform."
        )

    def batched_randn_generate(*args, **kwargs):
        raise RuntimeError(
            "rand_mv.batched_randn_generate requires CUDA or MPS; "
            "neither backend is available on this platform."
        )

    def run_tests():
        raise RuntimeError(
            "rand_mv tests require CUDA or MPS."
        )

elif torch.cuda.is_available():
    from .cuda.rand_mv_new import (  # noqa: F401
        batched_randn_matmul,
        batched_randn_matmul_sectioned,
        batched_randn_matmul_multi_input_sectioned,
        batched_randn_generate,
        run_tests,
    )

else:
    # MPS path
    from .metal.rand_mv import (  # noqa: F401
        batched_randn_matmul,
        batched_randn_generate,
    )

    def run_tests():
        raise RuntimeError(
            "rand_mv CUDA/Triton reference tests require CUDA. "
            "The Metal implementation has its own tests under tests/pie_kernels_metal/."
        )
