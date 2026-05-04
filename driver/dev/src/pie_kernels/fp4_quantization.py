"""FP4 block-scale quantization helpers — CUDA-only re-exports.

No Metal equivalent exists. Callers must gate invocations by CUDA availability
(e.g. ``is_apple_silicon()``-style early returns); importing this module on a
non-CUDA system will fail at ``import flashinfer`` time, which is the
intended behavior.
"""

from flashinfer.fp4_quantization import (  # type: ignore[import-not-found]  # noqa: F401
    block_scale_interleave,
)
