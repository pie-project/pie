"""Fused MoE weight-permutation helpers — CUDA-only re-exports.

No Metal equivalent exists. Callers must gate invocations by CUDA availability;
importing this module on a non-CUDA system will fail at ``import flashinfer``
time, which is the intended behavior.

Flattens ``flashinfer.fused_moe.core`` — pie_kernels owns the surface shape.
"""

from flashinfer.fused_moe.core import (  # type: ignore[import-not-found]  # noqa: F401
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
