"""pie_kernels.metal: Metal-accelerated kernels for macOS (Apple Silicon).

Mirrors the subset of the upstream `flashinfer` API that pie_driver_dev uses, so
`pie_kernels` can dispatch between this module and upstream `flashinfer` based
on device availability.
"""

__version__ = "0.3.0"
__author__ = "Pie Team"

from ._wrappers import (
    BatchAttentionWithAttentionSinkWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    append_paged_kv_cache,
    apply_llama31_rope_pos_ids_inplace,
    apply_rope_pos_ids_inplace,
    apply_rope_with_cos_sin_cache_inplace,
    get_batch_indices_positions,
    get_seq_lens,
    mm_fp8,
    trtllm_fp4_block_scale_moe,
)
from .rand_mv import batched_randn_matmul, batched_randn_generate

__all__ = [
    "BatchAttentionWithAttentionSinkWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchDecodeWithPagedKVCacheWrapper",
    "apply_rope_pos_ids_inplace",
    "apply_llama31_rope_pos_ids_inplace",
    "apply_rope_with_cos_sin_cache_inplace",
    "append_paged_kv_cache",
    "get_seq_lens",
    "get_batch_indices_positions",
    "mm_fp8",
    "trtllm_fp4_block_scale_moe",
    "batched_randn_matmul",
    "batched_randn_generate",
]
