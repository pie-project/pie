#pragma once

// Dequantize the touched pages of a quantized KV layer into bf16 scratch
// (`k_bf16_pages` / `v_bf16_pages`), so a plain bf16 attention kernel can read
// them. De-branded from driver/cuda/src/kernels/kv_paged.cu —
// `launch_dequant_kv_cache_layer_to_bf16_active` and its per-scheme kernels
// (`dequant_fp8_pages_active_kernel`,
// `dequant_fp8_per_token_head_pages_active_kernel`,
// `dequant_int8_per_token_head_pages_active_kernel`). The `KvCacheLayerView`
// argument is replaced with raw pointers + scalar params (so this TU does not
// pull in kv_cache.hpp); the `CUDA_CHECK(cudaGetLastError())` post-launch check
// is dropped (the ABI layer centralizes it). Fp4Block is not lifted here.
//
// `page_indices` is the *active* page list: `num_pages_in_batch` page ids whose
// contents should be dequantized in place at their physical page slot in the
// bf16 scratch (k_bf16_pages[page * page_elems + ...]). The scratch is sized for
// the whole pool; only the touched pages are written.
//
// SCALE LAYOUT matches kv_append_quant.cuh / ops/attention_naive_paged.cu:
//   per-token-head: fp32[num_pages * page_size * num_kv_heads], with
//     scale_idx = (page * page_size + slot) * num_kv_heads + head.

#include <cstdint>
#include <cuda_runtime.h>

#include "kv_append_quant.cuh"  // KvQuantScheme, KvQuantFp8Kind

namespace pie_cuda_device::kernels {

// Dequantize the active pages of a quantized KV layer into bf16 scratch.
// `k_scales`/`v_scales` are ignored for Fp8PerTensor (pass nullptr). No-op for
// Native (the layer is already bf16).
void dequant_kv_layer_to_bf16_active(
    const void* k_pages,                  // quantized page pool
    const void* v_pages,
    const float* k_scales,                // per-token-head scales or nullptr
    const float* v_scales,
    void* k_bf16_pages,                   // bf16 scratch [pages, page_size, h_kv, d]
    void* v_bf16_pages,
    const std::uint32_t* kv_page_indices, // [num_pages_in_batch] active pages
    int num_pages_in_batch,
    int page_size,
    int num_kv_heads,
    int head_dim,
    KvQuantScheme scheme,
    KvQuantFp8Kind fp8_kind,              // only used by Fp8PerTensor
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
