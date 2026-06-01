#pragma once

// Quantizing paged-KV append. Writes freshly-projected bf16 K/V
// `[total_tokens, num_kv_heads, head_dim]` into quantized KV pages, computing
// the per-(token,head) (or per-tensor) scales the matching decode/dequant path
// expects.
//
// De-branded from driver/cuda/src/kernels/kv_paged.cu — specifically the
// quantized write kernels `write_kv_fp8_per_tensor_kernel`,
// `write_kv_per_token_head_kernel<UseFp8>` and their dispatch inside
// `launch_write_kv_to_pages`. The `KvCacheLayerView` argument is replaced with
// explicit raw pointers + scalar params (same convention as
// `attention_naive_paged.cu` / `write_kv_to_pages_bf16`), so this TU does not
// pull in kv_cache.hpp. Layout is the NHD (non-HND) page layout only — the
// quantized schemes are NHD in the old driver too.
//
// Per-token destination is resolved exactly like `write_kv_to_pages_bf16`:
//   pre_kv_len_r   = total_kv_after_r - num_new_tokens_r
//   abs_kv_pos     = pre_kv_len_r + offset_in_new_tokens
//   page_idx_in_r  = abs_kv_pos / page_size
//   offset_in_page = abs_kv_pos % page_size
//   actual_page    = kv_page_indices[kv_page_indptr[r] + page_idx_in_r]
//
// SCALE LAYOUT (per-token-head schemes): one fp32 scale per (page, slot, head),
//   scale_idx = (actual_page * page_size + offset_in_page) * num_kv_heads + h
// i.e. a flat buffer of length [num_pages * page_size * num_kv_heads], head
// being the fastest-varying axis. This matches `load_kv_scalar`'s
//   token_head = (page_id * page_size + slot) * num_kv_heads + kv_head
// in ops/attention_naive_paged.cu, and the dequant-active path below.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// KV storage scheme tag — value-compatible with KvCacheScheme in
// kv_cache_format.hpp / ops/attention_naive_paged.cu.
enum class KvQuantScheme : std::uint8_t {
    Native = 0,
    Fp8PerTensor = 1,
    Int8PerTokenHead = 2,
    Fp8PerTokenHead = 3,
    Fp4Block = 4,
};

// FP8 storage interpretation for the per-tensor scheme. Per-token-head FP8 is
// always E4M3 (mirrors the old driver, which hardcodes __NV_E4M3 there).
enum class KvQuantFp8Kind : std::uint8_t {
    E4M3 = 0,
    E5M2 = 1,
};

// Quantizing KV append. `k_pages`/`v_pages` point at the quantized page pool
// (fp8 bytes for the fp8 schemes, int8 for Int8PerTokenHead). `k_scales` /
// `v_scales` may be nullptr for Fp8PerTensor (no side scales); they are
// fp32[num_pages * page_size * num_kv_heads] for the per-token-head schemes.
//
// For `Native` this is a no-op (the caller should use write_kv_to_pages_bf16).
void write_kv_to_pages_quant(
    void* k_pages,                                 // quantized: [pages, page_size, h_kv, d] storage units
    void* v_pages,
    float* k_scales,                               // [pages * page_size * h_kv] or nullptr (per-tensor)
    float* v_scales,
    const void* k_curr,                            // bf16 [total_tokens, h_kv, d]
    const void* v_curr,
    const std::uint32_t* qo_indptr,                // [R+1]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,           // [R+1]
    const std::uint32_t* kv_last_page_lens,        // [R]
    int total_tokens,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    KvQuantScheme scheme,
    KvQuantFp8Kind fp8_kind,                        // only used by Fp8PerTensor
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
