// Paged reference-attention launchers.
//
// The device text -- `naive_paged_attn<BLOCK>`, `naive_paged_decode<BLOCK>`,
// the five helpers they call and the two enum mirrors -- moved to
// `crates/kernels-cuda-new/csrc/src/attn/attention_naive_paged.cuh`, which
// this file includes. There is ONE text: the ahead-of-time build compiles it
// through this translation unit and NVRTC compiles the same header.
//
// Kept here: the four `<<<grid, block, smem, stream>>>`, the dynamic
// shared-memory size, `KvCacheLayerView` unpacking, and
// `check_head_dim_supported`, which THROWS and so can never cross.
#include "attn/attention_naive_paged.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "attn/attention_naive_paged.cuh"

#include "cuda_check.hpp"

#include "attn/kv_paged.hpp"

namespace pie_cuda_driver::kernels::attn {

namespace {

// The one block width instantiated anywhere. Both kernels are compiled
// against it and every launch below opens `dim3 block(BLOCK)`; the two must
// agree or the halving reduction folds through shared memory the launch never
// wrote.
constexpr int BLOCK = 128;
constexpr int MAX_HEAD_DIM = device::kMaxHeadDim;
static_assert(MAX_HEAD_DIM == BLOCK * 8,
              "acc[] in the kernels is sized (kMaxHeadDim + BLOCK - 1) / BLOCK "
              "and was written as a literal 8 at BLOCK = 128");

// The mirrors in `attention_naive_paged.cuh` exist because the host enums live
// in headers that pull `<cstdint>`, which NVRTC cannot answer. This is the one
// translation unit that sees both spellings, so it is where they are CHECKED.
// `mxfp4_marlin.cuh` keeps its mirror in step with a comment; a comment is one
// careless renumbering away from decoding fp8 pages as int8.
#define PIE_SCHEME_MIRRORS_HOST(name)                                       \
    static_assert(static_cast<std::uint8_t>(KvCacheScheme::name) ==         \
                      static_cast<std::uint8_t>(device::KvScheme::name),    \
                  "device::KvScheme::" #name " drifted from KvCacheScheme")
PIE_SCHEME_MIRRORS_HOST(Native);
PIE_SCHEME_MIRRORS_HOST(Fp8PerTensor);
PIE_SCHEME_MIRRORS_HOST(Int8PerTokenHead);
PIE_SCHEME_MIRRORS_HOST(Fp8PerTokenHead);
PIE_SCHEME_MIRRORS_HOST(Fp4Block);
#undef PIE_SCHEME_MIRRORS_HOST

#define PIE_DTYPE_MIRRORS_HOST(name)                                        \
    static_assert(static_cast<std::uint8_t>(DType::name) ==                 \
                      static_cast<std::uint8_t>(device::KvDType::name),     \
                  "device::KvDType::" #name " drifted from DType")
PIE_DTYPE_MIRRORS_HOST(BF16);
PIE_DTYPE_MIRRORS_HOST(FP16);
PIE_DTYPE_MIRRORS_HOST(FP32);
PIE_DTYPE_MIRRORS_HOST(INT8);
PIE_DTYPE_MIRRORS_HOST(INT32);
PIE_DTYPE_MIRRORS_HOST(INT64);
PIE_DTYPE_MIRRORS_HOST(UINT8);
PIE_DTYPE_MIRRORS_HOST(FP8_E4M3);
PIE_DTYPE_MIRRORS_HOST(FP8_E5M2);
PIE_DTYPE_MIRRORS_HOST(INT4_PACKED);
#undef PIE_DTYPE_MIRRORS_HOST

void check_head_dim_supported(int head_dim, const char* caller) {
    if (head_dim > 0 && head_dim <= MAX_HEAD_DIM) return;
    throw std::runtime_error(
        std::string(caller) + ": head_dim must be in [1, " +
        std::to_string(MAX_HEAD_DIM) + "]; got " + std::to_string(head_dim));
}

}  // namespace

void attention_naive_paged_bf16(
    const void* q,
    const void* k_pages, const void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads,
    int head_dim, int page_size,
    cudaStream_t stream,
    int window_left,
    float sm_scale,
    float logits_soft_cap,
    float* lse_out)
{
    if (num_requests <= 0 || total_tokens <= 0) return;
    check_head_dim_supported(head_dim, "attention_naive_paged_bf16");
    // We launch one block per (request, qo_offset, q_head) — qo_offset
    // is bounded by the largest single-request qo_len. We don't have
    // that bound on hand at the host side, so use `total_tokens` as
    // the conservative upper bound and let the kernel early-exit when
    // `qo_off ≥ qo_hi - qo_lo`. This wastes blocks on small requests
    // but keeps the launch shape uniform.
    dim3 grid(num_requests, total_tokens, num_q_heads);
    dim3 block(BLOCK);
    const std::size_t smem = (head_dim + BLOCK) * sizeof(float);
    device::naive_paged_attn<BLOCK><<<grid, block, smem, stream>>>(
        static_cast<const device::bf16*>(q),
        k_pages,
        v_pages,
        nullptr,
        nullptr,
        static_cast<device::bf16*>(o),
        qo_indptr_d, kv_page_indices_d,
        kv_page_indptr_d, kv_last_page_lens_d,
        nullptr,
        nullptr,
        num_q_heads, num_kv_heads, head_dim, page_size,
        device::KvScheme::Native,
        device::KvDType::BF16,
        0,
        window_left, sm_scale, logits_soft_cap, lse_out);
    CUDA_CHECK(cudaGetLastError());
}

void attention_naive_paged_decode(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int num_requests,
    int num_q_heads,
    cudaStream_t stream,
    int window_left,
    float sm_scale,
    float logits_soft_cap,
    float* lse_out)
{
    if (num_requests <= 0) return;
    check_head_dim_supported(kv_layer.head_dim, "attention_naive_paged_decode");
    dim3 grid(num_requests, num_q_heads);
    dim3 block(BLOCK);
    const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);
    device::naive_paged_decode<BLOCK><<<grid, block, smem, stream>>>(
        static_cast<const device::bf16*>(q),
        kv_layer.k_pages,
        kv_layer.v_pages,
        static_cast<const float*>(kv_layer.k_scales),
        static_cast<const float*>(kv_layer.v_scales),
        static_cast<device::bf16*>(o),
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        num_q_heads,
        kv_layer.num_kv_heads,
        kv_layer.head_dim,
        kv_layer.page_size,
        static_cast<device::KvScheme>(kv_layer.scheme),
        static_cast<device::KvDType>(kv_layer.storage_dtype),
        kv_layer.block_size,
        window_left,
        sm_scale,
        logits_soft_cap,
        lse_out);
    CUDA_CHECK(cudaGetLastError());
}

void attention_naive_paged(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_pages_in_batch,
    int num_q_heads,
    cudaStream_t stream,
    int window_left,
    float sm_scale,
    float logits_soft_cap,
    float* lse_out)
{
    (void)num_pages_in_batch;
    if (num_requests <= 0 || total_tokens <= 0) return;
    check_head_dim_supported(kv_layer.head_dim, "attention_naive_paged");
    dim3 grid(num_requests, total_tokens, num_q_heads);
    dim3 block(BLOCK);
    const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);
    device::naive_paged_attn<BLOCK><<<grid, block, smem, stream>>>(
        static_cast<const device::bf16*>(q),
        kv_layer.k_pages,
        kv_layer.v_pages,
        static_cast<const float*>(kv_layer.k_scales),
        static_cast<const float*>(kv_layer.v_scales),
        static_cast<device::bf16*>(o),
        qo_indptr_d,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        nullptr,
        nullptr,
        num_q_heads,
        kv_layer.num_kv_heads,
        kv_layer.head_dim,
        kv_layer.page_size,
        static_cast<device::KvScheme>(kv_layer.scheme),
        static_cast<device::KvDType>(kv_layer.storage_dtype),
        kv_layer.block_size,
        window_left,
        sm_scale,
        logits_soft_cap,
        lse_out);
    CUDA_CHECK(cudaGetLastError());
}

void attention_naive_paged_custom(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t* mask_d,
    const std::int32_t* mask_indptr_d,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    cudaStream_t stream,
    float sm_scale,
    float logits_soft_cap,
    float* lse_out)
{
    if (num_requests <= 0 || total_tokens <= 0) return;
    check_head_dim_supported(kv_layer.head_dim, "attention_naive_paged_custom");
    dim3 grid(num_requests, total_tokens, num_q_heads);
    dim3 block(BLOCK);
    const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);
    device::naive_paged_attn<BLOCK><<<grid, block, smem, stream>>>(
        static_cast<const device::bf16*>(q),
        kv_layer.k_pages,
        kv_layer.v_pages,
        static_cast<const float*>(kv_layer.k_scales),
        static_cast<const float*>(kv_layer.v_scales),
        static_cast<device::bf16*>(o),
        qo_indptr_d,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        mask_d,
        mask_indptr_d,
        num_q_heads,
        kv_layer.num_kv_heads,
        kv_layer.head_dim,
        kv_layer.page_size,
        static_cast<device::KvScheme>(kv_layer.scheme),
        static_cast<device::KvDType>(kv_layer.storage_dtype),
        kv_layer.block_size,
        /*window_left=*/-1,
        sm_scale,
        logits_soft_cap,
        lse_out);
    CUDA_CHECK(cudaGetLastError());
}

void attention_naive_paged(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_pages_in_batch,
    int num_q_heads,
    cudaStream_t stream,
    int window_left,
    float sm_scale)
{
    dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, num_pages_in_batch, stream);
    attention_naive_paged_bf16(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        total_tokens, num_requests, num_q_heads, kv_layer.num_kv_heads,
        kv_layer.head_dim, kv_layer.page_size, stream, window_left, sm_scale);
}

}  // namespace pie_cuda_driver::kernels::attn
