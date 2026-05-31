#include "attention_naive_paged.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Lifted VERBATIM from driver/cuda/src/ops/attention_naive_paged.cu — the
// raw-pointer bf16 prefill path only. The only changes are the namespace
// (`pie_cuda_driver::ops` → `pie_cuda_device::ops`), renaming the entry
// point (`launch_attention_naive_paged_bf16` → `attention_naive_paged_bf16`)
// to match the new device-library convention, and dropping the legacy
// `CUDA_CHECK(cudaGetLastError())` post-launch check (the ABI layer
// centralizes that, mirroring kernels/rmsnorm.cu). The KvCacheLayerView
// overloads, the decode kernel and the `_custom` variant are not lifted, so
// this TU never includes kv_cache.hpp.
//
// `load_kv_scalar` and `naive_paged_attn_kernel` are templated over the KV
// storage scheme via the `KvCacheScheme`/`DType` enum tags; the raw bf16
// path always passes `KvCacheScheme::Native` + `DType::BF16`. The two enum
// definitions are lifted verbatim from kv_cache_format.hpp / tensor.hpp into
// this namespace so the kernel body stays unchanged without pulling in those
// headers.

namespace pie_cuda_device::ops {

// Lifted verbatim from driver/cuda/src/kv_cache_format.hpp.
enum class KvCacheScheme : std::uint8_t {
    Native,
    Fp8PerTensor,
    Int8PerTokenHead,
    Fp8PerTokenHead,
    Fp4Block,
};

// Lifted verbatim from driver/cuda/src/tensor.hpp (only the members the
// naive-paged kernel references are needed: BF16, FP8_E4M3, FP8_E5M2).
enum class DType : std::uint8_t {
    BF16 = 0,
    FP16 = 1,
    FP32 = 2,
    INT8 = 3,
    INT32 = 4,
    INT64 = 5,
    UINT8 = 6,
    FP8_E4M3 = 7,
    FP8_E5M2 = 8,
};

namespace {

constexpr int BLOCK = 128;
constexpr int MAX_HEAD_DIM = BLOCK * 8;

void check_head_dim_supported(int head_dim, const char* caller) {
    if (head_dim > 0 && head_dim <= MAX_HEAD_DIM) return;
    throw std::runtime_error(
        std::string(caller) + ": head_dim must be in [1, " +
        std::to_string(MAX_HEAD_DIM) + "]; got " + std::to_string(head_dim));
}

__device__ __forceinline__ float fp4_e2m1_value(std::uint8_t code) {
    const bool neg = (code & 0x8) != 0;
    const int mag = code & 0x7;
    float v = 0.f;
    switch (mag) {
        case 0: v = 0.f; break;
        case 1: v = 0.5f; break;
        case 2: v = 1.f; break;
        case 3: v = 1.5f; break;
        case 4: v = 2.f; break;
        case 5: v = 3.f; break;
        case 6: v = 4.f; break;
        default: v = 6.f; break;
    }
    return neg ? -v : v;
}

__device__ __forceinline__ float fp8_to_float(__nv_fp8_storage_t x,
                                              DType storage_dtype) {
    const auto fp8_kind = storage_dtype == DType::FP8_E5M2 ? __NV_E5M2
                                                           : __NV_E4M3;
    return __half2float(__nv_cvt_fp8_to_halfraw(x, fp8_kind));
}

__device__ __forceinline__ float load_kv_scalar(
    const void* pages_raw,
    const float* scales,
    KvCacheScheme scheme,
    DType storage_dtype,
    int block_size,
    int page_size,
    int num_kv_heads,
    int head_dim,
    std::uint32_t page_id,
    int slot,
    int kv_head,
    int dim)
{
    const long long token_head =
        (static_cast<long long>(page_id) * page_size + slot) *
        num_kv_heads + kv_head;
    switch (scheme) {
        case KvCacheScheme::Native: {
            const auto* pages = static_cast<const __nv_bfloat16*>(pages_raw);
            return __bfloat162float(
                pages[token_head * static_cast<long long>(head_dim) + dim]);
        }
        case KvCacheScheme::Fp8PerTensor: {
            const auto* pages = static_cast<const __nv_fp8_storage_t*>(pages_raw);
            return fp8_to_float(
                pages[token_head * static_cast<long long>(head_dim) + dim],
                storage_dtype);
        }
        case KvCacheScheme::Fp8PerTokenHead: {
            const auto* pages = static_cast<const __nv_fp8_storage_t*>(pages_raw);
            const float q = fp8_to_float(
                pages[token_head * static_cast<long long>(head_dim) + dim],
                DType::FP8_E4M3);
            return q * scales[token_head];
        }
        case KvCacheScheme::Int8PerTokenHead: {
            const auto* pages = static_cast<const std::int8_t*>(pages_raw);
            return static_cast<float>(
                pages[token_head * static_cast<long long>(head_dim) + dim]) *
                scales[token_head];
        }
        case KvCacheScheme::Fp4Block: {
            const auto* pages = static_cast<const std::uint8_t*>(pages_raw);
            const int packed_d = (head_dim + 1) / 2;
            const int bs = block_size > 0 ? block_size : 16;
            const int blocks_per_head = (head_dim + bs - 1) / bs;
            const long long packed_idx =
                token_head * static_cast<long long>(packed_d) + dim / 2;
            const int shift = (dim & 1) ? 4 : 0;
            const std::uint8_t code = (pages[packed_idx] >> shift) & 0xf;
            const long long scale_idx =
                token_head * static_cast<long long>(blocks_per_head) + dim / bs;
            return fp4_e2m1_value(code) * scales[scale_idx];
        }
    }
    return 0.f;
}

__device__ __forceinline__ bool custom_mask_allows(
    const std::uint8_t* mask,
    const std::int32_t* mask_indptr,
    int request_idx,
    int qo_off,
    int kv_idx,
    int kv_total)
{
    if (mask == nullptr) return true;
    const long long bit = static_cast<long long>(qo_off) * kv_total + kv_idx;
    const long long byte = static_cast<long long>(mask_indptr[request_idx]) +
                           (bit >> 3);
    return ((mask[byte] >> (bit & 7)) & 1) != 0;
}

__device__ __forceinline__ float transform_logit(float dot,
                                                 float scale,
                                                 float logits_soft_cap)
{
    dot *= scale;
    if (logits_soft_cap > 0.f) {
        dot = logits_soft_cap * tanhf(dot / logits_soft_cap);
    }
    return dot;
}

// One block per (request_idx, qo_offset, q_head). Threads cover the
// `head_dim` axis (`head_dim ≤ 1024` keeps us under 1 thread per dim).
//
// The kernel loops over the request's pages, two-pass softmax: the
// first pass computes `row_max = max_{kv} q·k_kv * sm_scale`, the
// second accumulates `Σ exp(q·k - row_max) * v` and the matching
// denominator. We use fp32 throughout for the partial sums because
// the per-pass passes touch O(head_dim · kv_len) bf16 reads — keeping
// the reduction in fp32 avoids the catastrophic cancellation you'd
// see in bf16 for long contexts.
__global__ void naive_paged_attn_kernel(
    const __nv_bfloat16* __restrict__ q,
    const void*          __restrict__ k_pages,
    const void*          __restrict__ v_pages,
    const float*         __restrict__ k_scales,
    const float*         __restrict__ v_scales,
    __nv_bfloat16*       __restrict__ o,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    const std::uint8_t*  __restrict__ custom_mask,
    const std::int32_t*  __restrict__ custom_mask_indptr,
    int num_q_heads, int num_kv_heads,
    int head_dim, int page_size,
    KvCacheScheme scheme,
    DType storage_dtype,
    int block_size,
    int window_left,
    float sm_scale,
    float logits_soft_cap,
    float* __restrict__ lse_out)
{
    const int r        = blockIdx.x;          // request idx
    const int qo_off   = blockIdx.y;          // offset within this request
    const int q_head   = blockIdx.z;          // q head idx
    const int tid      = threadIdx.x;
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);

    const std::uint32_t qo_lo  = qo_indptr[r];
    const std::uint32_t qo_hi  = qo_indptr[r + 1];
    if (qo_off >= int(qo_hi - qo_lo)) return;
    const int qo_global = static_cast<int>(qo_lo) + qo_off;

    const std::uint32_t pg_lo = kv_page_indptr[r];
    const std::uint32_t pg_hi = kv_page_indptr[r + 1];
    const int last_page_len   = static_cast<int>(kv_last_page_lens[r]);
    const int num_full_pages  = static_cast<int>(pg_hi - pg_lo) - 1;
    const int kv_total        = (num_full_pages > 0)
                                    ? num_full_pages * page_size + last_page_len
                                    : last_page_len;

    // Causal: a query at offset `qo_off` from this request's start
    // sees the first `kv_total - (qo_hi - qo_lo) + qo_off + 1` KV
    // rows. The trailing `(qo_hi - qo_lo)` KV rows correspond to this
    // fire's own queries.
    const int qo_len  = static_cast<int>(qo_hi - qo_lo);
    const bool use_custom_mask = custom_mask != nullptr;
    const int kv_lim = use_custom_mask
        ? kv_total
        : kv_total - qo_len + qo_off + 1;

    // Q row pointer.
    const __nv_bfloat16* q_row =
        q + (static_cast<long long>(qo_global) * num_q_heads + q_head) * head_dim;

    extern __shared__ float smem[];
    float* q_smem = smem;                     // [head_dim] q values, fp32
    float* reduce = smem + head_dim;          // [BLOCK] reduction scratch

    // Stage Q into shared memory in fp32.
    for (int d = tid; d < head_dim; d += BLOCK) {
        q_smem[d] = __bfloat162float(q_row[d]);
    }
    __syncthreads();

    const float scale = (sm_scale > 0.f) ? sm_scale
                        : (1.0f / sqrtf(static_cast<float>(head_dim)));

    // ── Pass 1: row_max ──
    float local_max = -INFINITY;
    for (int kv = tid; kv < kv_lim; kv += BLOCK) {
        // Sliding window check.
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot     = kv % page_size;
        const std::uint32_t pg_id = kv_page_indices[pg_lo + page_idx];
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * load_kv_scalar(
                k_pages, k_scales, scheme, storage_dtype, block_size,
                page_size, num_kv_heads, head_dim, pg_id, slot, kv_head, d);
        }
        dot = transform_logit(dot, scale, logits_soft_cap);
        if (use_custom_mask &&
            !custom_mask_allows(custom_mask, custom_mask_indptr, r, qo_off,
                                kv, kv_total)) {
            continue;
        }
        if (dot > local_max) local_max = dot;
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];
    __syncthreads();

    // ── Pass 2: exp + V weighted sum ──
    // Each thread owns a slice of the head_dim output dims; we
    // accumulate across kv rows in registers.
    const int dims_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float acc[8];                             // upper bound: 1024/128
    for (int i = 0; i < dims_per_thread; ++i) acc[i] = 0.f;

    float local_z = 0.f;
    for (int kv = 0; kv < kv_lim; ++kv) {
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot     = kv % page_size;
        const std::uint32_t pg_id = kv_page_indices[pg_lo + page_idx];
        if (use_custom_mask &&
            !custom_mask_allows(custom_mask, custom_mask_indptr, r, qo_off,
                                kv, kv_total)) {
            continue;
        }

        // Cooperatively compute q·k across threads, reducing through
        // shared memory. This pass dominates runtime — the reduction
        // is BLOCK-wide for every KV row.
        float partial = 0.f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += q_smem[d] * load_kv_scalar(
                k_pages, k_scales, scheme, storage_dtype, block_size,
                page_size, num_kv_heads, head_dim, pg_id, slot, kv_head, d);
        }
        reduce[tid] = partial;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float dot   = transform_logit(reduce[0], scale, logits_soft_cap);
        const float w     = expf(dot - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();
        // Accumulate V*w into per-thread `acc[i]` for this thread's
        // slice of head_dim.
        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * BLOCK;
            if (d < head_dim) {
                const float v = load_kv_scalar(
                    v_pages, v_scales, scheme, storage_dtype, block_size,
                    page_size, num_kv_heads, head_dim, pg_id, slot, kv_head, d);
                acc[i] = fmaf(v, w, acc[i]);
            }
        }
    }

    // Broadcast `local_z` (only thread 0 has it) to every thread.
    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = z_shared > 0.f ? 1.0f / z_shared : 0.f;
    if (tid == 0 && lse_out != nullptr) {
        lse_out[static_cast<long long>(qo_global) * num_q_heads + q_head] =
            z_shared > 0.f ? (logf(z_shared) + row_max) : -INFINITY;
    }

    __nv_bfloat16* o_row =
        o + (static_cast<long long>(qo_global) * num_q_heads + q_head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * BLOCK;
        if (d < head_dim) {
            o_row[d] = __float2bfloat16(acc[i] * inv_z);
        }
    }
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
    naive_paged_attn_kernel<<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        k_pages,
        v_pages,
        nullptr,
        nullptr,
        static_cast<__nv_bfloat16*>(o),
        qo_indptr_d, kv_page_indices_d,
        kv_page_indptr_d, kv_last_page_lens_d,
        nullptr,
        nullptr,
        num_q_heads, num_kv_heads, head_dim, page_size,
        KvCacheScheme::Native,
        DType::BF16,
        0,
        window_left, sm_scale, logits_soft_cap, lse_out);
}

}  // namespace pie_cuda_device::ops
