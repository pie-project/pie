//===-- attention_naive_paged.cuh - the paged reference attention ---------===//
//
// Two `__global__`s, the five `__device__` helpers only they call, and the two
// enum mirrors the cache descriptor is read through. No host code:
// `attention_naive_paged.cu` includes this and keeps all FOUR `<<<>>>`, so the
// ahead-of-time build and NVRTC compile ONE text. §21.7 is the record of what
// two texts cost.
//
// # What these two are
//
// The REFERENCE paged attention -- the thing every fused kernel is checked
// against. One block per (request, query offset, head) for the prefill form,
// one per (request, head) for decode; two-pass softmax with the reduction in
// fp32 throughout, because the passes touch O(head_dim * kv_len) bf16 reads
// and a bf16 accumulator cancels catastrophically at long context.
//
// It is deliberately slow and deliberately total: every KV cache scheme the
// tree supports is decoded HERE, in `load_kv_scalar`, one scalar at a time --
// native bf16, fp8 per-tensor, fp8 per-token-head, int8 per-token-head, and
// nvfp4 blockwise. That switch is the definition of what those schemes MEAN,
// which is why the enum mirrors below are checked against the host enums
// rather than commented.
//
// # `BLOCK` is a template parameter, and what it buys
//
// It was `constexpr int BLOCK = 128` in an anonymous namespace. It is now a
// template parameter, for two reasons and one is linkage:
//
//  1. It is already a compile-time quantity in every place it appears -- it
//     fixes the halving reduction `for (off = BLOCK/2; off > 0; off >>= 1)`
//     over `reduce[BLOCK]`, it fixes `dims_per_thread`, and it sizes the
//     per-thread accumulator `acc[]`. Naming it is not decoration; a launch
//     whose block width disagrees with the compiled `BLOCK` reduces through
//     shared memory it never wrote and reads `reduce[0]` as a partial sum.
//  2. A non-template `__global__` in a header carries §21.6's single-includer
//     constraint: host stub and function both take external linkage, so a
//     second translation unit that includes the header is a hard `multiple
//     definition` at link EVEN IF IT NEVER LAUNCHES IT. This is the reference
//     implementation; it is the file most likely to acquire a second includer,
//     and it is the one whose kernels a `LaunchRule` most wants to name --
//     `instantiation()` needs a template to name.
//
// **Exactly one value is instantiated anywhere: 128.** All four launchers in
// `attention_naive_paged.cu` open `dim3 block(BLOCK)` with `constexpr int
// BLOCK = 128` and size shared memory as `(head_dim + BLOCK) * sizeof(float)`.
// A row must cite that, per §17.6 -- take the value from the ahead-of-time
// launcher, do not guess it.
//
// `acc[]` was a literal `float acc[8]` with the comment "upper bound:
// 1024/128". It is now `acc[(kMaxHeadDim + BLOCK - 1) / BLOCK]`, which is 8 at
// BLOCK=128 -- the same array, with the arithmetic that produced the 8 written
// down instead of folded.
//
// # The two enum mirrors are CHECKED, not commented
//
// `KvCacheScheme` (in `attn/kv_cache_view.hpp`) and `DType` (in `tensor.hpp`)
// are host enums: those headers include `<cstdint>` and are compiled by the
// host compiler for the `shim.cpp` ABI, so neither can cross into a file NVRTC
// must read. `quant/mxfp4_marlin.cuh` set the precedent of mirroring a host
// enum -- but it kept the two in step with a COMMENT, which is a grep away
// from being wrong. `attention_naive_paged.cu` instead `static_assert`s every
// enumerator of both mirrors against the host original, so a renumbering that
// would silently decode fp8 pages as int8 is a compile error in the one
// translation unit that sees both spellings. That is the same bridge
// `pack_dense_mask.cu` builds for its mirrored struct.
//
// # Which launcher becomes a row, and which does not
//
// None of the four. All four are `<<<grid, block, smem, stream>>>` with a
// DYNAMIC shared-memory size, `(head_dim + BLOCK) * sizeof(float)`, and no
// ported rule computes smem from a head dimension plus a block width. Three of
// them use `dim3 grid(num_requests, total_tokens, num_q_heads)` -- a THREE-axis
// grid whose middle axis is a conservative upper bound on per-request query
// length, with the kernel early-exiting on the overshoot. `attn.rs` has no
// three-axis rule at all, and inventing one whose second axis is "total tokens
// as a bound on the largest request" under an existing name is exactly the
// failure §17 warns about. All four are stated for whoever writes them, with
// the line they were read from.
//
// # Linkage
//
// **Template-only.** Both `__global__`s are `template <int BLOCK>`, so no host
// stub takes external linkage until something instantiates one and this header
// has NO single-includer constraint. Contrast `pack_dense_mask.cuh` and the
// `write_mla` half of `mla_paged.cuh`, which do.
//
// # NVRTC
//
// `<cmath>`, `<cstdint>`, `<stdexcept>` and `<string>` were the external
// includes and all four are gone -- the fixed-width names come from `device::`,
// and `throw` lives in the `.cu` with `check_head_dim_supported`. What remains
// is `<cuda_fp16.h>`, `<cuda_bf16.h>` and `<cuda_fp8.h>`, three of the seven
// spellings §15 makes resolve to NVIDIA's headers under nvcc and to this
// workspace's shims under NVRTC.
//
// `<cuda_fp16.h>` is included EXPLICITLY and BEFORE `<cuda_fp8.h>`, which the
// `.cu` did not do. §15.4: NVIDIA's `<cuda_fp8.h>` gates its `__half` interop
// on `__CUDA_FP16_TYPES_EXIST__`, so `__nv_cvt_fp8_to_halfraw` -- which this
// file calls -- exists only if some earlier include already defined the half
// types. The `.cu` got them transitively through `<cuda_bf16.h>`; relying on
// that is relying on a vendor-header edge, and four files in this tree do.
// Naming the dependency costs one line.
//
// `tanhf`, `expf`, `logf`, `sqrtf`, `fmaxf` and `fmaf` are builtins NVRTC
// accepts with no header -- measured, as `dsa_indexer.cuh` records for the
// same set. `INFINITY` is NOT: NVRTC does not define the macro, so the two
// `-INFINITY` become `device::neg_inf()`, which is `__int_as_float(0xff800000)`
// and therefore the same bits. `attention_naive.cuh` made the same move.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

// Pulled in by name rather than by `using namespace`, so `device::` means the
// same thing in this header and in `attention_naive_paged.cu`: inside
// `attn::device` the qualifier resolves to THIS namespace, and a prelude name
// not re-exported here stops resolving in the `.cu` the moment it includes
// this file.
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::bf16_to_f32;
using ::pie_cuda_driver::kernels::device::f32_to_bf16;
using ::pie_cuda_driver::kernels::device::i8;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::neg_inf;
using ::pie_cuda_driver::kernels::device::u8;
using ::pie_cuda_driver::kernels::device::u32;

/// Device mirror of `KvCacheScheme` (`attn/kv_cache_view.hpp`).
///
/// The host enum cannot cross: its header pulls `<cstdint>`, which NVRTC
/// answers 0 of 31 times. `attention_naive_paged.cu` sees both spellings and
/// `static_assert`s every enumerator, so this is a checked mirror rather than
/// a commented one.
enum class KvScheme : u8 {
    Native = 0,
    Fp8PerTensor = 1,
    Int8PerTokenHead = 2,
    Fp8PerTokenHead = 3,
    Fp4Block = 4,
};

/// Device mirror of `DType` (`tensor.hpp`). Only `BF16`, `FP8_E4M3` and
/// `FP8_E5M2` are read here, but every enumerator is mirrored and asserted:
/// a partial mirror is a renumbering waiting to happen.
enum class KvDType : u8 {
    BF16 = 0,
    FP16 = 1,
    FP32 = 2,
    INT8 = 3,
    INT32 = 4,
    INT64 = 5,
    UINT8 = 6,
    FP8_E4M3 = 7,
    FP8_E5M2 = 8,
    INT4_PACKED = 9,
};

/// The bound `acc[]` is sized against, and the largest head dim these two
/// kernels accept. `attention_naive_paged.cu` reads it as
/// `device::kMaxHeadDim` in `check_head_dim_supported`, so the array and the
/// predicate that keeps launches inside it are ONE constant, not two.
constexpr int kMaxHeadDim = 1024;


__device__ __forceinline__ float paged_fp4_e2m1_value(device::u8 code) {
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
                                              device::KvDType storage_dtype) {
    const auto fp8_kind = storage_dtype == device::KvDType::FP8_E5M2 ? __NV_E5M2
                                                           : __NV_E4M3;
    return __half2float(__nv_cvt_fp8_to_halfraw(x, fp8_kind));
}

__device__ __forceinline__ float load_kv_scalar(
    const void* pages_raw,
    const float* scales,
    device::KvScheme scheme,
    device::KvDType storage_dtype,
    int block_size,
    int page_size,
    int num_kv_heads,
    int head_dim,
    device::u32 page_id,
    int slot,
    int kv_head,
    int dim)
{
    const long long token_head =
        (static_cast<long long>(page_id) * page_size + slot) *
        num_kv_heads + kv_head;
    switch (scheme) {
        case device::KvScheme::Native: {
            const auto* pages = static_cast<const device::bf16*>(pages_raw);
            return device::bf16_to_f32(
                pages[token_head * static_cast<long long>(head_dim) + dim]);
        }
        case device::KvScheme::Fp8PerTensor: {
            const auto* pages = static_cast<const __nv_fp8_storage_t*>(pages_raw);
            return fp8_to_float(
                pages[token_head * static_cast<long long>(head_dim) + dim],
                storage_dtype);
        }
        case device::KvScheme::Fp8PerTokenHead: {
            const auto* pages = static_cast<const __nv_fp8_storage_t*>(pages_raw);
            const float q = fp8_to_float(
                pages[token_head * static_cast<long long>(head_dim) + dim],
                device::KvDType::FP8_E4M3);
            return q * scales[token_head];
        }
        case device::KvScheme::Int8PerTokenHead: {
            const auto* pages = static_cast<const device::i8*>(pages_raw);
            return static_cast<float>(
                pages[token_head * static_cast<long long>(head_dim) + dim]) *
                scales[token_head];
        }
        case device::KvScheme::Fp4Block: {
            const auto* pages = static_cast<const device::u8*>(pages_raw);
            const int packed_d = (head_dim + 1) / 2;
            const int bs = block_size > 0 ? block_size : 16;
            const int blocks_per_head = (head_dim + bs - 1) / bs;
            const long long packed_idx =
                token_head * static_cast<long long>(packed_d) + dim / 2;
            const int shift = (dim & 1) ? 4 : 0;
            const device::u8 code = (pages[packed_idx] >> shift) & 0xf;
            const long long scale_idx =
                token_head * static_cast<long long>(blocks_per_head) + dim / bs;
            return paged_fp4_e2m1_value(code) * scales[scale_idx];
        }
    }
    return 0.f;
}

__device__ __forceinline__ bool custom_mask_allows(
    const device::u8* mask,
    const device::i32* mask_indptr,
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
template <int BLOCK>
__global__ void naive_paged_attn(
    const device::bf16* __restrict__ q,
    const void*          __restrict__ k_pages,
    const void*          __restrict__ v_pages,
    const float*         __restrict__ k_scales,
    const float*         __restrict__ v_scales,
    device::bf16*       __restrict__ o,
    const device::u32* __restrict__ qo_indptr,
    const device::u32* __restrict__ kv_page_indices,
    const device::u32* __restrict__ kv_page_indptr,
    const device::u32* __restrict__ kv_last_page_lens,
    const device::u8*  __restrict__ custom_mask,
    const device::i32*  __restrict__ custom_mask_indptr,
    int num_q_heads, int num_kv_heads,
    int head_dim, int page_size,
    device::KvScheme scheme,
    device::KvDType storage_dtype,
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

    const device::u32 qo_lo  = qo_indptr[r];
    const device::u32 qo_hi  = qo_indptr[r + 1];
    if (qo_off >= int(qo_hi - qo_lo)) return;
    const int qo_global = static_cast<int>(qo_lo) + qo_off;

    const device::u32 pg_lo = kv_page_indptr[r];
    const device::u32 pg_hi = kv_page_indptr[r + 1];
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
    const device::bf16* q_row =
        q + (static_cast<long long>(qo_global) * num_q_heads + q_head) * head_dim;

    extern __shared__ float smem[];
    float* q_smem = smem;                     // [head_dim] q values, fp32
    float* reduce = smem + head_dim;          // [BLOCK] reduction scratch

    // Stage Q into shared memory in fp32.
    for (int d = tid; d < head_dim; d += BLOCK) {
        q_smem[d] = device::bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    const float scale = (sm_scale > 0.f) ? sm_scale
                        : (1.0f / sqrtf(static_cast<float>(head_dim)));

    // ── Pass 1: row_max ──
    float local_max = device::neg_inf();
    for (int kv = tid; kv < kv_lim; kv += BLOCK) {
        // Sliding window check.
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot     = kv % page_size;
        const device::u32 pg_id = kv_page_indices[pg_lo + page_idx];
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
    float acc[(kMaxHeadDim + BLOCK - 1) / BLOCK];   // 8 at BLOCK=128
    for (int i = 0; i < dims_per_thread; ++i) acc[i] = 0.f;

    float local_z = 0.f;
    for (int kv = 0; kv < kv_lim; ++kv) {
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot     = kv % page_size;
        const device::u32 pg_id = kv_page_indices[pg_lo + page_idx];
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
            z_shared > 0.f ? (logf(z_shared) + row_max) : device::neg_inf();
    }

    device::bf16* o_row =
        o + (static_cast<long long>(qo_global) * num_q_heads + q_head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * BLOCK;
        if (d < head_dim) {
            o_row[d] = device::f32_to_bf16(acc[i] * inv_z);
        }
    }
}

template <int BLOCK>
__global__ void naive_paged_decode(
    const device::bf16* __restrict__ q,
    const void*          __restrict__ k_pages,
    const void*          __restrict__ v_pages,
    const float*         __restrict__ k_scales,
    const float*         __restrict__ v_scales,
    device::bf16*       __restrict__ o,
    const device::u32* __restrict__ kv_page_indices,
    const device::u32* __restrict__ kv_page_indptr,
    const device::u32* __restrict__ kv_last_page_lens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    device::KvScheme scheme,
    device::KvDType storage_dtype,
    int block_size,
    int window_left,
    float sm_scale,
    float logits_soft_cap,
    float* __restrict__ lse_out)
{
    const int r = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    const device::u32 pg_lo = kv_page_indptr[r];
    const device::u32 pg_hi = kv_page_indptr[r + 1];
    const int last_page_len = static_cast<int>(kv_last_page_lens[r]);
    const int num_full_pages = static_cast<int>(pg_hi - pg_lo) - 1;
    const int kv_total = (num_full_pages > 0)
        ? num_full_pages * page_size + last_page_len
        : last_page_len;
    const int kv_lim = kv_total;

    const device::bf16* q_row =
        q + (static_cast<long long>(r) * num_q_heads + q_head) * head_dim;

    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce = smem + head_dim;
    for (int d = tid; d < head_dim; d += BLOCK) {
        q_smem[d] = device::bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    const float scale = (sm_scale > 0.f) ? sm_scale
        : (1.0f / sqrtf(static_cast<float>(head_dim)));

    float local_max = device::neg_inf();
    for (int kv = tid; kv < kv_lim; kv += BLOCK) {
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot = kv % page_size;
        const device::u32 pg_id = kv_page_indices[pg_lo + page_idx];
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * load_kv_scalar(
                k_pages, k_scales, scheme, storage_dtype, block_size,
                page_size, num_kv_heads, head_dim, pg_id, slot, kv_head, d);
        }
        dot = transform_logit(dot, scale, logits_soft_cap);
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

    const int dims_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float acc[(kMaxHeadDim + BLOCK - 1) / BLOCK];
    for (int i = 0; i < dims_per_thread; ++i) acc[i] = 0.f;

    float local_z = 0.f;
    for (int kv = 0; kv < kv_lim; ++kv) {
        if (window_left >= 0 && kv < kv_lim - 1 - window_left) continue;
        const int page_idx = kv / page_size;
        const int slot = kv % page_size;
        const device::u32 pg_id = kv_page_indices[pg_lo + page_idx];
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
        const float dot = transform_logit(reduce[0], scale, logits_soft_cap);
        const float w = expf(dot - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();
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

    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = z_shared > 0.f ? 1.0f / z_shared : 0.f;
    if (tid == 0 && lse_out != nullptr) {
        lse_out[static_cast<long long>(r) * num_q_heads + q_head] =
            z_shared > 0.f ? (logf(z_shared) + row_max) : device::neg_inf();
    }

    device::bf16* o_row =
        o + (static_cast<long long>(r) * num_q_heads + q_head) * head_dim;
    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * BLOCK;
        if (d < head_dim) {
            o_row[d] = device::f32_to_bf16(acc[i] * inv_z);
        }
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device
