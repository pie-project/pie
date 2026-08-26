#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>

#include "prelude/device.cuh"

#include "flashinfer/attention/cascade.cuh"
#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/default_prefill_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/variants.cuh"

namespace pie::attn {

using f32 = float;

namespace fa2 {

template <typename Base, typename IdT>
struct PieScoreParams : Base {
    float* score_out = nullptr;
    const IdT* score_indptr = nullptr;
};

template <typename Base, typename IdT>
struct PieScoreWindowParams : PieScoreParams<Base, IdT> {
    std::uint32_t score_window = 0;
};

template <class Base>
struct PieScoreCapture : Base {

    float* score_row = nullptr;

    template <typename Params>
    __device__ __host__ PieScoreCapture(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : Base(params, batch_idx, smem_ptr) {
        if (params.score_out != nullptr && params.score_indptr != nullptr) {
            score_row = params.score_out +
                        static_cast<std::size_t>(params.score_indptr[batch_idx]);
        }
    }

    REGISTER_LOGITS_TRANSFORM(
        params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            const T out = this->Base::template LogitsTransform<Params, T>(
                params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx,
                kv_head_idx);
            const uint32_t len = this->kv_len;
            if (threadIdx.x == 0 && kv_idx < len && this->score_row != nullptr) {

                this->score_row[qo_head_idx * len + kv_idx] =
                    static_cast<float>(out) * params.sm_scale;
            }
            return out;
        })
};

template <class Base>
struct PieScoreCaptureWindow : Base {
    float* score_row = nullptr;

    std::uint32_t first_qo = 0;
    std::uint32_t window = 0;

    template <typename Params>
    __device__ __host__ PieScoreCaptureWindow(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : Base(params, batch_idx, smem_ptr) {
        if (params.score_out != nullptr && params.score_indptr != nullptr) {
            score_row = params.score_out +
                        static_cast<std::size_t>(params.score_indptr[batch_idx]);
        }
        window = params.score_window;
        first_qo = (this->qo_len > window) ? (this->qo_len - window) : 0u;
    }

    REGISTER_LOGITS_TRANSFORM(
        params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            const T out = this->Base::template LogitsTransform<Params, T>(
                params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx,
                kv_head_idx);
            const uint32_t len = this->kv_len;
            if (kv_idx < len && qo_idx < this->qo_len &&
                qo_idx >= this->first_qo && this->score_row != nullptr) {
                const std::size_t w = qo_idx - this->first_qo;
                const std::size_t row =
                    (static_cast<std::size_t>(qo_head_idx) * this->window + w) *
                    len;

                this->score_row[row + kv_idx] =
                    static_cast<float>(out) * params.sm_scale;
            }
            return out;
        })
};

using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = std::int32_t;

inline constexpr auto POS_ENC = ::flashinfer::PosEncodingMode::kNone;

using VariantWindow = ::flashinfer::DefaultAttention<false, true, false, false>;

using VariantWindowSoftcap = ::flashinfer::DefaultAttention<false, true, true, false>;

using VariantFull = ::flashinfer::DefaultAttention<false, false, false, false>;

using VariantFullSoftcap = ::flashinfer::DefaultAttention<false, false, true, false>;

using VariantCustom = ::flashinfer::DefaultAttention<true, true, false, false>;

using VariantCustomSoftcap = ::flashinfer::DefaultAttention<true, true, true, false>;

using DecodeParams = ::flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using PrefillParams = ::flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

using CaptureWindow = PieScoreCapture<VariantWindow>;
using CaptureFull = PieScoreCapture<VariantFull>;
using CapturePrefill = PieScoreCaptureWindow<VariantFull>;

using DecodeCaptureParams = PieScoreParams<DecodeParams, IdType>;
using PrefillCaptureParams = PieScoreWindowParams<PrefillParams, IdType>;

template <::flashinfer::MaskMode MASK, std::uint32_t CTA_TILE_Q, std::uint32_t NUM_MMA_Q,
          std::uint32_t NUM_MMA_KV, std::uint32_t NUM_MMA_D_QK, std::uint32_t NUM_MMA_D_VO,
          std::uint32_t NUM_WARPS_Q, std::uint32_t NUM_WARPS_KV, class Variant,
          class Params = PrefillParams>
using PagedTraits =
    ::flashinfer::KernelTraits<MASK, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                              NUM_WARPS_Q, NUM_WARPS_KV, POS_ENC, DTypeQ, DTypeKV, DTypeO, float,
                              IdType, Variant>;

template <class KTraits>
__device__ unsigned smem_bytes_paged =
    static_cast<unsigned>(sizeof(typename KTraits::SharedStoragePaged));

}

namespace merge_lse {

using DTypeIn = __nv_bfloat16;
using DTypeO = __nv_bfloat16;

using IdType = std::int32_t;

}

template <class T>
__global__ void logit_softcap(T* __restrict__ x, float cap, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv_cap = 1.f / cap;
    const float v = Elem<T>::to_f32(x[i]);
    x[i] = Elem<T>::from_f32(cap * tanhf(v * inv_cap));
}

template <class T>
__global__ void lse_log2_to_ln(T* __restrict__ lse, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float kLn2 = 0.69314718055994530942f;
    const float v = static_cast<float>(lse[i]);
    if (isfinite(v)) lse[i] = static_cast<T>(v * kLn2);
}

template <class T>
__global__ void sink_rescale(
    T* __restrict__ o,
    const float* __restrict__ lse,
    const T* __restrict__ sinks,
    i32 N,
    i32 num_q_heads,
    i32 head_dim)
{
    const i32 t = static_cast<i32>(blockIdx.x);
    const i32 h = static_cast<i32>(blockIdx.y);
    if (t >= N || h >= num_q_heads) return;

    constexpr float kLn2 = 0.69314718055994530942f;
    const float lse_val = lse[t * num_q_heads + h];
    const float sink = Elem<T>::to_f32(sinks[h]);
    float r;
    if (!isfinite(lse_val)) {

        r = 1.0f;
    } else {
        const float diff = lse_val * kLn2 - sink;
        r = 1.0f / (1.0f + __expf(-diff));
    }

    const i32 row_stride = num_q_heads * head_dim;
    T* row = o + static_cast<long long>(t) * row_stride + h * head_dim;
    for (i32 d = static_cast<i32>(threadIdx.x); d < head_dim;
         d += static_cast<i32>(blockDim.x)) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * r);
    }
}

__global__ void attn_score_fold_heads(
    const float* __restrict__ scores,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size,
    int num_q_heads,
    float* __restrict__ folded)
{
    const int request = static_cast<int>(blockIdx.x);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0 || num_q_heads <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;

    const usize base = static_cast<usize>(score_indptr[request]);
    const float* rows = scores + base;
    float* out = folded + base / static_cast<usize>(num_q_heads);
    const float inv_heads = 1.f / static_cast<float>(num_q_heads);

    for (int i = static_cast<int>(threadIdx.x) +
                 static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x);
         i < kv_len;
         i += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y)) {
        float total = 0.f;
        for (int h = 0; h < num_q_heads; ++h) {
            total += rows[static_cast<usize>(h) *
                              static_cast<usize>(kv_len) +
                          static_cast<usize>(i)];
        }
        out[i] = total * inv_heads;
    }
}

__global__ void attn_score_normalize(
    float* __restrict__ scores,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size)
{
    constexpr int kThreads = 256;
    __shared__ float shared[kThreads];

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;

    float* row = scores + static_cast<usize>(score_indptr[request]) +
                 static_cast<usize>(head) * static_cast<usize>(kv_len);

    float local = neg_inf();
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        local = fmaxf(local, row[i]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] =
                fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    float total = 0.f;
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        const float e = __expf(row[i] - row_max);
        row[i] = e;
        total += e;
    }
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = shared[0];
    if (denom <= 0.f) return;
    const float inv = 1.f / denom;
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        row[i] *= inv;
    }
}

__global__ void attn_prefill_score_normalize(
    float* __restrict__ scores,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size,
    int window)
{
    constexpr int kThreads = 256;
    __shared__ float shared[kThreads];

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int w = static_cast<int>(blockIdx.z);

    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;
    const int qo_len = static_cast<int>(qo_indptr[request + 1]) -
                       static_cast<int>(qo_indptr[request]);
    const int rows = window < qo_len ? window : qo_len;
    if (w >= rows) return;

    const int causal = kv_len - rows + w + 1;
    const int limit = causal < kv_len ? causal : kv_len;
    if (limit <= 0) return;

    float* row = scores + static_cast<usize>(score_indptr[request]) +
                 (static_cast<usize>(head) * static_cast<usize>(window) +
                  static_cast<usize>(w)) *
                     static_cast<usize>(kv_len);

    float local = neg_inf();
    for (int i = threadIdx.x; i < limit; i += kThreads) {
        local = fmaxf(local, row[i]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] =
                fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    float total = 0.f;
    for (int i = threadIdx.x; i < limit; i += kThreads) {
        total += __expf(row[i] - row_max);
    }
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = shared[0];

    const float inv = denom > 0.f ? 1.f / denom : 0.f;

    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        row[i] = i < limit ? __expf(row[i] - row_max) * inv : 0.f;
    }
}

__global__ void attn_prefill_score_fold(
    const float* __restrict__ scores,
    float* __restrict__ folded,
    const i32* __restrict__ score_indptr,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int page_size,
    int num_q_heads,
    int window)
{
    const int request = static_cast<int>(blockIdx.x);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;
    const int qo_len = static_cast<int>(qo_indptr[request + 1]) -
                       static_cast<int>(qo_indptr[request]);
    const int rows = window < qo_len ? window : qo_len;
    if (rows <= 0) return;

    const usize base = static_cast<usize>(score_indptr[request]);
    const usize out_base =
        base / (static_cast<usize>(num_q_heads) * static_cast<usize>(window));
    const float inv = 1.f / static_cast<float>(num_q_heads * rows);

    for (int k = static_cast<int>(threadIdx.x) +
                 static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x);
         k < kv_len;
         k += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y)) {
        float acc = 0.f;
        for (int h = 0; h < num_q_heads; ++h) {
            for (int w = 0; w < rows; ++w) {
                acc += scores[base +
                              (static_cast<usize>(h) *
                                   static_cast<usize>(window) +
                               static_cast<usize>(w)) *
                                  static_cast<usize>(kv_len) +
                              static_cast<usize>(k)];
            }
        }
        folded[out_base + static_cast<usize>(k)] = acc * inv;
    }
}


template <class T>
__global__ void merge_lse_combine(
    const T* __restrict__ o1,
    const float* __restrict__ lse1,
    const T* __restrict__ o2,
    const float* __restrict__ lse2,
    T* __restrict__ o_out,
    float* __restrict__ lse_out,
    int num_heads,
    int head_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;

    const float l1 = lse1[n * num_heads + h];
    const float l2 = lse2[n * num_heads + h];

    if (!isfinite(l2)) {

        if (o1 != o_out) {
            const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                o_out[off + d] = o1[off + d];
            }
        }
        if (lse_out != nullptr && lse_out != lse1) {
            if (threadIdx.x == 0) lse_out[n * num_heads + h] = l1;
        }
        return;
    }

    if (!isfinite(l1)) {
        const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            o_out[off + d] = o2[off + d];
        }
        if (lse_out != nullptr) {
            if (threadIdx.x == 0) lse_out[n * num_heads + h] = l2;
        }
        return;
    }

    const float lse_max = fmaxf(l1, l2);
    const float w1 = exp2f(l1 - lse_max);
    const float w2 = exp2f(l2 - lse_max);
    const float inv_total = 1.0f / (w1 + w2);

    const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v1 = Elem<T>::to_f32(o1[off + d]);
        const float v2 = Elem<T>::to_f32(o2[off + d]);
        o_out[off + d] = Elem<T>::from_f32((v1 * w1 + v2 * w2) * inv_total);
    }

    if (lse_out != nullptr && threadIdx.x == 0) {
        lse_out[n * num_heads + h] = lse_max + log2f(w1 + w2);
    }
}
}
