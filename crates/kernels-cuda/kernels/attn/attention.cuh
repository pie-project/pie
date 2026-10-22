#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>

#include "prim/device.cuh"

#include "attn/fa2/cascade.cuh"
#include "attn/fa2/decode.cuh"
#include "attn/fa2/default_decode_params.cuh"
#include "attn/fa2/default_prefill_params.cuh"
#include "attn/fa2/mask.cuh"
#include "attn/fa2/prefill.cuh"
#include "attn/fa2/variants.cuh"

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
    // A capture lands every key's score, the masked ones too.
    static constexpr bool skips_decode_window = false;

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

inline constexpr auto POS_ENC = ::pie::attn::fa2::PosEncodingMode::kNone;

using VariantWindow = ::pie::attn::fa2::DefaultAttention<false, true, false, false>;

using VariantWindowSoftcap = ::pie::attn::fa2::DefaultAttention<false, true, true, false>;

using VariantFull = ::pie::attn::fa2::DefaultAttention<false, false, false, false>;

using VariantFullSoftcap = ::pie::attn::fa2::DefaultAttention<false, false, true, false>;

using VariantCustom = ::pie::attn::fa2::DefaultAttention<true, true, false, false>;

using VariantCustomSoftcap = ::pie::attn::fa2::DefaultAttention<true, true, true, false>;

using DecodeParams = ::pie::attn::fa2::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using PrefillParams = ::pie::attn::fa2::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// An e4m3 pool's (vLLM's `fp8` KV, a unit scale): its pages read as they
// are, the queries bf16.
using DTypeKV8 = __nv_fp8_e4m3;
using DecodeParams8 = ::pie::attn::fa2::BatchDecodeParams<DTypeQ, DTypeKV8, DTypeO, IdType>;
using PrefillParams8 = ::pie::attn::fa2::BatchPrefillPagedParams<DTypeQ, DTypeKV8, DTypeO, IdType>;

using CaptureWindow = PieScoreCapture<VariantWindow>;
using CaptureFull = PieScoreCapture<VariantFull>;
using CapturePrefill = PieScoreCaptureWindow<VariantFull>;

using DecodeCaptureParams = PieScoreParams<DecodeParams, IdType>;
using PrefillCaptureParams = PieScoreWindowParams<PrefillParams, IdType>;
using DecodeCaptureParams8 = PieScoreParams<DecodeParams8, IdType>;
using PrefillCaptureParams8 = PieScoreWindowParams<PrefillParams8, IdType>;

template <typename Base>
struct RelBiasParams : Base {
    const float* rel_bias = nullptr;
    std::uint32_t rel_extent = 0;
    std::uint32_t rel_heads = 0;
    std::uint32_t log_floor = 0;
    float log_alpha = 0.f;
};

template <bool use_sliding_window>
struct VariantRelBias
    : ::pie::attn::fa2::DefaultAttention<false, use_sliding_window, false, false> {
    using Base = ::pie::attn::fa2::DefaultAttention<false, use_sliding_window, false, false>;

    const float* bias_rows = nullptr;
    std::uint32_t extent = 0;
    std::uint32_t heads = 0;
    float scale = 1.f;
    std::uint32_t log_floor = 0;
    float log_alpha = 0.f;

    template <typename Params>
    __device__ __host__ VariantRelBias(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : Base(params, batch_idx, smem_ptr) {
        this->sm_scale_log2 = ::pie::attn::fa2::math::log2e;
        scale = params.sm_scale;
        extent = params.rel_extent;
        heads = params.rel_heads;
        log_floor = params.log_floor;
        log_alpha = params.log_alpha;
        bias_rows = params.rel_bias +
                    static_cast<std::size_t>(params.q_indptr[batch_idx]) * heads * extent;
    }

    REGISTER_LOGITS_TRANSFORM(
        params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            float out = static_cast<float>(logits) * scale;
            if (qo_idx < this->qo_len && kv_idx < this->kv_len) {
                const int d = static_cast<int>(this->kv_len) - static_cast<int>(this->qo_len) +
                              static_cast<int>(qo_idx) - static_cast<int>(kv_idx);
                if (d >= 0 && d < static_cast<int>(extent)) {
                    out += bias_rows[(static_cast<std::size_t>(qo_idx) * heads + qo_head_idx) *
                                         extent +
                                     static_cast<std::size_t>(d)];
                }
                if (log_alpha != 0.f && log_floor != 0) {

                    const float n = static_cast<float>(this->kv_len - this->qo_len + qo_idx + 1);
                    const float ratio = n / static_cast<float>(log_floor);
                    if (ratio > 1.f) out *= 1.f + log_alpha * __logf(ratio);
                }
            }
            return static_cast<T>(out);
        })
};

using VariantRelBiasFull = VariantRelBias<false>;
using VariantRelBiasWindow = VariantRelBias<true>;
using DecodeRelParams = RelBiasParams<DecodeParams>;
using PrefillRelParams = RelBiasParams<PrefillParams>;
using DecodeRelParams8 = RelBiasParams<DecodeParams8>;
using PrefillRelParams8 = RelBiasParams<PrefillParams8>;

template <::pie::attn::fa2::MaskMode MASK, std::uint32_t CTA_TILE_Q, std::uint32_t NUM_MMA_Q,
          std::uint32_t NUM_MMA_KV, std::uint32_t NUM_MMA_D_QK, std::uint32_t NUM_MMA_D_VO,
          std::uint32_t NUM_WARPS_Q, std::uint32_t NUM_WARPS_KV, class Variant,
          class Params = PrefillParams, class KV = DTypeKV>
using PagedTraits =
    ::pie::attn::fa2::KernelTraits<MASK, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                              NUM_WARPS_Q, NUM_WARPS_KV, POS_ENC, DTypeQ, KV, DTypeO, float,
                              IdType, Variant>;

template <class KTraits>
__device__ unsigned smem_bytes_paged =
    static_cast<unsigned>(sizeof(typename KTraits::SharedStoragePaged));


using RaggedParams = ::pie::attn::fa2::BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

static_assert(sizeof(RaggedParams) == 312,
              "fa2_abi::PrefillRaggedParams mirrors a 312-byte BatchPrefillRaggedParams");
static_assert(alignof(RaggedParams) == 8,
              "fa2_abi::PrefillRaggedParams mirrors an 8-byte-aligned BatchPrefillRaggedParams");

struct RaggedRefParams : RaggedParams {
    const IdType* ref_start = nullptr;
};

static_assert(sizeof(RaggedRefParams) == 320,
              "fa2_abi::PrefillRaggedRefParams mirrors a 320-byte RaggedRefParams");

struct ReferenceSelfOnly : VariantFull {
    std::uint32_t ref_start = 0;

    template <typename Params>
    __device__ __host__ ReferenceSelfOnly(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : VariantFull(params, batch_idx, smem_ptr) {
        const IdType at = params.ref_start[batch_idx];
        ref_start = at < 0 ? 0u : static_cast<std::uint32_t>(at);
    }

    REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
        return qo_idx < ref_start || kv_idx >= ref_start;
    })
};

struct RaggedTagParams : RaggedParams {
    const IdType* q_tags = nullptr;
    const IdType* kv_tags = nullptr;
};

static_assert(sizeof(RaggedTagParams) == 328,
              "fa2_abi::PrefillRaggedTagParams mirrors a 328-byte RaggedTagParams");

struct ReferenceTags : VariantFull {
    std::uint32_t q_base = 0;
    std::uint32_t kv_base = 0;

    template <typename Params>
    __device__ __host__ ReferenceTags(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : VariantFull(params, batch_idx, smem_ptr) {
        q_base = static_cast<std::uint32_t>(params.q_indptr[batch_idx]);
        kv_base = static_cast<std::uint32_t>(params.kv_indptr[batch_idx]);
    }

    // A query tile's padding rows run past the group's `qo_len` (and the
    // last group's past the table): no tag is read for a row or key the
    // group does not hold, so the read stays inside the table.
    REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
        if (qo_idx >= this->qo_len || kv_idx >= this->kv_len) return false;
        const IdType t = params.q_tags[q_base + qo_idx];
        return t < 0 || params.kv_tags[kv_base + kv_idx] == t;
    })
};

struct RaggedClassParams : RaggedParams {
    const IdType* q_classes = nullptr;
    const IdType* kv_classes = nullptr;
    const std::uint8_t* table = nullptr;
    std::uint32_t count = 0;
};

static_assert(sizeof(RaggedClassParams) == 344,
              "fa2_abi::PrefillRaggedClassParams mirrors a 344-byte RaggedClassParams");

struct ClassTable : VariantFull {
    std::uint32_t q_base = 0;
    std::uint32_t kv_base = 0;

    template <typename Params>
    __device__ __host__ ClassTable(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : VariantFull(params, batch_idx, smem_ptr) {
        q_base = static_cast<std::uint32_t>(params.q_indptr[batch_idx]);
        kv_base = static_cast<std::uint32_t>(params.kv_indptr[batch_idx]);
    }

    // As `ReferenceTags`: no class is read for a padding row past the group.
    REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
        if (qo_idx >= this->qo_len || kv_idx >= this->kv_len) return false;
        const IdType qc = params.q_classes[q_base + qo_idx];
        const IdType kc = params.kv_classes[kv_base + kv_idx];
        return qc < 0 || kc < 0 ||
               params.table[static_cast<std::uint32_t>(qc) * params.count +
                            static_cast<std::uint32_t>(kc)] != 0;
    })
};

struct RaggedBiasParams : RaggedParams {
    const float* bias = nullptr;
    std::uint32_t max_len = 0;
};

static_assert(sizeof(RaggedBiasParams) == 328,
              "fa2_abi::PrefillRaggedBiasParams mirrors a 328-byte RaggedBiasParams");

struct RelativeBias : VariantFull {
    const float* bias = nullptr;

    std::uint32_t span = 1;

    std::uint32_t centre = 0;

    template <typename Params>
    __device__ __host__ RelativeBias(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : VariantFull(params, batch_idx, smem_ptr) {
        bias = params.bias;
        centre = params.max_len - 1;
        span = 2 * params.max_len - 1;
        sm_scale_log2 = ::pie::attn::fa2::math::log2e;
    }

    REGISTER_LOGITS_TRANSFORM(
        params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {

            const long long d = static_cast<long long>(kv_idx) -
                                static_cast<long long>(qo_idx) +
                                static_cast<long long>(centre);
            const std::uint32_t at =
                d < 0 ? 0u
                      : (d >= static_cast<long long>(span) ? span - 1
                                                           : static_cast<std::uint32_t>(d));
            const float b = bias[static_cast<std::size_t>(qo_head_idx) * span + at];
            return static_cast<T>(static_cast<float>(logits) * params.sm_scale + b);
        })
};

}

namespace merge_lse {

using DTypeIn = __nv_bfloat16;
using DTypeO = __nv_bfloat16;

using IdType = std::int32_t;

}

// The split kv's partial states folded one warp a (row, head): each lane
// holds HEAD_DIM / 32 columns, reads every partial's base-2 log-sum-exp and
// its normalized output straight from global memory, and lands the row's
// output and log-sum-exp. A row of no partials lands zeros and -inf, one
// partial lands as it stands. `win` is the staged `(count, start)` seat.
// With `sinks`, each head's output is scaled as `sink_rescale` scales it,
// by 1 / (1 + exp(sink - lse)), from the f32 sum before it rounds. Rows lie
// HEAD_DIM apart; LIVE of their columns are folded (the rest no one's).
template <int HEAD_DIM, class S = float, int LIVE = HEAD_DIM>
__global__ void __launch_bounds__(256) merge_partials_warp(
    const merge_lse::DTypeIn* __restrict__ v,
    const float* __restrict__ s,
    const merge_lse::IdType* __restrict__ indptr,
    merge_lse::DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged,
    u32 max_seq_len,
    const u32* __restrict__ seq_len_ptr,
    u32 num_heads,
    const S* __restrict__ sinks,
    const u32* __restrict__ win)
{
    static_assert(LIVE % 64 == 0 && LIVE <= HEAD_DIM, "two columns a lane at least");
    constexpr int PER = LIVE / 32;
    const u32 staged = seq_len_ptr != nullptr ? *seq_len_ptr : max_seq_len;
    const u32 rows = (win != nullptr && win[0] < staged) ? win[0] : staged;
    const u32 row_base = win != nullptr ? win[1] : 0;
    const u32 lane = threadIdx.x & 31;
    const u32 warps = (gridDim.x * blockDim.x) >> 5;
    const u32 col = lane * PER;
    for (u32 pair = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; pair < rows * num_heads;
         pair += warps) {
        const u32 pos = pair / num_heads;
        const u32 head = pair % num_heads;
        const int begin = indptr[pos];
        const int end = indptr[pos + 1];
        float m = -::pie::attn::fa2::math::inf;
        for (int k = begin + static_cast<int>(lane); k < end; k += 32) {
            m = fmaxf(m, s[static_cast<long long>(k) * num_heads + head]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        }
        float acc[PER];
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = 0.f;
        float d = 0.f;
        if (m != -::pie::attn::fa2::math::inf) {
#pragma unroll 4
            for (int k = begin; k < end; ++k) {
                const long long at = static_cast<long long>(k) * num_heads + head;
                const float w = exp2f(s[at] - m);
                d += w;
                const merge_lse::DTypeIn* row = v + at * HEAD_DIM + col;
#pragma unroll
                for (int j = 0; j < PER; j += 2) {
                    const u32 two = *reinterpret_cast<const u32*>(row + j);
                    acc[j] += w * Elem<bf16>::to_f32(bf16(static_cast<unsigned short>(two)));
                    acc[j + 1] += w * Elem<bf16>::to_f32(bf16(static_cast<unsigned short>(two >> 16)));
                }
            }
        }
        float inv = d > 0.f ? 1.f / d : 0.f;
        if (sinks != nullptr && d > 0.f) {
            constexpr float kLn2 = 0.69314718055994530942f;
            const float lse = m + log2f(d);
            inv *= 1.0f / (1.0f + __expf(-(lse * kLn2 - Elem<S>::to_f32(sinks[head]))));
        }
        const long long out = (static_cast<long long>(row_base) + pos) * num_heads + head;
        merge_lse::DTypeO* dst = v_merged + out * HEAD_DIM + col;
#pragma unroll
        for (int j = 0; j < PER; j += 2) {
            const u32 lo = Elem<bf16>::from_f32(acc[j] * inv).raw;
            const u32 hi = Elem<bf16>::from_f32(acc[j + 1] * inv).raw;
            *reinterpret_cast<u32*>(dst + j) = lo | (hi << 16);
        }
        if (s_merged != nullptr && lane == 0) {
            s_merged[out] = d > 0.f ? m + log2f(d) : -::pie::attn::fa2::math::inf;
        }
    }
}

template <class T>
__global__ void logit_softcap(T* __restrict__ x, float cap, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv_cap = 1.f / cap;
    const float v = Elem<T>::to_f32(x[i]);
    x[i] = Elem<T>::from_f32(cap * tanhf(v * inv_cap));
}

// `logit_softcap` eight elements a thread, read and written 16 bytes at a
// time: the same value for every element.
template <class T>
__global__ void logit_softcap_vec8(T* __restrict__ x, float cap, usize runs) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= runs) return;
    const float inv_cap = 1.f / cap;
    uint4 w = reinterpret_cast<const uint4*>(x)[i];
    T* e = reinterpret_cast<T*>(&w);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        e[k] = Elem<T>::from_f32(cap * tanhf(Elem<T>::to_f32(e[k]) * inv_cap));
    }
    reinterpret_cast<uint4*>(x)[i] = w;
}

template <class T, class S = T>
__global__ void sink_rescale(
    T* __restrict__ o,
    const float* __restrict__ lse,
    const S* __restrict__ sinks,
    i32 N,
    i32 num_q_heads,
    i32 head_dim,
    const u32* __restrict__ win)
{
    const i32 t = static_cast<i32>(blockIdx.x);

    if (win != nullptr && t >= static_cast<i32>(win[0])) return;

    const i32 t_row = win != nullptr ? t + static_cast<i32>(win[1]) : t;
    const i32 h = static_cast<i32>(blockIdx.y);
    if (t >= N || h >= num_q_heads) return;

    constexpr float kLn2 = 0.69314718055994530942f;
    const float lse_val = lse[t_row * num_q_heads + h];
    const float sink = Elem<S>::to_f32(sinks[h]);
    float r;
    if (!isfinite(lse_val)) {

        r = 1.0f;
    } else {
        const float diff = lse_val * kLn2 - sink;
        r = 1.0f / (1.0f + __expf(-diff));
    }

    const i32 row_stride = num_q_heads * head_dim;
    T* row = o + static_cast<long long>(t_row) * row_stride + h * head_dim;
    for (i32 d = static_cast<i32>(threadIdx.x); d < head_dim;
         d += static_cast<i32>(blockDim.x)) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * r);
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
    int head_dim,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y;

    const float l1 = lse1[n_row * num_heads + h];
    const float l2 = lse2[n_row * num_heads + h];

    if (!isfinite(l2)) {

        if (o1 != o_out) {
            const long long off = (static_cast<long long>(n_row) * num_heads + h) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                o_out[off + d] = o1[off + d];
            }
        }
        if (lse_out != nullptr && lse_out != lse1) {
            if (threadIdx.x == 0) lse_out[n_row * num_heads + h] = l1;
        }
        return;
    }

    if (!isfinite(l1)) {
        const long long off = (static_cast<long long>(n_row) * num_heads + h) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            o_out[off + d] = o2[off + d];
        }
        if (lse_out != nullptr) {
            if (threadIdx.x == 0) lse_out[n_row * num_heads + h] = l2;
        }
        return;
    }

    const float lse_max = fmaxf(l1, l2);
    const float w1 = exp2f(l1 - lse_max);
    const float w2 = exp2f(l2 - lse_max);
    const float inv_total = 1.0f / (w1 + w2);

    const long long off = (static_cast<long long>(n_row) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v1 = Elem<T>::to_f32(o1[off + d]);
        const float v2 = Elem<T>::to_f32(o2[off + d]);
        o_out[off + d] = Elem<T>::from_f32((v1 * w1 + v2 * w2) * inv_total);
    }

    if (lse_out != nullptr && threadIdx.x == 0) {
        lse_out[n_row * num_heads + h] = lse_max + log2f(w1 + w2);
    }
}
}
