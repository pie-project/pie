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


}
