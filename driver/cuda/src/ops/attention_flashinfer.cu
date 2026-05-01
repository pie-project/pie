#include "ops/attention_flashinfer.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

#include "cuda_check.hpp"

namespace pie_cuda_driver::ops {

namespace {

using DTypeQ  = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO  = __nv_bfloat16;
using IdType  = int32_t;

constexpr uint32_t HEAD_DIM = 128;
constexpr auto POS_ENC = ::flashinfer::PosEncodingMode::kNone;

using AttnVariant = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/false,
    /*use_sliding_window=*/false,
    /*use_logits_soft_cap=*/false,
    /*use_alibi=*/false>;

using DecodeParams = ::flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// flashinfer's `GetPtrFromBaseOffset` is `(base + offset_bytes) reinterpret to T*`.
template <typename T>
inline T* offset_ptr(void* base, std::int64_t off) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(base) + off);
}

// Wraps the templated work-estimation function so DecodePlan can call it.
// GROUP_SIZE = num_q_heads / num_kv_heads (GQA ratio). Phase 1 only
// instantiates GROUP_SIZE=2 (Qwen3-0.6B); M3 will add 1, 4, 8.
template <uint32_t GROUP_SIZE>
struct DecodeWorkEstimator {
    cudaError_t operator()(bool& split_kv, uint32_t& max_grid_size,
                           uint32_t& max_num_pages_per_batch, uint32_t& new_batch_size,
                           uint32_t& gdy, uint32_t batch_size, IdType* kv_indptr_h,
                           uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                           cudaStream_t stream)
    {
        return ::flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
            split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
            batch_size, kv_indptr_h, num_qo_heads, page_size, enable_cuda_graph, stream);
    }
};

}  // namespace

// ── DecodePlanCache (private impl) ─────────────────────────────────────────

struct DecodePlanCache {
    ::flashinfer::DecodePlanInfo plan_info;
    int num_requests = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int page_size = 0;
    bool valid = false;
};

void DecodePlanCacheDeleter::operator()(DecodePlanCache* p) const noexcept {
    delete p;
}

DecodePlanCachePtr make_decode_plan() {
    return DecodePlanCachePtr(new DecodePlanCache{});
}

void plan_attention_flashinfer_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream)
{
    if (head_dim != static_cast<int>(HEAD_DIM)) {
        throw std::runtime_error(
            "flashinfer decode: only HEAD_DIM=128 instantiated; got " + std::to_string(head_dim));
    }
    const int gqa_group_size = num_q_heads / num_kv_heads;

    std::vector<IdType> indptr_h_buf(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        indptr_h_buf[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    auto plan_for = [&](auto work_estimator) {
        return ::flashinfer::DecodePlan<HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
            workspace.float_buffer(), workspace.float_bytes(),
            workspace.int_buffer(), workspace.page_locked_int(),
            workspace.int_bytes(),
            cache.plan_info,
            indptr_h_buf.data(),
            static_cast<uint32_t>(num_requests),
            static_cast<uint32_t>(num_q_heads),
            static_cast<uint32_t>(page_size),
            /*enable_cuda_graph=*/true,
            stream, work_estimator);
    };

    cudaError_t status;
    switch (gqa_group_size) {
        case 1: status = plan_for(DecodeWorkEstimator<1>{}); break;
        case 2: status = plan_for(DecodeWorkEstimator<2>{}); break;
        case 4: status = plan_for(DecodeWorkEstimator<4>{}); break;
        case 8: status = plan_for(DecodeWorkEstimator<8>{}); break;
        default:
            throw std::runtime_error(
                "flashinfer decode: unsupported GQA group size " +
                std::to_string(gqa_group_size) +
                " (instantiated: 1, 2, 4, 8)");
    }
    CUDA_CHECK(status);

    cache.num_requests = num_requests;
    cache.num_q_heads  = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim     = head_dim;
    cache.page_size    = page_size;
    cache.valid        = true;
}

void dispatch_attention_flashinfer_decode_bf16(
    const DecodePlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_flashinfer_decode_bf16: cache is empty; "
            "call plan_attention_flashinfer_decode_bf16 first");
    }

    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(cache.num_kv_heads),
        static_cast<uint32_t>(cache.page_size),
        static_cast<uint32_t>(cache.head_dim),
        static_cast<uint32_t>(cache.num_requests),
        ::flashinfer::QKVLayout::kNHD,
        static_cast<DTypeKV*>(k_pages),
        static_cast<DTypeKV*>(v_pages),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indptr_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_last_page_lens_d)));

    DecodeParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = static_cast<DTypeO*>(o);
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.num_qo_heads = static_cast<uint32_t>(cache.num_q_heads);
    params.q_stride_n = static_cast<IdType>(cache.num_q_heads * cache.head_dim);
    params.q_stride_h = static_cast<IdType>(cache.head_dim);
    params.window_left = -1;
    params.logits_soft_cap = 0.f;
    params.sm_scale = 1.0f / std::sqrt(static_cast<float>(cache.head_dim));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    params.request_indices    = offset_ptr<IdType>(int_buf, cache.plan_info.request_indices_offset);
    params.kv_tile_indices    = offset_ptr<IdType>(int_buf, cache.plan_info.kv_tile_indices_offset);
    params.o_indptr           = offset_ptr<IdType>(int_buf, cache.plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr  = offset_ptr<IdType>(int_buf, cache.plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size  = static_cast<uint32_t>(cache.plan_info.padded_batch_size);
    params.partition_kv       = cache.plan_info.split_kv;

    DTypeO* tmp_v = nullptr;
    float*  tmp_s = nullptr;
    if (cache.plan_info.split_kv) {
        tmp_v = offset_ptr<DTypeO>(float_buf, cache.plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, cache.plan_info.s_offset);
        if (cache.plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                offset_ptr<bool>(int_buf, cache.plan_info.block_valid_mask_offset);
        }
    }

    cudaError_t status = ::flashinfer::BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
    CUDA_CHECK(status);
}

// ── Prefill ────────────────────────────────────────────────────────────────

namespace {

using PrefillParams = ::flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

}  // namespace

void launch_attention_flashinfer_prefill_bf16(
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream)
{
    if (head_dim != static_cast<int>(HEAD_DIM)) {
        throw std::runtime_error(
            "flashinfer prefill: only HEAD_DIM=128 instantiated");
    }

    // 1. paged_kv_t — same construction as decode.
    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(num_requests),
        ::flashinfer::QKVLayout::kNHD,
        static_cast<DTypeKV*>(k_pages),
        static_cast<DTypeKV*>(v_pages),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indptr_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_last_page_lens_d)));

    // 2. Plan.
    ::flashinfer::PrefillPlanInfo plan_info;
    std::vector<IdType> qo_h(num_requests + 1);
    std::vector<IdType> kv_h(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        qo_h[r] = static_cast<IdType>(qo_indptr_h[r]);
        kv_h[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    auto status = ::flashinfer::PrefillPlan<IdType>(
        workspace.float_buffer(), workspace.float_bytes(),
        workspace.int_buffer(), workspace.page_locked_int(),
        workspace.int_bytes(),
        plan_info,
        qo_h.data(), kv_h.data(),
        /*total_num_rows=*/static_cast<uint32_t>(total_tokens),
        /*batch_size=*/static_cast<uint32_t>(num_requests),
        /*num_qo_heads=*/static_cast<uint32_t>(num_q_heads),
        /*num_kv_heads=*/static_cast<uint32_t>(num_kv_heads),
        /*head_dim_qk=*/HEAD_DIM,
        /*head_dim_vo=*/HEAD_DIM,
        /*page_size=*/static_cast<uint32_t>(page_size),
        /*enable_cuda_graph=*/false,
        /*sizeof_dtype_o=*/sizeof(DTypeO),
        /*window_left=*/-1,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/false,
        /*num_colocated_ctas=*/0,
        stream);
    CUDA_CHECK(status);

    // 3. Build params.
    PrefillParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = const_cast<IdType*>(reinterpret_cast<const IdType*>(qo_indptr_d));
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<DTypeO*>(o);
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_q_heads / num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(num_q_heads);
    params.q_stride_n = static_cast<IdType>(num_q_heads * head_dim);
    params.q_stride_h = static_cast<IdType>(head_dim);
    params.window_left = -1;
    params.logits_soft_cap = 0.f;
    params.sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf   = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    params.request_indices   = offset_ptr<IdType>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr          = offset_ptr<IdType>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = offset_ptr<IdType>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv      = plan_info.split_kv;
    params.max_total_num_rows = static_cast<uint32_t>(plan_info.total_num_rows);
    params.merge_indptr      = nullptr;
    params.block_valid_mask  = nullptr;
    params.total_num_rows    = nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    DTypeO* tmp_v = nullptr;
    float*  tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr = offset_ptr<IdType>(int_buf, plan_info.merge_indptr_offset);
        tmp_v = offset_ptr<DTypeO>(float_buf, plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, plan_info.s_offset);
    }

    // 4. Dispatch on cta_tile_q (16 / 64 / 128).
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
        status = ::flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC,
            /*USE_FP16_QK_REDUCTION=*/false,
            ::flashinfer::MaskMode::kCausal,
            AttnVariant, PrefillParams>(
            params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
    });
    CUDA_CHECK(status);
}

// ── Prefill with custom mask ───────────────────────────────────────────────

namespace {

using AttnVariantCustom = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/true,
    /*use_sliding_window=*/false,
    /*use_logits_soft_cap=*/false,
    /*use_alibi=*/false>;

}  // namespace

void launch_attention_flashinfer_prefill_custom_bf16(
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t*  mask_d,
    const std::int32_t*  mask_indptr_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream)
{
    if (head_dim != static_cast<int>(HEAD_DIM)) {
        throw std::runtime_error(
            "flashinfer prefill (custom mask): only HEAD_DIM=128 instantiated");
    }

    // 1. paged_kv_t (same as kCausal path).
    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(num_requests),
        ::flashinfer::QKVLayout::kNHD,
        static_cast<DTypeKV*>(k_pages),
        static_cast<DTypeKV*>(v_pages),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indptr_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_last_page_lens_d)));

    // 2. Plan (same as kCausal — the planner doesn't care about mask mode).
    ::flashinfer::PrefillPlanInfo plan_info;
    std::vector<IdType> qo_h(num_requests + 1);
    std::vector<IdType> kv_h(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        qo_h[r] = static_cast<IdType>(qo_indptr_h[r]);
        kv_h[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    auto status = ::flashinfer::PrefillPlan<IdType>(
        workspace.float_buffer(), workspace.float_bytes(),
        workspace.int_buffer(), workspace.page_locked_int(),
        workspace.int_bytes(),
        plan_info,
        qo_h.data(), kv_h.data(),
        static_cast<uint32_t>(total_tokens),
        static_cast<uint32_t>(num_requests),
        static_cast<uint32_t>(num_q_heads),
        static_cast<uint32_t>(num_kv_heads),
        HEAD_DIM, HEAD_DIM,
        static_cast<uint32_t>(page_size),
        /*enable_cuda_graph=*/false,
        sizeof(DTypeO),
        /*window_left=*/-1,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/false,
        /*num_colocated_ctas=*/0,
        stream);
    CUDA_CHECK(status);

    // 3. Build params, including custom mask pointers.
    PrefillParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = const_cast<std::uint8_t*>(mask_d);
    params.q_indptr = const_cast<IdType*>(reinterpret_cast<const IdType*>(qo_indptr_d));
    params.maybe_mask_indptr = const_cast<IdType*>(mask_indptr_d);
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<DTypeO*>(o);
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_q_heads / num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(num_q_heads);
    params.q_stride_n = static_cast<IdType>(num_q_heads * head_dim);
    params.q_stride_h = static_cast<IdType>(head_dim);
    params.window_left = -1;
    params.logits_soft_cap = 0.f;
    params.sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf   = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    params.request_indices   = offset_ptr<IdType>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr          = offset_ptr<IdType>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = offset_ptr<IdType>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv      = plan_info.split_kv;
    params.max_total_num_rows = static_cast<uint32_t>(plan_info.total_num_rows);
    params.merge_indptr      = nullptr;
    params.block_valid_mask  = nullptr;
    params.total_num_rows    = nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    DTypeO* tmp_v = nullptr;
    float*  tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr = offset_ptr<IdType>(int_buf, plan_info.merge_indptr_offset);
        tmp_v = offset_ptr<DTypeO>(float_buf, plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, plan_info.s_offset);
    }

    // 4. Dispatch on cta_tile_q with kCustom + AttnVariantCustom (use_custom_mask=true).
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
        status = ::flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC,
            /*USE_FP16_QK_REDUCTION=*/false,
            ::flashinfer::MaskMode::kCustom,
            AttnVariantCustom, PrefillParams>(
            params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
    });
    CUDA_CHECK(status);
}

}  // namespace pie_cuda_driver::ops
