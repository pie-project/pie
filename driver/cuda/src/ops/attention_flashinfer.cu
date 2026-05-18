#include "ops/attention_flashinfer.hpp"

#include <type_traits>

#include <cmath>
#include <stdexcept>
#include <string>
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

constexpr auto POS_ENC = ::flashinfer::PosEncodingMode::kNone;

// Causal-mask variant with sliding-window enabled. With
// `params.window_left = -1` flashinfer's runtime constructor sets the
// effective window to `kv_len`, making the per-element mask predicate
// trivially-true — i.e. behaves like full causal attention. Forward
// graphs that need true sliding-window attention (Mistral, Gemma-3
// sliding layers, OLMo-3 sliding layers) just set `window_left ≥ 0`.
using AttnVariant = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/false,
    /*use_sliding_window=*/true,
    /*use_logits_soft_cap=*/false,
    /*use_alibi=*/false>;

// Same as `AttnVariant` plus per-element `logits_soft_cap` (Gemma-2
// `attn_logit_softcapping = 50`). flashinfer applies
// `logits = cap * tanh(logits / cap)` inside the softmax; we only
// route here when `params.logits_soft_cap > 0`.
using AttnVariantSoftcap = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/false,
    /*use_sliding_window=*/true,
    /*use_logits_soft_cap=*/true,
    /*use_alibi=*/false>;

using AttnVariantFull = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/false,
    /*use_sliding_window=*/false,
    /*use_logits_soft_cap=*/false,
    /*use_alibi=*/false>;

using AttnVariantFullSoftcap = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/false,
    /*use_sliding_window=*/false,
    /*use_logits_soft_cap=*/true,
    /*use_alibi=*/false>;

using DecodeParams = ::flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// flashinfer's `GetPtrFromBaseOffset` is `(base + offset_bytes) reinterpret to T*`.
template <typename T>
inline T* offset_ptr(void* base, std::int64_t off) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(base) + off);
}

// Wraps the templated work-estimation function so DecodePlan can call it.
// GROUP_SIZE = num_q_heads / num_kv_heads (GQA ratio); HEAD_DIM is the
// per-head feature width. Adding a (head_dim, group_size) pair adds one
// CUDA template instantiation, which dominates compile time — so we
// only enumerate the pairs production checkpoints actually use.
// Cascade-merge (`VariableLengthMergeStates`) only instantiates head_dim
// ∈ {64, 128, 256, 512}. For other head_dims (e.g. Phi-3-mini at 96), we
// force `split_kv = false` at planning time so the partition-kv path —
// and its post-merge — is never taken.
constexpr bool head_dim_supports_cascade_merge(uint32_t hd) {
    return hd == 64 || hd == 128 || hd == 256 || hd == 512;
}

int current_device_major() {
    thread_local int cached_device = -1;
    thread_local int cached_major = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        cached_device = dev;
        cached_major = prop.major;
    }
    return cached_major;
}

template <uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
struct DecodeWorkEstimator {
    cudaError_t operator()(bool& split_kv, uint32_t& max_grid_size,
                           uint32_t& max_num_pages_per_batch, uint32_t& new_batch_size,
                           uint32_t& gdy, uint32_t batch_size, IdType* kv_indptr_h,
                           uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                           cudaStream_t stream)
    {
        const auto rc = ::flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
            split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
            batch_size, kv_indptr_h, num_qo_heads, page_size, enable_cuda_graph, stream);
        if constexpr (!head_dim_supports_cascade_merge(HEAD_DIM)) {
            split_kv = false;
            new_batch_size = batch_size;
        }
        if (current_device_major() >= 9 && batch_size <= 512) {
            split_kv = false;
            new_batch_size = batch_size;
        }
        return rc;
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
    bool enable_pdl = false;
    bool valid = false;
};

struct PrefillPlanCache {
    ::flashinfer::PrefillPlanInfo plan_info;
    int total_tokens = 0;
    int num_requests = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int page_size = 0;
    int window_left = -1;
    bool full_attention_variant = false;
    bool enable_pdl = false;
    bool valid = false;
};

void DecodePlanCacheDeleter::operator()(DecodePlanCache* p) const noexcept {
    delete p;
}

void PrefillPlanCacheDeleter::operator()(PrefillPlanCache* p) const noexcept {
    delete p;
}

DecodePlanCachePtr make_decode_plan() {
    return DecodePlanCachePtr(new DecodePlanCache{});
}

PrefillPlanCachePtr make_prefill_plan() {
    return PrefillPlanCachePtr(new PrefillPlanCache{});
}

std::uint8_t decode_plan_graph_layout(const DecodePlanCache& cache) {
    if (!cache.valid) return 0;
    return cache.plan_info.split_kv ? 1u : 0u;
}

std::uint8_t prefill_plan_graph_layout(const PrefillPlanCache& cache) {
    if (!cache.valid) return 0;
    std::uint8_t tile_class = 0;
    switch (cache.plan_info.cta_tile_q) {
        case 16:  tile_class = 1; break;
        case 32:  tile_class = 2; break;
        case 64:  tile_class = 3; break;
        case 128: tile_class = 4; break;
        default:  tile_class = 0; break;
    }
    const std::uint8_t variant_class =
        cache.full_attention_variant ? 1u : 0u;
    return static_cast<std::uint8_t>(
        (cache.plan_info.split_kv ? 1u : 0u) |
        (tile_class << 1) |
        (variant_class << 5));
}

namespace {

bool current_device_supports_pdl() {
    return current_device_major() >= 9;
}

template <uint32_t HEAD_DIM>
cudaError_t plan_decode_for_head_dim(
    DecodePlanCache& cache,
    const std::vector<IdType>& indptr_h_buf,
    uint32_t num_requests, uint32_t num_q_heads, uint32_t page_size,
    int gqa_group_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph)
{
    auto plan_for = [&](auto work_estimator) {
        return ::flashinfer::DecodePlan<HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
            workspace.float_buffer(), workspace.float_bytes(),
            workspace.int_buffer(), workspace.page_locked_int(),
            workspace.int_bytes(),
            cache.plan_info,
            const_cast<IdType*>(indptr_h_buf.data()),
            num_requests, num_q_heads, page_size,
            enable_cuda_graph,
            stream, work_estimator);
    };
    switch (gqa_group_size) {
        case 1: return plan_for(DecodeWorkEstimator<HEAD_DIM, 1>{});
        case 2: return plan_for(DecodeWorkEstimator<HEAD_DIM, 2>{});
        case 3: return plan_for(DecodeWorkEstimator<HEAD_DIM, 3>{});
        case 4: return plan_for(DecodeWorkEstimator<HEAD_DIM, 4>{});
        case 8: return plan_for(DecodeWorkEstimator<HEAD_DIM, 8>{});
    }
    throw std::runtime_error(
        "flashinfer decode: unsupported GQA group size " +
        std::to_string(gqa_group_size) +
        " (instantiated: 1, 2, 3, 4, 8)");
}

}  // namespace

void plan_attention_flashinfer_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph)
{
    const int gqa_group_size = num_q_heads / num_kv_heads;

    std::vector<IdType> indptr_h_buf(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        indptr_h_buf[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    cudaError_t status;
    switch (head_dim) {
        case 64:
            status = plan_decode_for_head_dim<64>(
                cache, indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph);
            break;
        case 96:
            status = plan_decode_for_head_dim<96>(
                cache, indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph);
            break;
        case 128:
            status = plan_decode_for_head_dim<128>(
                cache, indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph);
            break;
        case 256:
            status = plan_decode_for_head_dim<256>(
                cache, indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph);
            break;
        case 512:
            status = plan_decode_for_head_dim<512>(
                cache, indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph);
            break;
        default:
            throw std::runtime_error(
                "flashinfer decode: unsupported head_dim " +
                std::to_string(head_dim) +
                " (instantiated: 64, 128, 256, 512)");
    }
    CUDA_CHECK(status);

    cache.num_requests = num_requests;
    cache.num_q_heads  = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim     = head_dim;
    cache.page_size    = page_size;
    cache.enable_pdl   = current_device_supports_pdl();
    cache.valid        = true;
}

void plan_attention_flashinfer_prefill_bf16(
    PrefillPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    int window_left,
    bool full_attention_variant)
{
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        throw std::runtime_error(
            "flashinfer prefill plan: instantiated for HEAD_DIM in {64, 128, 256, 512}; got " +
            std::to_string(head_dim));
    }

    std::vector<IdType> qo_h(num_requests + 1);
    std::vector<IdType> kv_h(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        qo_h[r] = static_cast<IdType>(qo_indptr_h[r]);
        kv_h[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    const bool head_dim_supports_split =
        (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512);

    auto status = ::flashinfer::PrefillPlan<IdType>(
        workspace.float_buffer(), workspace.float_bytes(),
        workspace.int_buffer(), workspace.page_locked_int(),
        workspace.int_bytes(),
        cache.plan_info,
        qo_h.data(), kv_h.data(),
        static_cast<uint32_t>(total_tokens),
        static_cast<uint32_t>(num_requests),
        static_cast<uint32_t>(num_q_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(page_size),
        enable_cuda_graph,
        sizeof(DTypeO),
        window_left,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/!head_dim_supports_split,
        /*num_colocated_ctas=*/0,
        stream);
    CUDA_CHECK(status);

    cache.total_tokens = total_tokens;
    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.window_left = window_left;
    cache.full_attention_variant = full_attention_variant;
    cache.enable_pdl = current_device_supports_pdl();
    cache.valid = true;
}

namespace {

template <uint32_t HEAD_DIM, class Variant>
cudaError_t dispatch_decode_for_head_dim_v(
    const DecodePlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
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
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.num_qo_heads = static_cast<uint32_t>(cache.num_q_heads);
    params.q_stride_n = static_cast<IdType>(cache.num_q_heads * cache.head_dim);
    params.q_stride_h = static_cast<IdType>(cache.head_dim);
    params.window_left = window_left;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : (1.0f / std::sqrt(static_cast<float>(cache.head_dim)));
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

    return ::flashinfer::BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM, POS_ENC, Variant, DecodeParams>(
        params, tmp_v, tmp_s, cache.enable_pdl, stream);
}

// Soft-cap-aware HEAD_DIM dispatch. Routes to either the plain or the
// soft-cap variant based on `logits_soft_cap > 0`. The plan that drove
// us here is computed against `AttnVariant` (no softcap); the soft-cap
// kernel's shared-memory footprint is identical, so the plan stays
// correct across both code paths.
template <uint32_t HEAD_DIM>
cudaError_t dispatch_decode_for_head_dim(
    const DecodePlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (logits_soft_cap > 0.f) {
        return dispatch_decode_for_head_dim_v<HEAD_DIM, AttnVariantSoftcap>(
            cache, q, k_pages, v_pages, o,
            kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
    }
    return dispatch_decode_for_head_dim_v<HEAD_DIM, AttnVariant>(
        cache, q, k_pages, v_pages, o,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        workspace, stream, window_left, /*soft_cap=*/0.f, sm_scale, lse_out);
}

}  // namespace

void dispatch_attention_flashinfer_decode_bf16(
    const DecodePlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages, void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_flashinfer_decode_bf16: cache is empty; "
            "call plan_attention_flashinfer_decode_bf16 first");
    }
    cudaError_t status;
    switch (cache.head_dim) {
        case 64:
            status = dispatch_decode_for_head_dim<64>(cache, q, k_pages, v_pages, o,
                kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
            break;
        case 128:
            status = dispatch_decode_for_head_dim<128>(cache, q, k_pages, v_pages, o,
                kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
            break;
        case 256:
            status = dispatch_decode_for_head_dim<256>(cache, q, k_pages, v_pages, o,
                kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
            break;
        case 512:
            status = dispatch_decode_for_head_dim<512>(cache, q, k_pages, v_pages, o,
                kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
            break;
        default:
            throw std::runtime_error(
                "flashinfer decode dispatch: unsupported head_dim " +
                std::to_string(cache.head_dim));
    }
    CUDA_CHECK(status);
}

// ── Prefill ────────────────────────────────────────────────────────────────

namespace {

using PrefillParams = ::flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// Prefill is templated on (HEAD_DIM × CTA_TILE_Q × MASK_MODE × VARIANT).
// Production checkpoints we support: 64 (Llama-3.2-1B), 128 (Llama-3-8B
// / Qwen / Mistral / Phi). Gemma's 256 path lives in its own forward.
template <uint32_t HEAD_DIM, ::flashinfer::MaskMode MASK, class Variant>
cudaError_t prefill_dispatch_for_head_dim(
    PrefillParams& params, const ::flashinfer::PrefillPlanInfo& plan_info,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream)
{
    cudaError_t status;
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
        status = ::flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC,
            /*USE_FP16_QK_REDUCTION=*/true,
            MASK,
            Variant, PrefillParams>(
            params, tmp_v, tmp_s, enable_pdl, stream);
    });
    return status;
}

}  // namespace

void dispatch_attention_flashinfer_prefill_bf16(
    const PrefillPlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_flashinfer_prefill_bf16: cache is empty; "
            "call plan_attention_flashinfer_prefill_bf16 first");
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

    PrefillParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = const_cast<IdType*>(reinterpret_cast<const IdType*>(qo_indptr_d));
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<DTypeO*>(o);
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(cache.num_q_heads / cache.num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(cache.num_q_heads);
    params.q_stride_n = static_cast<IdType>(cache.num_q_heads * cache.head_dim);
    params.q_stride_h = static_cast<IdType>(cache.head_dim);
    params.window_left = cache.window_left;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : (1.0f / std::sqrt(static_cast<float>(cache.head_dim)));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    const auto& plan_info = cache.plan_info;
    params.request_indices = offset_ptr<IdType>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices = offset_ptr<IdType>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = offset_ptr<IdType>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr = offset_ptr<IdType>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = offset_ptr<IdType>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv = plan_info.split_kv;
    params.max_total_num_rows = static_cast<uint32_t>(plan_info.total_num_rows);
    params.merge_indptr = nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    DTypeO* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr = offset_ptr<IdType>(int_buf, plan_info.merge_indptr_offset);
        tmp_v = offset_ptr<DTypeO>(float_buf, plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                offset_ptr<bool>(int_buf, plan_info.block_valid_mask_offset);
        }
    }

    auto dispatch = [&]<class Variant>(::std::type_identity<Variant>) {
        switch (cache.head_dim) {
            case 64:
                return prefill_dispatch_for_head_dim<64, ::flashinfer::MaskMode::kNone, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 128:
                return prefill_dispatch_for_head_dim<128, ::flashinfer::MaskMode::kNone, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 256:
                return prefill_dispatch_for_head_dim<256, ::flashinfer::MaskMode::kNone, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 512:
                return prefill_dispatch_for_head_dim<512, ::flashinfer::MaskMode::kNone, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
        }
        return cudaErrorInvalidValue;
    };

    cudaError_t status;
    if (cache.full_attention_variant && logits_soft_cap > 0.f) {
        status = dispatch(::std::type_identity<AttnVariantFullSoftcap>{});
    } else if (cache.full_attention_variant) {
        status = dispatch(::std::type_identity<AttnVariantFull>{});
    } else if (logits_soft_cap > 0.f) {
        status = dispatch(::std::type_identity<AttnVariantSoftcap>{});
    } else {
        status = dispatch(::std::type_identity<AttnVariant>{});
    }
    CUDA_CHECK(status);
}

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
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        throw std::runtime_error(
            "flashinfer prefill: instantiated for HEAD_DIM ∈ {64, 128, 256, 512}; got " +
            std::to_string(head_dim));
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

    // The cascade-merge kernel `VariableLengthMergeStates` only instantiates
    // head_dim ∈ {64, 128, 256, 512}. For other head dims (e.g. Phi-3-mini at
    // 96), force the prefill to a single-CTA-per-request schedule by disabling
    // split-KV — that path skips the cascade merge entirely.
    const bool head_dim_supports_split =
        (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512);

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
        /*head_dim_qk=*/static_cast<uint32_t>(head_dim),
        /*head_dim_vo=*/static_cast<uint32_t>(head_dim),
        /*page_size=*/static_cast<uint32_t>(page_size),
        /*enable_cuda_graph=*/false,
        /*sizeof_dtype_o=*/sizeof(DTypeO),
        /*window_left=*/window_left,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/!head_dim_supports_split,
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
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_q_heads / num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(num_q_heads);
    params.q_stride_n = static_cast<IdType>(num_q_heads * head_dim);
    params.q_stride_h = static_cast<IdType>(head_dim);
    params.window_left = window_left;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : (1.0f / std::sqrt(static_cast<float>(head_dim)));
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
    const bool enable_pdl = current_device_supports_pdl();

    // 4. Dispatch on (HEAD_DIM, soft-cap variant). The lambda is
    // templated on the variant via a type-tag (`std::type_identity`)
    // because flashinfer's variant types are not default-constructible.
    auto dispatch = [&]<class Variant>(::std::type_identity<Variant>) {
        if (head_dim == 64) {
            return prefill_dispatch_for_head_dim<64, ::flashinfer::MaskMode::kCausal, Variant>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        } else if (head_dim == 256) {
            return prefill_dispatch_for_head_dim<256, ::flashinfer::MaskMode::kCausal, Variant>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        } else if (head_dim == 512) {
            return prefill_dispatch_for_head_dim<512, ::flashinfer::MaskMode::kCausal, Variant>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        } else {
            return prefill_dispatch_for_head_dim<128, ::flashinfer::MaskMode::kCausal, Variant>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        }
    };
    if (logits_soft_cap > 0.f) {
        status = dispatch(::std::type_identity<AttnVariantSoftcap>{});
    } else {
        status = dispatch(::std::type_identity<AttnVariant>{});
    }
    CUDA_CHECK(status);
}

// ── Prefill with custom mask ───────────────────────────────────────────────

namespace {

// Custom-mask variant. Sliding window doesn't compose with `kCustom`
// (the user-supplied bitmap is the source of truth) so the `use_sliding_window`
// template flag stays off — the runtime mask predicate would only AND
// in a redundant window check.
using AttnVariantCustom = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/true,
    /*use_sliding_window=*/false,
    /*use_logits_soft_cap=*/false,
    /*use_alibi=*/false>;

using AttnVariantCustomSoftcap = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/true,
    /*use_sliding_window=*/false,
    /*use_logits_soft_cap=*/true,
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
    cudaStream_t stream,
    int /* window_left */,  // ignored — kCustom owns the mask
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        throw std::runtime_error(
            "flashinfer prefill (custom mask): instantiated for HEAD_DIM ∈ {64, 128, 256, 512}; got " +
            std::to_string(head_dim));
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

    // See note above re: head_dims unsupported by `VariableLengthMergeStates`.
    const bool head_dim_supports_split =
        (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512);

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
        static_cast<uint32_t>(head_dim), static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(page_size),
        /*enable_cuda_graph=*/false,
        sizeof(DTypeO),
        /*window_left=*/-1,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/!head_dim_supports_split,
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
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_q_heads / num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(num_q_heads);
    params.q_stride_n = static_cast<IdType>(num_q_heads * head_dim);
    params.q_stride_h = static_cast<IdType>(head_dim);
    params.window_left = -1;  // kCustom — caller-supplied bitmap is the source of truth
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : 1.0f / std::sqrt(static_cast<float>(head_dim));
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
    const bool enable_pdl = current_device_supports_pdl();

    // 4. Dispatch on (HEAD_DIM, cta_tile_q) with kCustom; pick variant
    //    based on logits_soft_cap (mirrors the kCausal path's dispatch).
    auto dispatch = [&](auto variant_tag) {
        using V = typename decltype(variant_tag)::type;
        if (head_dim == 64) {
            return prefill_dispatch_for_head_dim<64, ::flashinfer::MaskMode::kCustom, V>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        } else if (head_dim == 256) {
            return prefill_dispatch_for_head_dim<256, ::flashinfer::MaskMode::kCustom, V>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        } else if (head_dim == 512) {
            return prefill_dispatch_for_head_dim<512, ::flashinfer::MaskMode::kCustom, V>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        } else {
            return prefill_dispatch_for_head_dim<128, ::flashinfer::MaskMode::kCustom, V>(
                params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        }
    };
    if (logits_soft_cap > 0.f) {
        status = dispatch(::std::type_identity<AttnVariantCustomSoftcap>{});
    } else {
        status = dispatch(::std::type_identity<AttnVariantCustom>{});
    }
    CUDA_CHECK(status);
}

}  // namespace pie_cuda_driver::ops
