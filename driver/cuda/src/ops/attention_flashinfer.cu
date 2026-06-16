#include "ops/attention_flashinfer.hpp"

#include <algorithm>
#include <type_traits>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
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
#include "kernels/kv_paged.hpp"
#include "ops/attention_flashinfer_hopper.hpp"

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

inline ::flashinfer::QKVLayout kv_layout(bool hnd_layout) {
    return hnd_layout ? ::flashinfer::QKVLayout::kHND
                      : ::flashinfer::QKVLayout::kNHD;
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

bool force_split_kv_small_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_FLASHINFER_FORCE_SPLIT_KV_SMALL");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool static_nonsplit_decode_plan_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_CUDA_STATIC_DECODE_PLAN");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

std::size_t align_up_bytes(std::size_t n, std::size_t alignment) {
    return (n + alignment - 1) / alignment * alignment;
}

template <uint32_t HEAD_DIM, uint32_t GROUP_SIZE, class Variant>
struct DecodeWorkEstimator {
    cudaError_t operator()(bool& split_kv, uint32_t& max_grid_size,
                           uint32_t& max_num_pages_per_batch, uint32_t& new_batch_size,
                           uint32_t& gdy, uint32_t batch_size, IdType* kv_indptr_h,
                           uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                           cudaStream_t stream)
    {
        const auto rc = ::flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, HEAD_DIM, POS_ENC, Variant, DecodeParams>(
            split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
            batch_size, kv_indptr_h, num_qo_heads, page_size, enable_cuda_graph, stream);
        if constexpr (!head_dim_supports_cascade_merge(HEAD_DIM)) {
            split_kv = false;
            new_batch_size = batch_size;

            // With split_kv forced false, DecodePlan sets padded_batch_size =
            // batch_size, but DecodeSplitKVIndptr still tiles each request's
            // KV pages into ceil_div(pages_i, max_num_pages_per_batch) work
            // items using the chunk size the (split-kv) estimator chose
            // above. If any request has more pages than that chunk size,
            // DecodeSplitKVIndptr emits more than `batch_size` (request,
            // tile) entries, overflowing the request_indices/kv_tile_indices/
            // o_indptr buffers (sized for padded_batch_size = batch_size) and
            // mapping multiple grid blocks onto the same request — clobbering
            // other requests' output rows. Bump the chunk size to the largest
            // per-request page count so every request maps to exactly one
            // tile, keeping request_indices_vec.size() == batch_size ==
            // padded_batch_size. (kv_chunk_size_ptr is otherwise unused when
            // partition_kv == false.)
            uint32_t max_pages_per_req = 1;
            for (uint32_t i = 0; i < batch_size; ++i) {
                uint32_t pages_i = static_cast<uint32_t>(kv_indptr_h[i + 1] - kv_indptr_h[i]);
                max_pages_per_req = std::max(max_pages_per_req, pages_i);
            }
            max_num_pages_per_batch = std::max(max_num_pages_per_batch, max_pages_per_req);
        }
        if (!force_split_kv_small_enabled() &&
            current_device_major() >= 8 && batch_size <= 512) {
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
    int num_pages_in_batch = 0;
    bool enable_pdl = false;
    bool full_attention_variant = false;
    bool hnd_layout = false;
    bool valid = false;
    int static_nonsplit_num_requests = 0;
    std::vector<IdType> static_request_indices;
    std::vector<IdType> static_kv_tile_indices;
    std::vector<IdType> static_o_indptr;
    std::vector<IdType> indptr_h_buf;
};

struct PrefillPlanCache {
    ::flashinfer::PrefillPlanInfo plan_info;
    HopperPrefillPlan sm90_plan;
    int total_tokens = 0;
    int num_requests = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int page_size = 0;
    int window_left = -1;
    bool full_attention_variant = false;
    bool causal_mask = true;
    bool hnd_layout = false;
    bool use_sm90 = false;
    bool enable_pdl = false;
    bool valid = false;
    std::vector<IdType> qo_h_buf;
    std::vector<IdType> kv_h_buf;
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

std::uint32_t decode_plan_graph_layout(const DecodePlanCache& cache) {
    if (!cache.valid) return 0;
    return static_cast<std::uint32_t>(
        (cache.plan_info.split_kv ? 2u : 1u) |
        (cache.full_attention_variant ? 4u : 0u) |
        (cache.hnd_layout ? 8u : 0u));
}

std::uint32_t prefill_plan_graph_layout(const PrefillPlanCache& cache) {
    if (!cache.valid) return 0;
    if (cache.use_sm90) {
        return 0x00800000u |
               static_cast<std::uint32_t>(
                   hopper_prefill_graph_layout(cache.sm90_plan));
    }
    std::uint32_t tile_class = 0;
    switch (cache.plan_info.cta_tile_q) {
        case 16:  tile_class = 1; break;
        case 32:  tile_class = 2; break;
        case 64:  tile_class = 3; break;
        case 128: tile_class = 4; break;
        default:  tile_class = 0; break;
    }
    const std::uint32_t variant_class =
        cache.full_attention_variant ? 1u : 0u;
    const auto padded_batch_size = static_cast<std::uint32_t>(
        std::min<std::int64_t>(cache.plan_info.padded_batch_size,
                               0x000fffff));
    return static_cast<std::uint32_t>(
        0x100u |
        (cache.plan_info.split_kv ? 1u : 0u) |
        (tile_class << 1) |
        (variant_class << 4) |
        (cache.hnd_layout ? 32u : 0u) |
        (cache.causal_mask ? 64u : 0u) |
        (padded_batch_size << 8));
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
    bool enable_cuda_graph,
    bool full_attention_variant)
{
    auto plan_for = [&](auto work_estimator) {
        if (full_attention_variant) {
            return ::flashinfer::DecodePlan<
                HEAD_DIM, POS_ENC, AttnVariantFull, DecodeParams>(
                workspace.float_buffer(), workspace.float_bytes(),
                workspace.int_buffer(), workspace.page_locked_int(),
                workspace.int_bytes(),
                cache.plan_info,
                const_cast<IdType*>(indptr_h_buf.data()),
                num_requests, num_q_heads, page_size,
                enable_cuda_graph,
                stream, work_estimator);
        }
        return ::flashinfer::DecodePlan<
            HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
                workspace.float_buffer(), workspace.float_bytes(),
                workspace.int_buffer(), workspace.page_locked_int(),
                workspace.int_bytes(),
                cache.plan_info,
                const_cast<IdType*>(indptr_h_buf.data()),
                num_requests, num_q_heads, page_size,
                enable_cuda_graph,
                stream, work_estimator);
    };
    // Must match the kernel-side DISPATCH_GQA_GROUP_SIZE set in
    // flashinfer/utils.cuh ({1, 2, 3, 4, 8}). Other group sizes (5/6/7)
    // are routed to the prefill path by main.cpp's force_prefill_path
    // gate, so they never reach this dispatch.
    switch (gqa_group_size) {
        case 1:
            return full_attention_variant
                ? plan_for(DecodeWorkEstimator<HEAD_DIM, 1, AttnVariantFull>{})
                : plan_for(DecodeWorkEstimator<HEAD_DIM, 1, AttnVariant>{});
        case 2:
            return full_attention_variant
                ? plan_for(DecodeWorkEstimator<HEAD_DIM, 2, AttnVariantFull>{})
                : plan_for(DecodeWorkEstimator<HEAD_DIM, 2, AttnVariant>{});
        case 3:
            return full_attention_variant
                ? plan_for(DecodeWorkEstimator<HEAD_DIM, 3, AttnVariantFull>{})
                : plan_for(DecodeWorkEstimator<HEAD_DIM, 3, AttnVariant>{});
        case 4:
            return full_attention_variant
                ? plan_for(DecodeWorkEstimator<HEAD_DIM, 4, AttnVariantFull>{})
                : plan_for(DecodeWorkEstimator<HEAD_DIM, 4, AttnVariant>{});
        case 8:
            return full_attention_variant
                ? plan_for(DecodeWorkEstimator<HEAD_DIM, 8, AttnVariantFull>{})
                : plan_for(DecodeWorkEstimator<HEAD_DIM, 8, AttnVariant>{});
    }
    throw std::runtime_error(
        "flashinfer decode: unsupported GQA group size " +
        std::to_string(gqa_group_size) + " (instantiated: 1, 2, 3, 4, 8)");
}

bool can_use_static_nonsplit_decode_plan(uint32_t num_requests) {
    // DecodeWorkEstimator above already overrides FlashInfer's split-kv choice
    // to false for the TP1 latency shapes we care about. In that case the
    // schedule is independent of KV lengths, so avoid rerunning the full
    // FlashInfer planner for every decode batch.
    return static_nonsplit_decode_plan_enabled() &&
           current_device_major() >= 8 &&
           num_requests > 0 &&
           num_requests <= 512 &&
           !force_split_kv_small_enabled();
}

void refresh_static_nonsplit_decode_vectors(
    DecodePlanCache& cache,
    int num_requests)
{
    if (cache.static_nonsplit_num_requests == num_requests) {
        return;
    }
    cache.static_nonsplit_num_requests = num_requests;
    cache.static_request_indices.resize(num_requests);
    cache.static_kv_tile_indices.assign(num_requests, IdType{0});
    cache.static_o_indptr.resize(num_requests + 1);
    for (int r = 0; r < num_requests; ++r) {
        cache.static_request_indices[r] = static_cast<IdType>(r);
        cache.static_o_indptr[r] = static_cast<IdType>(r);
    }
    cache.static_o_indptr[num_requests] = static_cast<IdType>(num_requests);
}

void plan_static_nonsplit_decode(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool full_attention_variant,
    bool hnd_layout)
{
    refresh_static_nonsplit_decode_vectors(cache, num_requests);

    std::size_t cursor = 0;
    auto alloc = [&](std::size_t bytes, std::size_t alignment) {
        cursor = align_up_bytes(cursor, alignment);
        const std::size_t offset = cursor;
        cursor += bytes;
        return static_cast<std::int64_t>(offset);
    };

    auto& plan = cache.plan_info;
    plan.padded_batch_size = num_requests;
    plan.v_offset = 0;
    plan.s_offset = 0;
    plan.request_indices_offset =
        alloc(sizeof(IdType) * static_cast<std::size_t>(num_requests), 16);
    plan.kv_tile_indices_offset =
        alloc(sizeof(IdType) * static_cast<std::size_t>(num_requests), 16);
    plan.o_indptr_offset =
        alloc(sizeof(IdType) * static_cast<std::size_t>(num_requests + 1), 16);
    plan.kv_chunk_size_ptr_offset = alloc(sizeof(IdType), 1);
    plan.block_valid_mask_offset = 0;
    plan.enable_cuda_graph = enable_cuda_graph;
    plan.split_kv = false;

    if (cursor > workspace.int_bytes()) {
        throw std::runtime_error(
            "flashinfer decode static plan: attention int workspace too small");
    }

    auto* host = static_cast<std::uint8_t*>(workspace.page_locked_int());
    std::memcpy(host + plan.request_indices_offset,
                cache.static_request_indices.data(),
                sizeof(IdType) * static_cast<std::size_t>(num_requests));
    std::memcpy(host + plan.kv_tile_indices_offset,
                cache.static_kv_tile_indices.data(),
                sizeof(IdType) * static_cast<std::size_t>(num_requests));
    std::memcpy(host + plan.o_indptr_offset,
                cache.static_o_indptr.data(),
                sizeof(IdType) * static_cast<std::size_t>(num_requests + 1));
    *reinterpret_cast<IdType*>(host + plan.kv_chunk_size_ptr_offset) =
        static_cast<IdType>(page_size);

    CUDA_CHECK(cudaMemcpyAsync(
        workspace.int_buffer(), workspace.page_locked_int(), cursor,
        cudaMemcpyHostToDevice, stream));

    cache.num_requests = num_requests;
    cache.num_q_heads  = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim     = head_dim;
    cache.page_size    = page_size;
    cache.num_pages_in_batch = kv_page_indptr_h[num_requests];
    cache.enable_pdl   = current_device_supports_pdl();
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout   = hnd_layout;
    cache.valid        = true;
}

}  // namespace

void plan_attention_flashinfer_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool full_attention_variant,
    bool hnd_layout)
{
    const int gqa_group_size = num_q_heads / num_kv_heads;

    if (can_use_static_nonsplit_decode_plan(
            static_cast<uint32_t>(num_requests))) {
        plan_static_nonsplit_decode(
            cache, kv_page_indptr_h, num_requests, num_q_heads, num_kv_heads,
            head_dim, page_size, workspace, stream, enable_cuda_graph,
            full_attention_variant, hnd_layout);
        return;
    }

    cache.indptr_h_buf.resize(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        cache.indptr_h_buf[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    cudaError_t status;
    switch (head_dim) {
        case 64:
            status = plan_decode_for_head_dim<64>(
                cache, cache.indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph, full_attention_variant);
            break;
        case 96:
            status = plan_decode_for_head_dim<96>(
                cache, cache.indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph, full_attention_variant);
            break;
        case 128:
            status = plan_decode_for_head_dim<128>(
                cache, cache.indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph, full_attention_variant);
            break;
        case 256:
            status = plan_decode_for_head_dim<256>(
                cache, cache.indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph, full_attention_variant);
            break;
        case 512:
            status = plan_decode_for_head_dim<512>(
                cache, cache.indptr_h_buf,
                num_requests, num_q_heads, page_size, gqa_group_size,
                workspace, stream, enable_cuda_graph, full_attention_variant);
            break;
        default:
            throw std::runtime_error(
                "flashinfer decode: unsupported head_dim " +
                std::to_string(head_dim) +
                " (instantiated: 64, 96, 128, 256, 512)");
    }
    CUDA_CHECK(status);

    cache.num_requests = num_requests;
    cache.num_q_heads  = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim     = head_dim;
    cache.page_size    = page_size;
    cache.num_pages_in_batch = kv_page_indptr_h[num_requests];
    cache.enable_pdl   = current_device_supports_pdl();
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout   = hnd_layout;
    cache.valid        = true;
}

void plan_attention_flashinfer_prefill_bf16(
    PrefillPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
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
    bool full_attention_variant,
    bool hnd_layout,
    bool causal_mask)
{
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        throw std::runtime_error(
            "flashinfer prefill plan: instantiated for HEAD_DIM in {64, 128, 256, 512}; got " +
            std::to_string(head_dim));
    }
    cache.use_sm90 = false;
    cache.sm90_plan.valid = false;
    if (!hnd_layout &&
        kv_last_page_lens_h != nullptr &&
        hopper_prefill_supported(
            head_dim, window_left, total_tokens, num_requests)) {
        plan_attention_flashinfer_prefill_sm90_bf16(
            cache.sm90_plan,
            qo_indptr_h,
            kv_page_indptr_h,
            kv_last_page_lens_h,
            total_tokens,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            workspace,
            stream,
            enable_cuda_graph,
            causal_mask,
            window_left);
        cache.total_tokens = total_tokens;
        cache.num_requests = num_requests;
        cache.num_q_heads = num_q_heads;
        cache.num_kv_heads = num_kv_heads;
        cache.head_dim = head_dim;
        cache.page_size = page_size;
        cache.window_left = window_left;
        cache.full_attention_variant = full_attention_variant;
        cache.causal_mask = causal_mask;
        cache.hnd_layout = hnd_layout;
        cache.enable_pdl = current_device_supports_pdl();
        cache.use_sm90 = true;
        cache.valid = true;
        return;
    }

    cache.qo_h_buf.resize(num_requests + 1);
    cache.kv_h_buf.resize(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        cache.qo_h_buf[r] = static_cast<IdType>(qo_indptr_h[r]);
        cache.kv_h_buf[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    const bool head_dim_supports_split =
        (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512);
    const bool disable_split_kv =
        !head_dim_supports_split;

    auto status = ::flashinfer::PrefillPlan<IdType>(
        workspace.float_buffer(), workspace.float_bytes(),
        workspace.int_buffer(), workspace.page_locked_int(),
        workspace.int_bytes(),
        cache.plan_info,
        cache.qo_h_buf.data(), cache.kv_h_buf.data(),
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
        disable_split_kv,
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
    cache.causal_mask = causal_mask;
    cache.hnd_layout = hnd_layout;
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
        kv_layout(cache.hnd_layout),
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
    if (cache.full_attention_variant && window_left < 0 && logits_soft_cap <= 0.f) {
        return dispatch_decode_for_head_dim_v<HEAD_DIM, AttnVariantFull>(
            cache, q, k_pages, v_pages, o,
            kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
    }
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
        case 96:
            status = dispatch_decode_for_head_dim<96>(cache, q, k_pages, v_pages, o,
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

void dispatch_attention_flashinfer_decode(
    const DecodePlanCache& cache,
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
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
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, cache.num_pages_in_batch, stream);
    dispatch_attention_flashinfer_decode_bf16(
        cache, q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
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
    if (cache.use_sm90) {
        dispatch_attention_flashinfer_prefill_sm90_bf16(
            cache.sm90_plan,
            q,
            k_pages,
            v_pages,
            o,
            kv_page_indices_d,
            workspace,
            stream,
            logits_soft_cap,
            sm_scale,
            lse_out);
        return;
    }

    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(cache.num_kv_heads),
        static_cast<uint32_t>(cache.page_size),
        static_cast<uint32_t>(cache.head_dim),
        static_cast<uint32_t>(cache.num_requests),
        kv_layout(cache.hnd_layout),
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

    auto dispatch_causal = [&]<class Variant>(::std::type_identity<Variant>) {
        switch (cache.head_dim) {
            case 64:
                return prefill_dispatch_for_head_dim<64, ::flashinfer::MaskMode::kCausal, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 128:
                return prefill_dispatch_for_head_dim<128, ::flashinfer::MaskMode::kCausal, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 256:
                return prefill_dispatch_for_head_dim<256, ::flashinfer::MaskMode::kCausal, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 512:
                return prefill_dispatch_for_head_dim<512, ::flashinfer::MaskMode::kCausal, Variant>(
                    params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
        }
        return cudaErrorInvalidValue;
    };
    auto dispatch_full = [&]<class Variant>(::std::type_identity<Variant>) {
        switch (cache.head_dim) {
            case 64:
                return cache.causal_mask
                    ? prefill_dispatch_for_head_dim<64, ::flashinfer::MaskMode::kCausal, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream)
                    : prefill_dispatch_for_head_dim<64, ::flashinfer::MaskMode::kNone, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 128:
                return cache.causal_mask
                    ? prefill_dispatch_for_head_dim<128, ::flashinfer::MaskMode::kCausal, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream)
                    : prefill_dispatch_for_head_dim<128, ::flashinfer::MaskMode::kNone, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 256:
                return cache.causal_mask
                    ? prefill_dispatch_for_head_dim<256, ::flashinfer::MaskMode::kCausal, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream)
                    : prefill_dispatch_for_head_dim<256, ::flashinfer::MaskMode::kNone, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
            case 512:
                return cache.causal_mask
                    ? prefill_dispatch_for_head_dim<512, ::flashinfer::MaskMode::kCausal, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream)
                    : prefill_dispatch_for_head_dim<512, ::flashinfer::MaskMode::kNone, Variant>(
                        params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream);
        }
        return cudaErrorInvalidValue;
    };

    cudaError_t status;
    if (cache.full_attention_variant && logits_soft_cap > 0.f) {
        status = dispatch_full(::std::type_identity<AttnVariantFullSoftcap>{});
    } else if (cache.full_attention_variant) {
        status = dispatch_full(::std::type_identity<AttnVariantFull>{});
    } else if (logits_soft_cap > 0.f) {
        status = dispatch_causal(::std::type_identity<AttnVariantSoftcap>{});
    } else {
        status = dispatch_causal(::std::type_identity<AttnVariant>{});
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
    float* lse_out,
    bool hnd_layout)
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
        kv_layout(hnd_layout),
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

    // 4. Dispatch on (HEAD_DIM, soft-cap variant). The lambda is
    // templated on the variant via a type-tag (`std::type_identity`)
    // because flashinfer's variant types are not default-constructible.
    auto dispatch = [&]<class Variant>(::std::type_identity<Variant>) {
        const bool enable_pdl = current_device_supports_pdl();
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

void launch_attention_flashinfer_prefill(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    const int num_pages_in_batch = kv_page_indptr_h[num_requests];
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, num_pages_in_batch, stream);
    launch_attention_flashinfer_prefill_bf16(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        qo_indptr_h, kv_page_indptr_h,
        total_tokens, num_requests, num_q_heads, kv_layer.num_kv_heads,
        kv_layer.head_dim, kv_layer.page_size, workspace, stream,
        window_left, logits_soft_cap, sm_scale,
        lse_out, kv_layer.hnd_layout);
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
    float* lse_out,
    bool hnd_layout)
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
        kv_layout(hnd_layout),
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

    // 4. Dispatch on (HEAD_DIM, cta_tile_q) with kCustom; pick variant
    //    based on logits_soft_cap (mirrors the kCausal path's dispatch).
    auto dispatch = [&](auto variant_tag) {
        using V = typename decltype(variant_tag)::type;
        const bool enable_pdl = current_device_supports_pdl();
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

void launch_attention_flashinfer_prefill_custom(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
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
    int num_q_heads,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    const int num_pages_in_batch = kv_page_indptr_h[num_requests];
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, num_pages_in_batch, stream);
    launch_attention_flashinfer_prefill_custom_bf16(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        mask_d, mask_indptr_d, qo_indptr_h, kv_page_indptr_h,
        total_tokens, num_requests, num_q_heads, kv_layer.num_kv_heads,
        kv_layer.head_dim, kv_layer.page_size, workspace, stream,
        window_left, logits_soft_cap, sm_scale,
        lse_out, kv_layer.hnd_layout);
}

}  // namespace pie_cuda_driver::ops
