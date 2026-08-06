//===-- attention_flashinfer_common.cuh -------------------------*- CUDA -*-===//
//
// Shared body of the FA2 attention driver, included by attention_flashinfer.cu
// (the primary translation unit, which owns the public entry points) and by
// one attention_flashinfer_hd<N>.cu per head_dim in src/kernels.def.
//
// Why the split: every FA2 entry point is templated on HEAD_DIM, and nvcc's
// front end (cicc) and assembler (ptxas) are single-threaded *per translation
// unit*. Instantiating all head_dims in one file made this the critical path of
// the whole CUDA build by a wide margin, with the rest of the job graph idle
// waiting on it. Fixed per-TU overhead is a few seconds, so trading one long
// TU for several concurrent short ones is close to a pure win.
//
// The mechanism is `AttnHd<HEAD_DIM>`: a class template whose members are
// ordinary (non-template) static functions, so a single explicit instantiation
// per head_dim pulls in everything that head_dim needs. Mask mode and attention
// variant are therefore selected *inside* the members -- an explicit
// instantiation of a class template does not instantiate its member templates,
// so any axis left as a template parameter would escape the split.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "ops/attention_flashinfer.hpp"

#include <algorithm>
#include <type_traits>

#include <cstdio>
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
#include "kernels_manifest.hpp"
#include "kernels/kv_paged.hpp"
#include "ops/attention_flashinfer_hopper.hpp"
#include "ops/attention_score_capture.cuh"

namespace pie_cuda_driver::ops {

// Deliberately a *named* namespace even though everything in it is private to
// the attention translation units: AttnHd's member definitions appear in every
// TU that includes this header, so anything they name must be one entity
// program-wide. An anonymous namespace would give each TU its own copies and
// make those definitions differ -- ill-formed, no diagnostic required.
namespace fa2 {

using DTypeQ  = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO  = __nv_bfloat16;
using IdType  = int32_t;

inline constexpr auto POS_ENC = ::flashinfer::PosEncodingMode::kNone;

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

using DecodeParams  = ::flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
using PrefillParams = ::flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// Score-observing decode (design §3). Both capture variants wrap a stock
// `DefaultAttention`, so attention output stays bit-identical to the
// uncaptured path; only the side-channel is new.
//
// Neither soft-cap nor sliding-window is instantiated here, and that is a
// deliberate refusal rather than an omission:
//
//  * soft-cap rewrites the score to `cap * tanh(s / cap)`, so what the hook
//    would record is not the quantity H2O/TOVA define their policy over;
//  * sliding-window masks positions in `LogitsMask`, which runs *after*
//    `LogitsTransform` -- captured rows would include scores the softmax
//    then discards, and the normalisation kernel (which derives its own
//    denominator) would silently spread mass onto masked positions.
//
// `dispatch_decode_capture` rejects both cases loudly instead.
using AttnScoreCapture     = PieScoreCapture<AttnVariant>;
using AttnScoreCaptureFull = PieScoreCapture<AttnVariantFull>;

using DecodeScoreParams = PieScoreParams<DecodeParams, IdType>;

// Score-observing prefill (SnapKV, design section 12). Only the full-attention
// causal variant is instantiated, for the same two reasons the decode capture
// refuses soft-cap and sliding-window: soft-cap records a quantity the policy
// is not defined over, and a sliding window masks in `LogitsMask`, which runs
// after `LogitsTransform`, so the normalisation kernel would spread mass onto
// positions the softmax discards.
using AttnScoreCapturePrefill = PieScoreCaptureWindow<AttnVariantFull>;
using PrefillScoreParams = PieScoreWindowParams<PrefillParams, IdType>;

/// Where a score-observing decode deposits `p[head, kv_idx]`, ragged over the
/// batch. Both pointers are device memory owned by the caller.
struct DecodeScoreSink {
    float* scores = nullptr;
    const IdType* indptr = nullptr;
};

/// Where a score-observing prefill deposits `p[head, window_row, kv_idx]`,
/// ragged over the batch. `window` is the observation-window width; a request
/// whose `qo_len` is shorter records `qo_len` rows and leaves the rest of its
/// slot untouched, so the caller must zero the buffer first.
struct PrefillScoreSink {
    float* scores = nullptr;
    const IdType* indptr = nullptr;
    std::uint32_t window = 0;
};

// flashinfer's `GetPtrFromBaseOffset` is `(base + offset_bytes) reinterpret to T*`.
template <typename T>
inline T* offset_ptr(void* base, std::int64_t off) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(base) + off);
}

inline ::flashinfer::QKVLayout kv_layout(bool hnd_layout) {
    return hnd_layout ? ::flashinfer::QKVLayout::kHND
                      : ::flashinfer::QKVLayout::kNHD;
}

/// Uniform diagnostic for a head_dim that kernels.def did not instantiate.
/// `where` names the call site, e.g. "flashinfer decode".
[[noreturn]] inline void throw_unsupported_head_dim(const char* where,
                                                    int head_dim) {
    throw std::runtime_error(
        std::string(where) + ": unsupported head_dim " +
        std::to_string(head_dim) + " (instantiated:" +
        attn_head_dim_list() + ")");
}

// Cascade-merge (`VariableLengthMergeStates`) only instantiates head_dim
// ∈ {64, 128, 256, 512}. For other head_dims (e.g. Phi-3-mini at 96), we
// force `split_kv = false` at planning time so the partition-kv path —
// and its post-merge — is never taken.
//
// Deliberately NOT kernels.def: this is upstream flashinfer's instantiation
// set, which pie does not control. It happens to equal our head_dim list
// today; routing it through the manifest would silently break the day the
// two diverge.
constexpr bool head_dim_supports_cascade_merge(uint32_t hd) {
    return hd == 64 || hd == 128 || hd == 256 || hd == 512;
}

inline int current_device_major() {
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

inline bool current_device_supports_pdl() {
    return current_device_major() >= 9;
}

// Whether to let FlashInfer's own work estimator decide split-kv at the small
// batch sizes the override below otherwise forces to non-split.
//
// Off, and the measurement says it has to stay off: the override is what makes
// the decode schedule independent of KV length, which is what lets
// `can_use_static_nonsplit_decode_plan` reuse one plan for every fire. Turn it
// on and the full FlashInfer planner runs per decoded token -- gemma-4-26B-A4B
// at 1k context goes from 22 s for 256 tokens to over 2400 s, a hundredfold,
// because the host planning dwarfs whatever the split buys the kernel.
//
// This is not an argument that non-split is optimal. At R=1 the unsplit grid
// is batch_size * num_kv_heads = 8 CTAs on 132 SMs, and gemma-4's decode
// attention measures ~50x off its KV-bandwidth roofline because of it. The
// argument is that the fix has to keep the plan static -- a precomputed split
// factor folded into the cached plan -- not hand planning back to the caller.
inline bool force_split_kv_small_enabled() {
    static const bool on =
        std::getenv("PIE_CUDA_FORCE_SPLIT_KV_SMALL") != nullptr;
    return on;
}

constexpr bool static_nonsplit_decode_plan_enabled() { return true; }

inline std::size_t align_up_bytes(std::size_t n, std::size_t alignment) {
    return (n + alignment - 1) / alignment * alignment;
}

// Wraps the templated work-estimation function so DecodePlan can call it.
// GROUP_SIZE = num_q_heads / num_kv_heads (GQA ratio); HEAD_DIM is the
// per-head feature width. Adding a (head_dim, group_size) pair adds one
// CUDA template instantiation, which dominates compile time — so both axes
// are enumerated in kernels.def rather than open-coded here.
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

}  // namespace fa2

using namespace fa2;  // NOLINT(google-build-using-namespace) -- private header

// ── Plan caches (private impl) ─────────────────────────────────────────────

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
    // True when the plan was built by `plan_static_nonsplit_decode`, whose
    // descriptor is `request_indices[r] = r`, `kv_tile_indices = 0`,
    // `o_indptr[r] = r`, `split_kv = false` -- a schedule that does NOT depend
    // on the page counts it was planned with. That independence is what lets a
    // caller hand the *launch* a different (compacted) page list than the one
    // it planned against, which is how `attn_page_mask` restricts a layer's
    // attention without a replan. Under any other plan the arrays ARE derived
    // from page counts, and substituting a shorter list is silently wrong.
    bool page_count_independent = false;
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
    /// The plan ran in graph mode (content-independent launch geometry) on
    /// the FA2 causal path — the executor may capture/replay the dispatch.
    /// False when graph mode was requested but demoted (SM90 route, split
    /// disabled for the head dim, or the graph carve exceeding the float
    /// workspace grant).
    bool graph_capturable = false;
    std::vector<IdType> qo_h_buf;
    std::vector<IdType> kv_h_buf;
};

// ── AttnHd: the per-head_dim instantiation unit ────────────────────────────

/// All FA2 work for one head_dim. Members are deliberately non-template so
/// that `template struct AttnHd<N>;` in a single .cu instantiates every kernel
/// that head_dim needs; see the file header for why that matters.
template <uint32_t HEAD_DIM>
struct AttnHd {
    static cudaError_t plan_decode(
        DecodePlanCache& cache,
        const std::vector<IdType>& indptr_h_buf,
        uint32_t num_requests, uint32_t num_q_heads, uint32_t page_size,
        int gqa_group_size,
        AttentionWorkspace& workspace,
        cudaStream_t stream,
        bool enable_cuda_graph,
        bool full_attention_variant);

    /// Soft-cap-aware decode dispatch. Routes to the plain, soft-cap or full
    /// variant; the plan that drove us here is computed against `AttnVariant`
    /// (no softcap), and the soft-cap kernel's shared-memory footprint is
    /// identical, so the plan stays correct across all three.
    static cudaError_t dispatch_decode(
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
        float* lse_out,
        bool broadcast_q = false);

    /// Score-observing decode. Same kernel, same plan, plus a
    /// `[num_qo_heads, kv_len]` per-request score row written through the
    /// variant's `LogitsTransform` hook. Returns `cudaErrorInvalidValue` if
    /// the configuration is one where a captured score would not mean what
    /// H2O/TOVA assume it means (soft-cap or sliding window).
    static cudaError_t dispatch_decode_capture(
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
        float* lse_out,
        const DecodeScoreSink& sink);

    /// Prefill with a built-in mask. `causal_mask` only reaches the kernel on
    /// the full-attention variants: the sliding-window variants are causal by
    /// construction, so a kNone instantiation of them would be dead.
    static cudaError_t prefill(
        PrefillParams& params,
        const ::flashinfer::PrefillPlanInfo& plan_info,
        DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream,
        bool full_attention_variant, bool causal_mask, float logits_soft_cap);

    /// Prefill that also records the observation window's scores (SnapKV).
    /// Refuses soft-cap and sliding-window loudly rather than recording a
    /// quantity the policy is not defined over.
    static cudaError_t prefill_capture(
        PrefillParams& base_params,
        const ::flashinfer::PrefillPlanInfo& plan_info,
        DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream,
        bool causal_mask, float logits_soft_cap, int window_left,
        const PrefillScoreSink& sink);

    /// Prefill against a caller-supplied mask bitmap (MaskMode::kCustom).
    static cudaError_t prefill_custom(
        PrefillParams& params,
        const ::flashinfer::PrefillPlanInfo& plan_info,
        DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream,
        float logits_soft_cap);

private:
    template <::flashinfer::MaskMode MASK, class Variant,
              class Params = PrefillParams>
    static cudaError_t run_prefill(
        Params& params, const ::flashinfer::PrefillPlanInfo& plan_info,
        DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream);

    template <class Variant, class Params = DecodeParams>
    static cudaError_t run_decode(
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
        float* lse_out,
        const DecodeScoreSink* sink = nullptr,
        // Every request reads the same Q row: the KV split issues one query
        // against several page slices, so only the input is shared.
        bool broadcast_q = false);
};

// ── AttnHd definitions ─────────────────────────────────────────────────────

template <uint32_t HEAD_DIM>
cudaError_t AttnHd<HEAD_DIM>::plan_decode(
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
    // Must be a subset of the kernel-side DISPATCH_GQA_GROUP_SIZE set in
    // flashinfer/utils.cuh ({1, 2, 3, 4, 8}). Group sizes outside that set
    // (5/6/7) are routed to the prefill path by main.cpp's force_prefill_path
    // gate, so they never reach this dispatch.
    switch (gqa_group_size) {
#define PIE_ATTN_DECODE_GQA(G)                                               \
        case G:                                                              \
            return full_attention_variant                                    \
                ? plan_for(DecodeWorkEstimator<HEAD_DIM, G, AttnVariantFull>{}) \
                : plan_for(DecodeWorkEstimator<HEAD_DIM, G, AttnVariant>{});
#include "kernels.def"
        default:
            break;
    }
    throw std::runtime_error(
        "flashinfer decode: unsupported GQA group size " +
        std::to_string(gqa_group_size) + " (instantiated:" +
        attn_decode_gqa_list() + ")");
}

template <uint32_t HEAD_DIM>
template <class Variant, class Params>
cudaError_t AttnHd<HEAD_DIM>::run_decode(
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
    float* lse_out,
    const DecodeScoreSink* sink,
    bool broadcast_q)
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

    Params params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.q_rope_offset = nullptr;
    params.paged_kv = paged_kv;
    params.o = static_cast<DTypeO*>(o);
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.num_qo_heads = static_cast<uint32_t>(cache.num_q_heads);
    params.q_stride_n = broadcast_q
        ? IdType{0}
        : static_cast<IdType>(cache.num_q_heads * cache.head_dim);
    params.q_stride_h = static_cast<IdType>(cache.head_dim);
    params.window_left = window_left;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : (1.0f / std::sqrt(static_cast<float>(cache.head_dim)));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    // `PieScoreCapture` records `kv_idx` verbatim, and the kernel derives that
    // from `paged_kv.rope_pos_offset` (`decode.cuh:541`). pie leaves it null,
    // which is what makes `kv_idx` an absolute KV index; if a future forward
    // graph ever sets it, every captured score silently lands at the wrong
    // position. Cheap to assert, impossible to notice otherwise.
    if constexpr (!std::is_same_v<Params, DecodeParams>) {
        if (paged_kv.rope_pos_offset != nullptr) {
            return cudaErrorInvalidValue;
        }
        if (sink == nullptr || sink->scores == nullptr ||
            sink->indptr == nullptr) {
            return cudaErrorInvalidValue;
        }
        params.score_out = sink->scores;
        params.score_indptr = sink->indptr;
    }

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

    // Bug#2 device R>1 diagnostic (PIE_DECODE_PARAM_DUMP): the concurrent-decode
    // corruption is per-request KV mis-attribution inside BatchDecode. Everything
    // in the plan/kernel is per-request-correct in code, so dump the RUNTIME
    // per-request KV bound + plan work-distribution the kernel actually reads —
    // the wrong field (kv_len, request_indices, o_indptr, padded_batch_size, or
    // batch_size) is the fix site. Env-gated, D2H copies (heavy) — off by default.
    if constexpr (false) {
        const int R = static_cast<int>(cache.num_requests);
        std::vector<IdType> h_indptr(R + 1), h_lastlen(R), h_reqidx(R), h_oindptr(R + 1);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_indptr.data(), kv_page_indptr_d, sizeof(IdType) * (R + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lastlen.data(), kv_last_page_lens_d, sizeof(IdType) * R, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_reqidx.data(), params.request_indices, sizeof(IdType) * R, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_oindptr.data(), params.o_indptr, sizeof(IdType) * (R + 1), cudaMemcpyDeviceToHost);
        std::fprintf(stderr,
            "[DECODE_PARAM] R=%d padded_batch=%u split_kv=%d paged_kv.batch_size=%u page_size=%d\n",
            R, params.padded_batch_size, static_cast<int>(params.partition_kv),
            static_cast<unsigned>(cache.num_requests), cache.page_size);
        for (int r = 0; r < R; ++r) {
            const int pages = static_cast<int>(h_indptr[r + 1]) - static_cast<int>(h_indptr[r]);
            const int kv_len = (pages - 1) * cache.page_size + static_cast<int>(h_lastlen[r]);
            std::fprintf(stderr,
                "[DECODE_PARAM]  r=%d indptr=[%d,%d) pages=%d last_page_len=%d kv_len=%d req_idx=%d o_indptr=%d\n",
                r, static_cast<int>(h_indptr[r]), static_cast<int>(h_indptr[r + 1]), pages,
                static_cast<int>(h_lastlen[r]), kv_len, static_cast<int>(h_reqidx[r]),
                static_cast<int>(h_oindptr[r]));
        }
    }

    return ::flashinfer::BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM, POS_ENC, Variant, Params>(
        params, tmp_v, tmp_s, cache.enable_pdl, stream);
}

template <uint32_t HEAD_DIM>
cudaError_t AttnHd<HEAD_DIM>::dispatch_decode(
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
    float* lse_out,
    bool broadcast_q)
{
    if (cache.full_attention_variant && window_left < 0 && logits_soft_cap <= 0.f) {
        return run_decode<AttnVariantFull, DecodeParams>(
            cache, q, k_pages, v_pages, o,
            kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out,
            /*sink=*/nullptr, broadcast_q);
    }
    if (logits_soft_cap > 0.f) {
        return run_decode<AttnVariantSoftcap, DecodeParams>(
            cache, q, k_pages, v_pages, o,
            kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out,
            /*sink=*/nullptr, broadcast_q);
    }
    return run_decode<AttnVariant, DecodeParams>(
        cache, q, k_pages, v_pages, o,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        workspace, stream, window_left, /*soft_cap=*/0.f, sm_scale, lse_out,
        /*sink=*/nullptr, broadcast_q);
}

template <uint32_t HEAD_DIM>
cudaError_t AttnHd<HEAD_DIM>::dispatch_decode_capture(
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
    float* lse_out,
    const DecodeScoreSink& sink)
{
    // Refuse rather than capture a score that does not mean what the caller
    // thinks. See the `AttnScoreCapture` alias for why each case is excluded.
    if (logits_soft_cap > 0.f || window_left >= 0) {
        return cudaErrorInvalidValue;
    }
    if (cache.full_attention_variant) {
        return run_decode<AttnScoreCaptureFull, DecodeScoreParams>(
            cache, q, k_pages, v_pages, o,
            kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            workspace, stream, window_left, /*soft_cap=*/0.f, sm_scale,
            lse_out, &sink);
    }
    return run_decode<AttnScoreCapture, DecodeScoreParams>(
        cache, q, k_pages, v_pages, o,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        workspace, stream, window_left, /*soft_cap=*/0.f, sm_scale,
        lse_out, &sink);
}

template <uint32_t HEAD_DIM>
template <::flashinfer::MaskMode MASK, class Variant, class Params>
cudaError_t AttnHd<HEAD_DIM>::run_prefill(
    Params& params, const ::flashinfer::PrefillPlanInfo& plan_info,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream)
{
    cudaError_t status;
    // CTA_TILE_Q is chosen by the planner from the batch's token count, so it
    // is a runtime axis: all four tiles stay instantiated.
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
        status = ::flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENC,
            /*USE_FP16_QK_REDUCTION=*/true,
            MASK,
            Variant, Params>(
            params, tmp_v, tmp_s, enable_pdl, stream);
    });
    return status;
}

template <uint32_t HEAD_DIM>
cudaError_t AttnHd<HEAD_DIM>::prefill(
    PrefillParams& params,
    const ::flashinfer::PrefillPlanInfo& plan_info,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream,
    bool full_attention_variant, bool causal_mask, float logits_soft_cap)
{
    using ::flashinfer::MaskMode;
    if (full_attention_variant) {
        if (logits_soft_cap > 0.f) {
            return causal_mask
                ? run_prefill<MaskMode::kCausal, AttnVariantFullSoftcap>(
                      params, plan_info, tmp_v, tmp_s, enable_pdl, stream)
                : run_prefill<MaskMode::kNone, AttnVariantFullSoftcap>(
                      params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
        }
        return causal_mask
            ? run_prefill<MaskMode::kCausal, AttnVariantFull>(
                  params, plan_info, tmp_v, tmp_s, enable_pdl, stream)
            : run_prefill<MaskMode::kNone, AttnVariantFull>(
                  params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
    }
    if (logits_soft_cap > 0.f) {
        return run_prefill<MaskMode::kCausal, AttnVariantSoftcap>(
            params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
    }
    return run_prefill<MaskMode::kCausal, AttnVariant>(
        params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
}

template <uint32_t HEAD_DIM>
cudaError_t AttnHd<HEAD_DIM>::prefill_capture(
    PrefillParams& base_params,
    const ::flashinfer::PrefillPlanInfo& plan_info,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream,
    bool causal_mask, float logits_soft_cap, int window_left,
    const PrefillScoreSink& sink)
{
    using ::flashinfer::MaskMode;
    if (logits_soft_cap > 0.f || window_left >= 0) return cudaErrorInvalidValue;
    if (sink.scores == nullptr || sink.indptr == nullptr || sink.window == 0) {
        return cudaErrorInvalidValue;
    }
    PrefillScoreParams params;
    static_cast<PrefillParams&>(params) = base_params;
    params.score_out = sink.scores;
    params.score_indptr = sink.indptr;
    params.score_window = sink.window;
    return causal_mask
        ? run_prefill<MaskMode::kCausal, AttnScoreCapturePrefill,
                      PrefillScoreParams>(
              params, plan_info, tmp_v, tmp_s, enable_pdl, stream)
        : run_prefill<MaskMode::kNone, AttnScoreCapturePrefill,
                      PrefillScoreParams>(
              params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
}

template <uint32_t HEAD_DIM>
cudaError_t AttnHd<HEAD_DIM>::prefill_custom(
    PrefillParams& params,
    const ::flashinfer::PrefillPlanInfo& plan_info,
    DTypeO* tmp_v, float* tmp_s, bool enable_pdl, cudaStream_t stream,
    float logits_soft_cap)
{
    using ::flashinfer::MaskMode;
    if (logits_soft_cap > 0.f) {
        return run_prefill<MaskMode::kCustom, AttnVariantCustomSoftcap>(
            params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
    }
    return run_prefill<MaskMode::kCustom, AttnVariantCustom>(
        params, plan_info, tmp_v, tmp_s, enable_pdl, stream);
}

// One explicit instantiation definition per head_dim lives in its own
// attention_flashinfer_hd<N>.cu; these declarations keep every other
// translation unit — including the primary one — from instantiating them
// again. An explicit instantiation definition is allowed to follow this
// declaration in the same TU, which is exactly what the hd<N>.cu files do.
#define PIE_ATTN_HEAD_DIM(HD) extern template struct AttnHd<HD>;
#include "kernels.def"

}  // namespace pie_cuda_driver::ops
