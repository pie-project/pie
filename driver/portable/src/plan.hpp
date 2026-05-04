#pragma once

// Plan: BPIQ wire payload → BatchPlan.
//
// Splits the work of `ForwardEngine::plan_` into composable phases:
//   1. extract_plan_arrays:   pull all 23+ typed views from the BPIQ wire blob
//   2. validate_plan_top_level: check the top-level invariants (batch shape)
//   3. resolve_active_adapter_id: enforce single-adapter-per-batch (v1)
//   4. plan_single_request:   build one ReqPlan + per-token positions/kv idxs
//   5. build_pure_decode_packing: M11 fast-path packing for the all-decode case
//
// `build_attn_mask_f16` (and the causal-only convenience wrapper) builds
// the per-request KQ mask required by `ggml_flash_attn_ext`.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "arch_spec.hpp"
#include "forward.hpp"
#include "shmem_schema.hpp"

namespace pie_portable_driver {

// ggml_flash_attn_ext requires the mask's row count to be a multiple of
// GGML_KQ_MASK_PAD (= 64). Pad accordingly when building per-request and
// packed-decode masks.
inline constexpr std::int32_t MASK_PAD = 64;

// Translate a logical position `p` in the request's sequence into the
// physical row index in the flat KV pool.
//   page = page_indices[indptr_off + p / page_size]
//   slot = p % page_size
//   physical = page * page_size + slot
inline std::int64_t physical_idx(const std::uint32_t* page_indices,
                                 std::int32_t indptr_off,
                                 std::int32_t page_size,
                                 std::int32_t pos) {
    const std::int32_t page_offset = pos / page_size;
    const std::int32_t slot        = pos % page_size;
    const std::int32_t page        = static_cast<std::int32_t>(
        page_indices[indptr_off + page_offset]);
    return static_cast<std::int64_t>(page) * page_size + slot;
}

// Per-token BRLE attention-mask override (M6 custom attention masks).
struct PerTokenMaskRuns {
    const std::uint32_t* runs;
    std::size_t          n_runs;
};

// Build the F16 KQ mask required by `ggml_flash_attn_ext` for one request.
// Layout: `[n_kv, n_tokens_pad]`. 0.0 where token i can attend to position
// j (j <= positions[i]); -INF otherwise (including padding rows).
//
// `per_token_runs` (when non-null) is an optional per-token BRLE override.
// When provided, that row of the mask is built from the BRLE runs instead
// of the default causal pattern. The BRLE alternates false/true starting
// with false, like logit masks. Causal beyond positions[i] is still
// enforced (BRLE can't unmask the future).
//
// `sliding_window > 0` clips the past below `positions[i] - W + 1`.
void build_attn_mask_f16(std::vector<std::uint16_t>& dst,
                         std::int32_t n_kv,
                         std::int32_t n_tokens,
                         std::int32_t n_tokens_pad,
                         const std::int32_t* positions,
                         const PerTokenMaskRuns* per_token_runs,
                         std::int32_t sliding_window = 0);

// Backward-compat shim for the offline test plans: causal-only, no SWA.
void build_causal_mask_f16(std::vector<std::uint16_t>& dst,
                           std::int32_t n_kv,
                           std::int32_t n_tokens,
                           std::int32_t n_tokens_pad,
                           const std::int32_t* positions);

// Typed views into the BPIQ wire payload. Shapes are validated by
// `validate_plan_top_level` and the per-request planner.
struct PlanArrays {
    std::span<const std::uint64_t> context_ids;
    std::span<const std::uint32_t> token_ids;
    std::span<const std::uint32_t> position_ids;
    std::span<const std::uint32_t> qo_indptr;
    std::span<const std::uint32_t> sampling_idx;
    std::span<const std::uint32_t> sampling_indptr;
    std::span<const std::uint32_t> request_num_samplers;
    std::span<const std::uint32_t> kv_page_indices;
    std::span<const std::uint32_t> kv_page_indptr;
    std::span<const std::uint32_t> kv_last_lens;
    std::span<const std::uint32_t> sampler_types;
    std::span<const float>         sampler_temps;
    std::span<const std::uint32_t> sampler_top_k;
    std::span<const float>         sampler_top_p;
    std::span<const float>         sampler_min_p;
    std::span<const std::uint32_t> sampler_seeds;
    std::span<const std::uint32_t> sampler_label_ids;
    std::span<const std::uint32_t> sampler_label_indptr;
    std::span<const std::uint32_t> logit_masks;
    std::span<const std::uint32_t> logit_mask_indptr;
    std::span<const std::uint32_t> flat_attn_masks;
    std::span<const std::uint32_t> attn_mask_indptr;
    std::span<const std::int64_t>  adapter_indices;
    std::span<const std::uint32_t> spec_token_ids;
    std::span<const std::uint32_t> spec_position_ids;
    std::span<const std::uint32_t> spec_indptr;

    std::int32_t n_request = 0;
    std::int32_t total_n_tokens = 0;
    bool         batch_has_drafts = false;
    bool         batch_has_attn_masks = false;
};

PlanArrays extract_plan_arrays(const schema::DecodedRequest& req);
void validate_plan_top_level(const PlanArrays& a);

// Returns the active adapter id (-1 if no request set one). Throws if
// requests in the same batch ask for different adapters — v1 enforces
// a single LoRA per fire_batch.
std::int64_t resolve_active_adapter_id(const PlanArrays& a);

void plan_single_request(const PlanArrays& a,
                         std::int32_t r,
                         std::int32_t page_size,
                         std::int32_t total_pages,
                         const ArchSpec& spec,
                         ForwardEngine::BatchPlan& plan);

// M11 packed-decode fast path. Caller has already verified every request
// has n_tokens == 1 and there are no custom attention masks, so the whole
// batch can be expressed as a single ne33-broadcast attention call per
// layer. Builds packed gather idxs + a single mask tensor of shape
// [max_n_kv, 64, 1, n_request]. Caller is responsible for setting
// plan.max_n_kv to the max n_kv across the batch's ReqPlans before
// calling. `sliding_window > 0` clips per-stream attention to
// [n_kv - W, n_kv); pass 0 for plain causal.
// `also_build_no_swa_mask`: when true AND sliding_window > 0, ALSO
// builds an additional mask in `packed_mask_full_f16` with NO sliding
// clip. Used by archs with mixed sliding+full layer patterns (Gemma 4)
// so the full-attention layers can attend the entire context.
void build_pure_decode_packing(ForwardEngine::BatchPlan& plan,
                               std::int32_t n_request,
                               std::int32_t page_size,
                               std::int32_t sliding_window,
                               bool also_build_no_swa_mask = false);

}  // namespace pie_portable_driver
