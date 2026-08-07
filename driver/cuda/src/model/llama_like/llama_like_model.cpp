#include "model/llama_like/llama_like_model.hpp"

#include "model/stage_hooks.hpp"
#include "ops/gemm.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace pie_cuda_driver::model {

namespace {

// Eligible fires run the DECLARED executor (`declared_forward.cpp`),
// which builds the fire's rows, lowers them against the traced plan and
// executes the resulting launch list. `PIE_DECLARED_FORWARD=0` disarms
// it back onto the hand-written body below.
//
// The flip is `.wiki/tart/dsl.md` cutover step 4(a), and it is smaller
// than it sounds in both directions. It does NOT make this the only
// path: a deployment the DSL cannot express never traces at all
// (`declared_` stays empty for TP, quantized projections, non-standard
// rope), and an expressible deployment still falls back PER FIRE
// wherever `declared_eligible` below refuses. So this changes nothing
// for the excluded and everything for the included, and the
// hand-written body stays where it is until each of those refusals has
// been closed on its own.
//
// Cached like `decode_fused_post_enabled` in llama_like.cpp so the gate
// costs one load.
bool declared_forward_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD");
        return v == nullptr || v[0] == '\0' || v[0] != '0';
    }();
    return enabled;
}

// Why a fire took the hand-written body instead of the declared one.
//
// The names are the terms of `body()`'s eligibility test, one each, and
// that is the point: after the default flip the only remaining question
// is which fires still fall back, and a work list has to say WHICH RULE
// refused, not how many did.
enum class DeclineReason {
    None,
    /// The deployment never traced — TP, quantized projections,
    /// non-standard rope. Not a fire property and not closeable per
    /// fire: this one is a DSL vocabulary question.
    NoPlan,
    WriteDescMissing,
    SlidingWindow,
    /// A PADDED-HEAD deployment asked to NARROW rows under the depth
    /// axis (a union or banded fire). Phi-3-mini pads head_dim 96 -> 128,
    /// so q/k stage at PHYSICAL width while a row window addresses at
    /// logical width. Stopping after layer `k` is free — the retired ops
    /// simply do not run, and address nothing — which is why the trace
    /// states the axis and this refuses only the half that costs.
    PaddedHeadNarrowing,
    /// A UNION prefill: full-depth rows beside truncated ones, so the
    /// tail layers would run over a row prefix and this fire's qo/kv
    /// CSRs would have to narrow with them. There is no prefill
    /// analogue of `depth_prefix_decode_plan` to narrow them against —
    /// a prepare-side piece of work, not a driver one.
    UnionPrefill,
    /// A truncated DECODE whose trace does not state the axis — the XQA
    /// and padded-head deployments, where `family.rs` withholds it
    /// because the body cannot honour it there.
    TruncatedAxisUnstated,
    FusedQkvUnstaged,
    /// A banded decode fire with a live band whose prefix plan the
    /// prepare did not stamp — the 14B device-geometry envelope class.
    BandedPlanMissing,
};

const char* decline_name(DeclineReason r) {
    switch (r) {
    case DeclineReason::None:               return "none";
    case DeclineReason::NoPlan:             return "no-plan";
    case DeclineReason::WriteDescMissing:   return "write-desc-missing";
    case DeclineReason::SlidingWindow:      return "sliding-window";
    case DeclineReason::PaddedHeadNarrowing:
        return "padded-head-narrowing";
    case DeclineReason::UnionPrefill:      return "union-prefill";
    case DeclineReason::TruncatedAxisUnstated:
        return "truncated-axis-unstated";
    case DeclineReason::FusedQkvUnstaged:   return "fused-qkv-unstaged";
    case DeclineReason::BandedPlanMissing:  return "banded-plan-missing";
    }
    return "?";
}

// One LOUD line the first time each reason is seen, and a per-fire line
// under `PIE_DECLARED_DECLINE_TRACE=1`.
//
// The latch is always on and deliberately so: it is bounded (one line
// per reason, six reasons) and it makes a class DISCOVERABLE without
// anyone having thought to arm a trace first. Every decline class this
// project has had to chase was one nobody knew was being taken.
struct DeclineShape {
    int n;
    int r;
    bool pure_decode;
    std::uint32_t max_layers;
    std::uint32_t full_depth_rows;
    std::uint32_t bands;
};

void note_decline(DeclineReason reason, const DeclineShape& shape) {
    if (reason == DeclineReason::None) return;
    static const bool trace = [] {
        const char* v = std::getenv("PIE_DECLARED_DECLINE_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    static std::atomic<bool> latched[16] = {};
    const std::size_t i = static_cast<std::size_t>(reason);
    if (!latched[i].exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[declared] fires are falling back to the hand-written "
                     "body: reason=%s (first sighting; "
                     "PIE_DECLARED_DECLINE_TRACE=1 counts them)\n",
                     decline_name(reason));
    } else if (trace) {
        // The SHAPE too, because "which rule refused" is only half of a
        // work list: closing a rule means knowing which of its shapes
        // actually arrive. A uniformly truncated prefill and a mixed one
        // decline through the same term and are different pieces of
        // work.
        std::fprintf(stderr,
                     "[declared] decline reason=%s N=%d R=%d decode=%d "
                     "k=%d full_rows=%d bands=%u\n",
                     decline_name(reason), shape.n, shape.r,
                     shape.pure_decode ? 1 : 0,
                     shape.max_layers == 0xffffffffu
                         ? -1 : static_cast<int>(shape.max_layers),
                     shape.full_depth_rows == 0xffffffffu
                         ? -1 : static_cast<int>(shape.full_depth_rows),
                     shape.bands);
    }
}

}  // namespace

LlamaLikeModel::LlamaLikeModel(
    Qwen3Weights weights,
    const HfConfig& hf_config,
    KvCache& kv_cache,
    const LlamaLikeForwardCfg& fwd_cfg)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      kv_cache_(kv_cache),
      fwd_cfg_(fwd_cfg)
{
    // Llama-like decode is graph-replay-safe because (a) the body is
    // host-work-free (the prepare hook hoisted DecodePlan out of the
    // capture region); (b) flashinfer's plan_info layout is pinned across
    // fires when enable_cuda_graph=true — padded_batch_size =
    // max_grid_size / gdy (stable), and the int_buf offsets are
    // deterministic from that. Quantized KV currently dequantizes active
    // physical pages into a BF16 scratch cache before FlashInfer; that
    // dequant launch shape depends on the live page count, while decode
    // graph keys only bucket request count/layout — replay would leave
    // newly-active pages stale, so we gate graph_safe on native BF16.
    caps_.graph_safe = kv_cache_.format().is_native_bf16();
    caps_.graph_padding_kv_write_safe = true;
    caps_.supports_compact_logits = true;
    caps_.supports_runtime_window = true;
    // Stage 6 increment 4: hook-carrying pure-decode fires may be captured.
    // The body's hook-adjacent work under capture is stream ops against
    // stable sideband-arena addresses (LayerScoreCapture on the plain decode
    // branch); the prefill capture and the page-mask path are excluded by
    // the batch-engine gate (decode-only + wants_page_mask == false). Same
    // BF16 gate as graph_safe: the dequant scratch path is not replayable.
    caps_.supports_hook_graph_capture = caps_.graph_safe;

    // Trace the declared plan now rather than on first fire: the facts it
    // needs (config + weight bindings) are all here, and an unrepresentable
    // config yields an empty plan, which `body` treats as "hand-written
    // path only" — never an error.
    if (declared_forward_enabled()) {
        declared_ = build_llama_like_declared_plan(
            hf_config_, fwd_cfg_, weights_, kv_cache_);
    }
    // The unionized supergraph (S3): capability = an emitted build exists
    // for exactly this deployment's digest (and the generated gate is on).
    caps_.supports_supergraph =
        static_cast<bool>(declared_) &&
        llama_like_supergraph_supported(declared_);
    // `body()` below forwards the slab width, and `llama_like_forward_paged`
    // acts on it -- but only a dense BF16 head can be reduced slab by slab, so
    // the weight decides too. Asked once here rather than inside the forward:
    // by then the graph key and the epilogue's token source are committed.
    caps_.supports_fused_lm_head_argmax =
        weights_.lm_head != nullptr &&
        ops::lm_head_argmax_supported(*weights_.lm_head);
    // force_prefill decode rides the plan-free BatchPrefill dispatch whose
    // capture-time launch config binds the synthetic lattice shape — those
    // pre-captured buckets replay pathologically (see the capability's
    // declaration). Everything else keeps the boot lattice.
    caps_.upfront_capture_safe = !fwd_cfg_.force_prefill_path;
}

void LlamaLikeModel::prepare(AttentionWorkspace& attn_ws,
                             const ForwardFn::PrepareInputs& in) {
    LlamaLikeForwardCfg runtime_cfg = fwd_cfg_;
    if (in.runtime_window_left >= -1) {
        runtime_cfg.sliding_window = in.runtime_window_left;
        runtime_cfg.per_layer_window_left.clear();
        if (in.runtime_window_left >= 0) {
            runtime_cfg.use_xqa_decode = false;
        }
    }
    prepare_llama_like_decode_plan(
        plan_, attn_ws, kv_cache_, hf_config_, runtime_cfg,
        in.qo_indptr_h,
        in.kv_page_indices_d,
        in.kv_page_indptr_h,
        in.kv_page_indptr_d,
        in.kv_last_page_lens_h,
        in.kv_last_page_lens_d,
        in.total_tokens,
        in.num_requests,
        in.is_pure_decode,
        in.have_custom_mask,
        in.attn_score_window,
        in.unmasked_prefix_rows,
        in.mask_suffix_page_counts_h,
        in.mask_suffix_last_lens_h,
        in.full_depth_rows,
        in.depth_band_k, in.depth_band_rows, in.depth_band_count);
}

void LlamaLikeModel::body(Workspace& ws,
                          KvCache& kv,
                          AttentionWorkspace& attn_ws,
                          ops::CublasHandle& cublas,
                          const ForwardFn::ForwardInputs& in) {
    // The declared executor covers the hand-written path's vocabulary;
    // anything it cannot express falls back, per fire, to the
    // hand-written body below. Build-time exclusions (TP, quantized
    // projections, non-standard rope, ...) already left `declared_` empty.
    // Hooked fires (A3, the Peel op): EVERY hook composition —
    // all-hooked, mixed (0 < fast_rows < R), none, and masked+hooked
    // (the mask arm carries the sites; the custom dispatch publishes no
    // scores, which is the hand-written contract) — walks the shape
    // trace.
    //
    // Written as a REASON rather than a conjunction (cutover step 4(b)).
    // The flip made "which fires still fall back" the whole remaining
    // question, and a boolean cannot answer it: the measurement it
    // forced instead — pairing `[fire]` and `[flat]` lines out of one
    // stderr written by two threads — is not sound, and a number nobody
    // can attribute is not a work list. Naming the term that refused
    // costs one enum and makes the list fall out of a `grep -c`.
    //
    // The reasons are split to the CAUSE, not to the term. Both depth
    // terms refuse for two unrelated reasons — a prefill fire, or a
    // decode fire missing something — and those have different owners
    // (`family.rs` withholding the axis, the prepare not stamping a
    // band plan) and different amounts of work. A work list that merges
    // them reads as two items when it is four, and whoever picks one up
    // finds out which after they start.
    const DeclineReason decline = [&]() -> DeclineReason {
        if (!static_cast<bool>(declared_)) return DeclineReason::NoPlan;
        // Explicit KV-write fires are in scope (declared_forward.hpp says
        // why: every graph-replayed decode fire carries them), but only
        // when the descriptors actually arrived — the same guard the
        // hand-written fused predicate applies.
        if (in.has_write_desc &&
            (in.w_page_d == nullptr || in.w_off_d == nullptr)) {
            return DeclineReason::WriteDescMissing;
        }
        if (in.runtime_window_left != -2) {
            return DeclineReason::SlidingWindow;
        }
        // STRUCTURAL S-4: truncated fires walk the declared trace when
        // the DECLARATION states the depth axis for the fire's shape
        // (pure-decode only; a truncated lane's prefill keeps the
        // hand-written body).
        // Phi-3-mini pads head_dim (96 -> 128), and the padded staging is
        // what makes a ROW WINDOW unserveable there: the buffers are laid
        // out at physical width while a window addresses at logical.
        const bool padded_head =
            hf_config_.head_dim != hf_config_.head_dim_kernel;
        if (in.max_layers != 0xffffffffu) {
            // The fire's OWN class states the axis or it does not — ask
            // the plan this fire will actually run, not the decode one.
            const auto& fire_plan =
                in.is_pure_decode ? declared_.decode : declared_.prefill;
            if (!fire_plan || fire_plan.view().depth_window == 0) {
                return DeclineReason::TruncatedAxisUnstated;
            }
            if (padded_head && in.full_depth_rows != 0xffffffffu) {
                return DeclineReason::PaddedHeadNarrowing;
            }
            // A truncated PREFILL is admitted only when the truncation
            // is UNIFORM. That is the cheap half of the axis: every row
            // sits at the same `k`, so the window stops after layer `k`
            // and narrows nothing. A UNION fire puts full-depth rows
            // beside truncated ones, and running the tail layers over a
            // row prefix means narrowing this fire's qo/kv CSRs with
            // them — there is no prefill analogue of
            // `depth_prefix_decode_plan` to narrow them against.
            if (!in.is_pure_decode &&
                in.full_depth_rows != 0xffffffffu) {
                return DeclineReason::UnionPrefill;
            }
        }
        // The trace committed to the fused QKV binding; a workspace without
        // the packed buffer cannot honour it (same availability check the
        // hand-written `use_fused_qkv` makes).
        if (declared_.fused_qkv && ws.qkv_fused.empty()) {
            return DeclineReason::FusedQkvUnstaged;
        }
        // ④ Act 1 (banded depth): the interpreter serves banded fires
        // when the DECODE-family band plans exist (every live band has
        // its prefix plan); prefill-family banded deployments keep the
        // hand-written body. Without any term here the declared leg
        // silently demoted mixed-k fires to full depth (caught live at
        // 14B: R=8 co-fires, no [depth-bands], no DECLINE).
        // The bands, asked of the ONE function the executor asks
        // (`llama_like_bands_apply`). This used to be a second copy of
        // the executor's condition, and the copy went stale one commit
        // after it was written — the argument it carried ("a prefill
        // trace does not state the depth axis, so the bands are never
        // read") was invalidated by the next commit teaching Prefill to
        // state it, and Qwen2.5-1.5B threw 5,080 times. A gate that
        // mirrors an executor is a duplicate, not a proof; this asks.
        //
        // What is left to decide here is the thing the executor cannot:
        // it THROWS on a stamped band with no usable prefix plan,
        // because by then the fire is committed. The gate turns that
        // into a fallback while there is still a choice.
        if (llama_like_bands_apply(declared_, plan_, in.is_pure_decode)) {
            // A band IS a row narrowing, so it meets the padded-head wall
            // for the same reason a union does.
            if (padded_head) return DeclineReason::PaddedHeadNarrowing;
            for (std::uint32_t j = 0; j < plan_.depth_band_count; ++j) {
                if (plan_.depth_band_rows[j] > 0 &&
                    !plan_.depth_band_plans[j]) {
                    return DeclineReason::BandedPlanMissing;
                }
            }
        }
        return DeclineReason::None;
    }();
    note_decline(decline,
                 DeclineShape{in.total_tokens, in.num_requests,
                              in.is_pure_decode, in.max_layers,
                              in.full_depth_rows, plan_.depth_band_count});
    const bool declared_eligible = decline == DeclineReason::None;
    if (declared_eligible) {
        llama_like_forward_declared(
            declared_, weights_, hf_config_, fwd_cfg_, plan_,
            ws, kv, attn_ws, cublas,
            in.token_ids, in.positions,
            in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
            in.kv_last_page_lens_d,
            in.qo_indptr_h, in.kv_page_indptr_h,
            in.total_tokens, in.num_requests, in.is_pure_decode,
            in.logit_row_indices_d, in.num_logit_rows,
            in.w_page_d, in.w_off_d,
            in.row_valid_d, in.has_write_desc,
            in.runtime_window_left,
            in.custom_mask_d, in.custom_mask_indptr_d,
            in.stage_hooks,
            in.lora,
            in.peel_window_d,
            in.unmasked_prefix_rows,
            in.mask_suffix_qo_indptr_d,
            in.mask_suffix_kv_page_indptr_d,
            in.max_layers,
            in.full_depth_rows);
        return;
    }
    LlamaLikeForwardCfg fwd = fwd_cfg_;
    fwd.logits_argmax_chunk_tokens = in.logits_argmax_chunk_tokens;

    llama_like_forward_paged(
        weights_, hf_config_, fwd, plan_,
        ws, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.logit_row_indices_d, in.num_logit_rows,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.w_page_d, in.w_off_d, in.row_valid_d, in.has_write_desc,
        in.runtime_window_left,
        /*vision=*/nullptr,
        in.stage_hooks,
        in.lora,
        in.peel_window_d,
        in.unmasked_prefix_rows,
        in.mask_suffix_qo_indptr_d,
        in.mask_suffix_kv_page_indptr_d,
        in.max_layers,
        in.full_depth_rows);
}

std::uint32_t LlamaLikeModel::graph_layout() {
    return llama_like_decode_graph_layout(plan_);
}

std::uint32_t LlamaLikeModel::supergraph_graph_layout() {
    return llama_like_supergraph_graph_layout(plan_);
}

std::uint64_t LlamaLikeModel::lora_stage(Workspace& ws,
                                         const LoraTable* lora,
                                         int total_tokens,
                                         cudaStream_t stream) {
    return llama_like_lora_stage(
        plan_, ws, lora, hf_config_, fwd_cfg_, total_tokens, stream);
}


bool LlamaLikeModel::supergraph_body(Workspace& ws,
                                     KvCache& kv,
                                     AttentionWorkspace& attn_ws,
                                     ops::CublasHandle& cublas,
                                     const ForwardFn::ForwardInputs& in,
                                     batch::SupergraphBuilder& sg) {
    // The declared gate's terms, restated (the capture calls this
    // directly, bypassing body()): descriptors present when promised,
    // config-window only, fused staging available when the trace
    // committed to it. Hooks and lora are outside the union by
    // eligibility; a caller passing them is a drift.
    if (!static_cast<bool>(declared_)) return false;
    if (in.stage_hooks != nullptr || in.lora != nullptr) return false;
    if (in.has_write_desc &&
        (in.w_page_d == nullptr || in.w_off_d == nullptr)) {
        return false;
    }
    if (in.runtime_window_left != -2) return false;
    if (declared_.fused_qkv && ws.qkv_fused.empty()) return false;
    return llama_like_forward_supergraph_build(
        declared_, weights_, hf_config_, fwd_cfg_, plan_,
        ws, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests,
        in.logit_row_indices_d, in.num_logit_rows,
        in.w_page_d, in.w_off_d,
        in.row_valid_d, in.has_write_desc,
        in.custom_mask_d, in.custom_mask_indptr_d,
        sg);
}

bool LlamaLikeModel::prefill_graph_capturable() const {
    return llama_like_prefill_graph_capturable(plan_);
}

}  // namespace pie_cuda_driver::model
