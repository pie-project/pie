#include "model/llama_like/llama_like_model.hpp"

#include "model/stage_hooks.hpp"
#include "ops/gemm.hpp"

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
    const bool declared_eligible =
        static_cast<bool>(declared_) &&
        // Explicit KV-write fires are in scope (declared_forward.hpp says
        // why: every graph-replayed decode fire carries them), but only
        // when the descriptors actually arrived — the same guard the
        // hand-written fused predicate applies.
        (!in.has_write_desc ||
         (in.w_page_d != nullptr && in.w_off_d != nullptr)) &&
        in.runtime_window_left == -2 &&

        // STRUCTURAL S-4: truncated fires walk the declared trace when
        // the DECLARATION states the depth axis for the fire's shape
        // (pure-decode only; a truncated lane's prefill keeps the
        // hand-written body).
        (in.max_layers == 0xffffffffu ||
         (in.is_pure_decode && declared_.decode &&
          declared_.decode.view().depth_window != 0)) &&
        // The trace committed to the fused QKV binding; a workspace without
        // the packed buffer cannot honour it (same availability check the
        // hand-written `use_fused_qkv` makes).
        (!declared_.fused_qkv || !ws.qkv_fused.empty()) &&
        // ④ Act 1 (banded depth): the interpreter serves banded fires
        // when the DECODE-family band plans exist (every live band has
        // its prefix plan); prefill-family banded deployments keep the
        // hand-written body. Without any term here the declared leg
        // silently demoted mixed-k fires to full depth (caught live at
        // 14B: R=8 co-fires, no [depth-bands], no DECLINE).
        (plan_.depth_band_count < 2 ||
         (in.is_pure_decode && [&] {
             for (std::uint32_t j = 0; j < plan_.depth_band_count; ++j) {
                 if (plan_.depth_band_rows[j] > 0 &&
                     !plan_.depth_band_plans[j]) {
                     return false;
                 }
             }
             return true;
         }()));
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

}  // namespace pie_cuda_driver::model
