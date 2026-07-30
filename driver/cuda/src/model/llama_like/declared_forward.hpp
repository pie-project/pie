#pragma once

// Declared-forward executor for the llama_like family (Stage 3,
// pie-application-plan.md §7): walk the traced form the `forward/` crate
// produced at model load and launch the SAME kernels, on the SAME workspace
// buffers, that `llama_like_forward_paged`'s unfused path launches. Nothing
// here generates a kernel — op→kernel selection is this driver's knowledge,
// exactly as `contract.hpp` states for the loader's specs.
//
// Scope: the hand-written path's full kernel vocabulary, reached through
// two emitter peepholes over the deliberately-unfused trace:
//   * RmsnormPerHead x2 + Rope → `launch_qk_rmsnorm_rope_bf16`, the
//     hand-written `fuse_qk_norm_rope` branch;
//   * Matmul(qkv) + SplitQkv + RmsnormPerHead x2 + Rope + KvAppend →
//     `launch_qkv_decode_qk_norm_rope_write_kv_bf16`, the hand-written
//     `fused_decode_qkv_post` branch, under the SAME predicate (including
//     the PIE_CUDA_DECODE_FUSED_POST gate).
// Bit-parity requires the same launches, not just the same math — the
// fused kernels round differently from their unfused sequences.
// Everything the trace cannot express yet (hooks, custom masks, TP,
// vision, quantized projections, non-standard rope, post-norm, qkv bias)
// falls back to `llama_like_forward_paged` — the caller gates, `build`
// refuses. Padded head_dim (Phi-3-mini's 96 → 128) is IN scope: the
// pad/strip staging around KV-write/attention is emitter knowledge, not
// trace vocabulary — the trace speaks the logical head_dim throughout.
//
// Explicit KV-write descriptors ARE handled (the hand-written
// `has_write_desc` branch, verbatim): every pure-decode fire that replays a
// forward graph carries them — `forward_graph_replay_eligible` REQUIRES
// `has_write_desc` (batch/forward.cpp) because pure-decode captures record
// the w_page/w_off write path — so excluding them would exclude decode
// entirely and reduce Stage 3's parity claim to the prefill step.

#include <cstdint>

#include "model/llama_like/llama_like.hpp"
#include "pie_forward/plan.hpp"

namespace pie_cuda_driver::model {

// The traced form plus what the executor needs to know about how it was
// traced. Built once at model construction (the facts are load-time facts;
// re-tracing per fire would contradict the trace's whole premise).
struct LlamaLikeDeclaredPlan {
    pie_forward::ForwardPlan plan;
    // The binding fact the trace was taken against; the per-fire gate
    // re-checks it against the workspace (`ws.qkv_fused` may be empty even
    // when the weight is bound) and falls back on mismatch.
    bool fused_qkv = false;

    explicit operator bool() const noexcept {
        return static_cast<bool>(plan);
    }
};

// Trace the family against this deployment's facts. Returns an empty plan
// (operator bool false) when the configuration is outside the v0 trace's
// vocabulary — the caller then keeps the hand-written path, silently: an
// unrepresentable config is a fallback, not an error.
LlamaLikeDeclaredPlan build_llama_like_declared_plan(
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const Qwen3Weights& w);

// Execute the traced form. Same argument surface as
// `llama_like_forward_paged` minus the inputs the eligibility gate already
// excluded (hooks, custom mask, write descriptor, vision). Reads the SAME
// `plan_state` the prepare hook filled — prepare() is unchanged and runs
// for both paths.
void llama_like_forward_declared(
    const LlamaLikeDeclaredPlan& declared,
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    int runtime_window_left);

}  // namespace pie_cuda_driver::model
