#pragma once

// The qwen3_5 declared EXECUTOR — the emission arms that walk the arc-1
// traced + validated hybrid plan (`declared_facts.hpp::Qwen35DeclaredPlan`)
// and launch EXACTLY the kernels `qwen3_5_forward_paged` launches on the
// TP=1 path, on the same workspace buffers, in the same order — the same
// bit-identity argument as `llama_like/declared_forward.hpp`. Arc 2 covered
// pure-decode fires; arc 3 added the PREFILL lowerings — the trace itself
// is decode/prefill-agnostic (crates/model-compiler/src/trace.rs: CausalConv1d /
// GatedDelta / Attention are opaque ops whose lowering the emitter picks
// per fire), so the same 16 arms now branch per fire exactly as the
// hand-written bodies branch: conv decode-update vs prefill walk, the
// recurrence step vs the chunked prefill family (warp-tiled / cached /
// batched GQA-aware FLA, selected by the shared env knobs in
// qwen3_5_forward.hpp), decode vs prefill attention plans from plan_state.
// A MIXED fire (prefill + decode rows co-batched) is not separate
// machinery: the hand-written body runs every `is_pure_decode == false`
// fire as one qo_indptr-windowed prefill shape, and the walk mirrors that.
//
// Scope (the honest slice; the caller's gate in qwen3_5_model.cpp encodes
// it): dense hybrid (0.8b shape), no stage hooks, no lora, no custom
// masks, no rs-buffer write/fold fires (excluded by DESIGN per the Stage-2
// verdict — see the gate comment). The MTP-adjacent fire shapes are
// DECLARED since this arc: state-only fires (num_logit_rows < 0) skip the
// final-norm / lm_head epilogue arms exactly as the hand-written body
// returns early; frozen-verify fires (state_cache.verify_frozen()) run the
// full pass with write_state=false plus the per-layer in-proj stash-write
// memcpys; commit-advance fires (commit_lens != nullptr) walk only the
// linear layers' [stash-load, conv, prep, recurrence] launches with
// commit_lens threaded into the batched conv prefill and the FLA
// recurrence — each mirrored launch-for-launch from
// `linear_attn_layer_body` / `qwen3_5_forward_paged`. MTP draft fires
// never reach `body()` at all (they are the drafter service's own entry
// points, `qwen3_5_mtp_forward` / `qwen3_5_mtp_process_cache`); the
// verify-window fires MTP inferlets route through body() are plain
// prefill-shaped fires (arc 3), and the frozen/commit/state-only services
// above currently have no runtime producer — they are mirrored so the
// declared walk serves the same contract the hand-written body serves,
// not because a caller exists today.
//
// Peepholes: NONE. The hand-written qwen3_5 decode path launches one
// kernel per trace op (family.rs's table is launch-for-launch); the only
// trace-op-to-many-launches spots are emitter LOWERINGS handled inside
// their arms (the GQA repeat_interleave materialisation inside GdnPrep,
// the head-dim-free KV/attention plan choice inside Attention), and the
// only many-ops-to-one-launch candidates (fused in_proj / qgkv banks) are
// binding FACTS the trace already resolved at build time. Contrast
// llama_like's fused decode-QKV peephole, which qwen3_5 has no analog of.

#include <cstdint>

#include "model/qwen3_5/declared_facts.hpp"
#include "model/qwen3_5/qwen3_5_moe.hpp"

// Forward-declared, not included: the executor only takes a pointer to
// it, and pulling in the MoE forward header would make every dense
// translation unit depend on the MoE block.
namespace pie_cuda_driver::model { struct Qwen3_5MoeMlpWorkspace; }
#include "model/qwen3_5/qwen3_5.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"
#include "model/stage_hooks.hpp"

namespace pie_cuda_driver::model {

// Parity-harness visibility for the qwen3_5 executor: the per-fire
// `[declared-qwen35-exec]` line (and the body-gate fallback line) are
// emitted when PIE_DECLARED_FORWARD_TRACE is set — the same env the
// llama_like executor's `[declared-forward]` line reads, so one flag
// lights up both families.
bool qwen35_declared_exec_trace_enabled();

// `PIE_DECLARED_MOE`. Default OFF, unlike the dense arc: the MoE half of
// the executor is newer, and an unset gate should keep every existing MoE
// deployment on the hand-written body it has always run.
bool qwen35_declared_moe_enabled();

// Boot validation (rung 4c-iii): every Launch symbol a class trace
// states must resolve in this executor's name→launcher registry, so a
// declaration/executor drift fails at model load, not mid-fire.
void qwen35_validate_stated_kernels(const pie_forward::ForwardPlan& plan);

// Execute one eligible fire by walking its CLASS trace (rung 5: the
// semantic walk is deleted from this executor). Returns false when the
// fire has no class (legacy harness shapes, live-fact mismatches) — the
// caller then runs the hand-written path. The caller
// (Qwen35Model::body) has already applied the eligibility gate; this
// function additionally throws (never silently diverges) when the plan
// carries an op or payload outside the executor's vocabulary — a trace
// whose shape drifted must fail loudly, exactly the llama_like executor's
// contract.
bool qwen3_5_forward_declared(
    const Qwen35DeclaredPlan& declared,
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const Qwen3_5PlanState& plan_state,
    Workspace& ws,
    Qwen3_5MoeMlpWorkspace* moe_ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    RecurrentStateCache& state_cache,
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
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::uint8_t* is_fresh_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    // Recurrent-only commit-advance (spec-decode repair): non-null device
    // [R] confirmed-prefix lengths — the hand-written `commit_len`
    // threading (`in.commit_advance_gather_d`; the rs_buffer_fold flavor
    // stays gate-excluded, so this is always the verify-stash replay).
    const std::int32_t* commit_lens,
    // A4: the fire's attached stage-hook programs (null = none). The
    // class traces carry the HookSite ops; qwen3_5's sites are
    // observation-only, so nothing else crosses.
    const StageHooks* stage_hooks);

// The same executor over the MoE weights. Two overloads rather than one
// generic parameter: the caller knows which family it is, and the template
// that serves both lives in the .cpp.
bool qwen3_5_forward_declared(
    const Qwen35DeclaredPlan& declared,
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const Qwen3_5PlanState& plan_state,
    Workspace& ws,
    Qwen3_5MoeMlpWorkspace* moe_ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    RecurrentStateCache& state_cache,
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
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::uint8_t* is_fresh_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    // Recurrent-only commit-advance (spec-decode repair): non-null device
    // [R] confirmed-prefix lengths — the hand-written `commit_len`
    // threading (`in.commit_advance_gather_d`; the rs_buffer_fold flavor
    // stays gate-excluded, so this is always the verify-stash replay).
    const std::int32_t* commit_lens,
    // A4: the fire's attached stage-hook programs (null = none). The
    // class traces carry the HookSite ops; qwen3_5's sites are
    // observation-only, so nothing else crosses.
    const StageHooks* stage_hooks);

}  // namespace pie_cuda_driver::model
