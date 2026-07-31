#pragma once

// The qwen3_5 declared EXECUTOR, arc 2's decode slice — the emission arms
// that walk the arc-1 traced + validated hybrid plan
// (`declared_facts.hpp::Qwen35DeclaredPlan`) and launch EXACTLY the kernels
// `qwen3_5_forward_paged` launches on the TP=1 PURE-DECODE path, on the
// same workspace buffers, in the same order — the same bit-identity
// argument as `llama_like/declared_forward.hpp`.
//
// Scope (the honest slice; the caller's gate in qwen3_5_model.cpp encodes
// it): dense hybrid (0.8b shape), pure-decode fires only, no stage hooks,
// no lora, no spec-decode services (frozen verify, commit-advance,
// rs-buffer write/fold all fall back per fire), no state-only fires
// (num_logit_rows < 0). Prefill fires fall back to the hand-written body —
// the prefill lowerings of the conv/recurrence ops are a later arc. MTP
// draft fires never reach `body()` at all (they are the drafter service's
// own entry points, `qwen3_5_mtp_forward` / `qwen3_5_mtp_process_cache`),
// so the base pass this executor serves is the same base pass the MTP
// verify wraps — but verify itself arrives as frozen-verify /
// commit-advance fires, which the gate excludes.
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
#include "model/qwen3_5/qwen3_5.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"

namespace pie_cuda_driver::model {

// Parity-harness visibility for the qwen3_5 executor: the per-fire
// `[declared-qwen35-exec]` line (and the body-gate fallback line) are
// emitted when PIE_DECLARED_FORWARD_TRACE is set — the same env the
// llama_like executor's `[declared-forward]` line reads, so one flag
// lights up both families.
bool qwen35_declared_exec_trace_enabled();

// Execute one eligible fire by walking `declared.plan`. The caller
// (Qwen35Model::body) has already applied the eligibility gate; this
// function additionally throws (never silently diverges) when the plan
// carries an op or payload outside the decode slice's vocabulary — a trace
// whose shape drifted must fail loudly, exactly the llama_like executor's
// contract.
void qwen3_5_forward_declared(
    const Qwen35DeclaredPlan& declared,
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const Qwen3_5PlanState& plan_state,
    Workspace& ws,
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
    int num_logit_rows);

}  // namespace pie_cuda_driver::model
