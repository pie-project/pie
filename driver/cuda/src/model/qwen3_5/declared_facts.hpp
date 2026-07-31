#pragma once

// Live facts extraction + boot-time trace validation for the qwen3_5 hybrid
// — arc 1 of the declared-qwen3_5 EXECUTOR (the honest list in commit
// 9c54b9b6: "live facts extraction" is the first slice; the twelve emission
// arms, workspace binder, plan_state/RS-cache plumbing etc. are later arcs).
//
// What this does at model construction, when PIE_DECLARED_FORWARD opted in:
// build `PieForwardQwen35HybridFacts` from the HfConfig + the model's ACTUAL
// bindings (the same evidence the hand-written forward reads — layer_types
// reduced to the full-attention interval, the driver's own rotary_dim
// derivation, the fused-projection binding pointers the env gates
// populated), trace the hybrid through the ABI wrapper, then run a
// STRUCTURAL validation of the traced form against the real config:
//
//   * total op count matches the formula for the config's layer schedule;
//   * the per-layer op-kind sequence matches the interval (full-attention
//     vs GDN layer bodies, dense vs MoE MLP);
//   * every weight name in the plan resolves against the bound weight set —
//     a name-resolution dry walk, the executor arc's weight resolver in
//     embryo (the first unresolvable name is logged loudly).
//
// One line is logged either way:
//   [declared-qwen35] traced ops=.. layers=.. interval=.. validation=OK
//   [declared-qwen35] traced ops=.. layers=.. interval=.. validation=refused(reason)
//
// NOTHING consumes the plan in body() yet — no execution, cold-start
// validation only. A config the trace cannot express (irregular
// layer_types schedule, quantized projections, TP>1, a mixed
// fused/unfused binding) yields an EMPTY plan with the reason logged once;
// never an error — the hand-written path is untouched either way, exactly
// `build_llama_like_declared_plan`'s empty-on-unrepresentable contract.
// (MTP is NOT a refusal: it is a per-fire drafter service around the base
// pass — family.rs's hybrid doc states it is not an op of the pass — and
// body() runs the base pass regardless, which is what the trace states.)

#include "model/config.hpp"
#include "model/qwen3_5/qwen3_5.hpp"
#include "model/qwen3_5/qwen3_5_moe.hpp"
#include "pie_forward/plan.hpp"

namespace pie_cuda_driver::model {

// Stage 3's opt-in gate, shared spelling with llama_like_model.cpp's
// `declared_forward_enabled`: PIE_DECLARED_FORWARD non-empty and not "0".
bool qwen35_declared_forward_enabled();

// The traced + validated form, held by the model for arc 2 (the emission
// arcs). Empty (operator bool false) when the configuration is outside the
// trace's vocabulary or the validation refused — the model's body() keeps
// the hand-written path either way.
struct Qwen35DeclaredPlan {
    pie_forward::ForwardPlan plan;
    // The binding facts the trace committed to (llama_like's `fused_qkv`
    // precedent); arc 2's per-fire gate re-checks them against the live
    // workspace before emitting.
    bool fused_gdn_in_proj = false;
    bool fused_full_attn_qgkv = false;
    // The layer-kind schedule as the reduced interval (the Metal
    // geometry's `is_full_attn` formula) — recorded so arc 2 need not
    // re-derive it from cfg.layer_types.
    int full_attn_interval = 0;

    explicit operator bool() const noexcept {
        return static_cast<bool>(plan);
    }
};

// Build, trace and validate against this deployment's facts, logging the
// one [declared-qwen35] line. Overloaded per weights struct because the
// dense and MoE model classes bind different layer-weight sets (and the
// MLP fact differs); the extraction and validation core is shared.
Qwen35DeclaredPlan build_qwen3_5_declared_plan(
    const HfConfig& cfg, const Qwen3_5Weights& w, int tp_size);
Qwen35DeclaredPlan build_qwen3_5_declared_plan(
    const HfConfig& cfg, const Qwen3_5MoeWeights& w, int tp_size);

}  // namespace pie_cuda_driver::model
