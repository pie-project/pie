#pragma once

// gemma-4's DECLARED forward: the plan built at load, and the executor
// that drives its flat launch list.
//
// The third family on tart. Unlike llama_like and qwen3_5 this one was
// never a walk — it is a lowered drive from its first line, because the
// declaration was complete before any executor existed (the trace lowers
// with an empty residue; `lower.rs::the_gemma4_residue_ledger`).

#include <string>

#include "model/config.hpp"
#include "model/gemma4/gemma4.hpp"
#include "pie_forward/plan.hpp"

namespace pie_cuda_driver::model {

// The plan a deployment holds, or an empty one with the reason it
// declined. Mirrors `Qwen35DeclaredPlan`'s contract: refusal is a
// FALLBACK, not an error — the hand-written pass is untouched either
// way.
struct Gemma4DeclaredPlan {
    pie_forward::ForwardPlan decode;
    pie_forward::ForwardPlan prefill;
    std::string facts_digest;
    bool usable = false;
};

// `PIE_DECLARED_FORWARD`, llama_like's polarity: default ON, `=0`
// disarms onto the hand-written pass.
bool gemma4_declared_forward_enabled();

// Derive this deployment's facts and trace both classes.
Gemma4DeclaredPlan build_gemma4_declared_plan(
    const HfConfig& cfg, const Gemma4Weights& w, int tp_size);

// Every kernel the plan states must be in this executor's registry. A
// symbol outside it means the trace and the driver drifted, and saying
// so at LOAD is what keeps a drift from becoming a wrong number.
void gemma4_validate_stated_kernels(const pie_forward::ForwardPlan& plan);

}  // namespace pie_cuda_driver::model
