#pragma once

// gpt-oss's DECLARED forward: the plan built at load, and the executor
// that drives its flat launch list.
//
// The fourth family on tart, and the first whose MoE block is stated end
// to end. gpt-oss has no forward of its own — it rides
// `mixtral_forward_paged` — so this drive mirrors that pass at tp=1 and
// declines everything else.

#include <string>

#include "model/config.hpp"
#include "model/llama_like/llama_like.hpp"
#include "model/mixtral/mixtral.hpp"
#include "pie_forward/plan.hpp"

namespace pie_cuda_driver::model {

// The plan a deployment holds, or an empty one with the reason it
// declined. Refusal is a FALLBACK, not an error — the hand-written pass
// is untouched either way.
struct GptOssDeclaredPlan {
    pie_forward::ForwardPlan decode;
    pie_forward::ForwardPlan prefill;
    std::string facts_digest;
    bool usable = false;
    // The fused leg's admission threshold in ROUTES, carried so the
    // drive can ask the same question the hand pass asks rather than
    // restating it.
    int max_routes = 0;
};

// `PIE_DECLARED_FORWARD`, the tree-wide gate: default ON, `=0` disarms.
bool gpt_oss_declared_forward_enabled();

// `PIE_DECLARED_FORWARD_GPT_OSS` — whether the drive EXECUTES. Opt-in
// while the arms are new.
bool gpt_oss_declared_drive_enabled();

// Derive this deployment's facts and trace the decode class.
GptOssDeclaredPlan build_gpt_oss_declared_plan(
    const HfConfig& cfg,
    const MixtralWeights& w,
    const LlamaLikeForwardCfg& fwd_cfg,
    int num_experts,
    int top_k);

// Drive the decode class's flat launch list. Returns false when this
// fire is outside what the declaration states, leaving the hand-written
// pass to run it — the eligibility answer, not an error.
bool gpt_oss_forward_declared(
    const GptOssDeclaredPlan& declared,
    const MixtralWeights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    int num_experts,
    int top_k,
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
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows);

// Every kernel the plan states must be in this executor's registry. A
// symbol outside it means the trace and the driver drifted, and saying
// so at LOAD is what keeps a drift from becoming a wrong number.
void gpt_oss_validate_stated_kernels(const pie_forward::ForwardPlan& plan);

// Every weight the plan NAMES must resolve against the bound set — empty
// string when it does, else the reason. A plan that fails this declines;
// an unbound weight found at the first fire fails the model load.
std::string gpt_oss_validate_stated_weights(
    const pie_forward::ForwardPlan& plan, const MixtralWeights& w);

}  // namespace pie_cuda_driver::model
