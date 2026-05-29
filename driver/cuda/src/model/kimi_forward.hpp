#pragma once

#include <cstdint>

#include "attention_workspace.hpp"
#include "distributed.hpp"
#include "kv_cache.hpp"
#include "mla_cache.hpp"
#include "model/kimi.hpp"
#include "ops/attention_mla.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct KimiForwardCfg {
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
    bool emit_logits = true;
    bool tp_greedy_argmax = false;
    void* greedy_pairs = nullptr;
    void* greedy_pairs_all = nullptr;
};

struct KimiWorkspace {
    DeviceTensor y;                 // [N, H]
    DeviceTensor norm_x;            // [N, H]
    DeviceTensor q_a;               // [N, q_lora_rank]
    DeviceTensor q_b;               // [N, local_heads*(qk_nope+qk_rope)]
    DeviceTensor q_nope;            // [N, local_heads*qk_nope]
    DeviceTensor kv_a_mqa;          // [N, kv_lora_rank+qk_rope]
    DeviceTensor kv_c;              // [N, kv_lora_rank]
    DeviceTensor k_pe;              // [N, qk_rope]
    DeviceTensor q_nope_latent;     // [N, local_heads*kv_lora_rank]
    DeviceTensor q_pe;              // [N, local_heads*qk_rope]
    DeviceTensor attn_latent;       // [N, local_heads*kv_lora_rank]
    DeviceTensor attn_v;            // [N, local_heads*v_head_dim]
    DeviceTensor attn_out;          // [N, H]
    DeviceTensor norm_y;            // [N, H]
    DeviceTensor gate;              // [N, max(local_I, routed_I)]
    DeviceTensor up;                // [N, max(local_I, routed_I)]
    DeviceTensor expert_gate_w;     // [routed_I, H] bf16 dequant scratch
    DeviceTensor expert_up_w;       // [routed_I, H] bf16 dequant scratch
    DeviceTensor expert_down_w;     // [H, routed_I] bf16 dequant scratch
    DeviceTensor router_logits;     // [N, num_experts]
    DeviceTensor topk_idx;          // [N, top_k] int32
    DeviceTensor topk_weights;      // [N, top_k] fp32
    DeviceTensor route_idx;         // [N*top_k] int32
    DeviceTensor route_w;           // [N*top_k] fp32
    DeviceTensor expert_in;         // [N*top_k, H]
    DeviceTensor expert_gate;       // [N*top_k, routed_I]
    DeviceTensor expert_up;         // [N*top_k, routed_I]
    DeviceTensor expert_out;        // [N*top_k, H]
    DeviceTensor moe_out;           // [N, H]
    DeviceTensor shared_gate;       // [N, shared_I]
    DeviceTensor shared_up;         // [N, shared_I]
    DeviceTensor shared_act;        // [N, shared_I]
    DeviceTensor shared_out;        // [N, H]
    DeviceTensor logits;            // [O, vocab]
    DeviceTensor probs;             // [O, vocab]

    static KimiWorkspace allocate(
        const HfConfig& cfg,
        int max_tokens,
        int max_logit_rows,
        int tp_size);
};

struct KimiPlanState {
    ops::MlaPlanCachePtr mla_plan;
};

void prepare_kimi_mla_plan(
    KimiPlanState& state,
    AttentionWorkspace& attn_ws,
    const MlaCache& cache,
    const HfConfig& cfg,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_h,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    bool causal,
    int tp_size);

void kimi_forward_paged(
    const KimiWeights& w,
    const HfConfig& cfg,
    const KimiForwardCfg& fwd_cfg,
    const KimiPlanState& plan_state,
    KimiWorkspace& kimi_ws,
    MlaCache& mla_cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    void* logits_out,
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
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0);

}  // namespace pie_cuda_driver::model
