#pragma once

#include <cstdint>

#include "attention_workspace.hpp"
#include "distributed.hpp"
#include "kv_cache.hpp"
#include "model/deepseek_v4.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct DsV4ForwardCfg {
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
    bool emit_logits = true;
    bool tp_greedy_argmax = false;
    void* greedy_pairs = nullptr;
    void* greedy_pairs_all = nullptr;
};

struct DsV4Workspace {
    // Multi-stream HC residual
    DeviceTensor hc_residual;    // [N, hc_mult, H] BF16 — main state
    DeviceTensor y;              // [N, H] — single-stream scratch
    DeviceTensor norm_x;         // [N, H]
    DeviceTensor norm_y;         // [N, H]

    // HC mixing scratch
    DeviceTensor hc_mixes_f32;   // [N, hc_mult*H] F32 — RMSNorm'd residual for GEMM
    DeviceTensor hc_gemm_out;    // [N, mix_hc] F32 — GEMM output
    DeviceTensor hc_post_mix;    // [N, hc_mult] F32
    DeviceTensor hc_comb_mix;    // [N, hc_mult, hc_mult] F32
    DeviceTensor hc_head_gemm;   // [N, hc_mult] F32 — head GEMM output

    // Attention Q/KV path
    DeviceTensor q_a;            // [N, q_lora_rank]
    DeviceTensor q;              // [N, num_heads * head_dim]
    DeviceTensor kv;             // [N, head_dim]  (single KV head)
    DeviceTensor q_rope;         // [N, num_heads * qk_rope_head_dim]
    DeviceTensor k_rope;         // [N, 1 * qk_rope_head_dim]
    DeviceTensor attn_out;       // [N, num_heads * head_dim]

    // Output projection
    DeviceTensor wo_a_out;       // [N, o_groups * o_lora_rank]
    DeviceTensor wo_b_out;       // [N, H]

    // MoE
    DeviceTensor router_logits;  // [N, E]
    DeviceTensor topk_idx;       // [N, K] int32
    DeviceTensor topk_weights;   // [N, K] fp32
    DeviceTensor moe_out;        // [N, H]

    // Shared expert
    DeviceTensor shared_gate;    // [N, moe_I]
    DeviceTensor shared_up;      // [N, moe_I]
    DeviceTensor shared_act;     // [N, moe_I]
    DeviceTensor shared_out;     // [N, H]

    // Logits
    DeviceTensor logits;         // [O, vocab]

    static DsV4Workspace allocate(
        const HfConfig& cfg,
        int max_tokens,
        int max_logit_rows,
        int tp_size);
};

struct DsV4PlanState {
    // FlashInfer plan for SWA layers
    void* flashinfer_plan = nullptr;
};

void dsv4_forward_paged(
    const DsV4Weights& w,
    const HfConfig& cfg,
    const DsV4ForwardCfg& fwd_cfg,
    DsV4Workspace& ws,
    KvCache& kv_cache,
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
