#pragma once

#include <cstdint>

#include "ops/attention_workspace.hpp"
#include "distributed.hpp"
#include "store/dsa_cache.hpp"
#include "store/kv_cache.hpp"
#include "store/mla_cache.hpp"
#include "model/glm5/glm5.hpp"
#include "model/kimi/kimi_forward.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct Glm5ForwardCfg {
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
    bool emit_logits = true;
};

// Scratch buffers for the GLM-5.1 forward pass. Mirrors KimiWorkspace
// (MLA + MoE) since GLM-5.1's per-layer compute is structurally the
// same — the only architectural difference is the DSA indexer, which
// this first pass skips.
struct Glm5Workspace {
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
    // DSA lightning-indexer scratch (prefill-path top-k selection).
    DeviceTensor idx_q;             // [N, index_n_heads*index_head_dim]
    DeviceTensor idx_k;             // [N, index_head_dim]
    DeviceTensor idx_w;             // [N, index_n_heads] (bf16)
    DeviceTensor idx_mask;          // [N, N] uint8 (1=attend, 0=skip)
    DeviceTensor norm_y;            // [N, H]
    DeviceTensor gate;              // [N, max(local_I, routed_I)]
    DeviceTensor up;                // [N, max(local_I, routed_I)]
    DeviceTensor router_logits;     // [N, num_experts]
    DeviceTensor topk_idx;          // [N, top_k] int32
    DeviceTensor topk_weights;      // [N, top_k] fp32
    DeviceTensor route_idx;         // [N*top_k] int32
    DeviceTensor route_w;           // [N*top_k] fp32
    DeviceTensor expert_in;         // [max_Ne, H]
    DeviceTensor expert_gate;       // [max_Ne, routed_I]
    DeviceTensor expert_up;         // [max_Ne, routed_I]
    DeviceTensor expert_out;        // [max_Ne, H]
    DeviceTensor moe_out;           // [N, H]
    DeviceTensor shared_gate;       // [N, shared_I]
    DeviceTensor shared_up;         // [N, shared_I]
    DeviceTensor shared_act;        // [N, shared_I]
    DeviceTensor shared_out;        // [N, H]
    DeviceTensor logits;            // [O, vocab]

    static Glm5Workspace allocate(
        const HfConfig& cfg,
        int max_tokens,
        int max_logit_rows,
        int max_position_embeddings,
        int tp_size);
};

std::size_t glm5_workspace_bytes(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int max_position_embeddings,
    int tp_size);

void glm5_forward_paged(
    const Glm5Weights& w,
    const HfConfig& cfg,
    const Glm5ForwardCfg& fwd_cfg,
    KimiPlanState& mla_plan,
    Glm5Workspace& ws,
    MlaCache& mla_cache,
    DsaCache& dsa_cache,
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
    const std::uint8_t* row_valid_d = nullptr,
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0);

}  // namespace pie_cuda_driver::model
