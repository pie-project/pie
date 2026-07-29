#pragma once

#include <cstdint>

#include "ops/attention_workspace.hpp"
#include "distributed.hpp"
#include "store/kv_cache.hpp"
#include "store/mla_cache.hpp"
#include "model/kimi/kimi.hpp"
#include "ops/attention_mla.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct KimiForwardCfg {
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
    bool emit_logits = true;
};

struct KimiWorkspace {
    DeviceTensor y;                 // [N, H]
    DeviceTensor norm_x;            // [N, H]
    DeviceTensor q_a;               // [N, q_lora_rank]
    // [N, q_lora_rank + kv_lora_rank + qk_rope], the landing buffer for the
    // fused q_a + kv_a projection. The GEMM is row-major, so each token's row
    // holds its q half followed by its kv half; the two consumers read their
    // half in place with a row pitch rather than the halves being split out.
    DeviceTensor qkv_a;
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
    // fp16 staging for the W4A16 decode GEMVs, whose inner loop is pure
    // `__hfma2` and so wants its activation already in fp16.
    DeviceTensor norm_y_fp16;       // [N, H]
    DeviceTensor expert_act_fp16;   // [N*top_k, routed_I]
    // Device-side aligned MoE scratch (batched-GEMM path).
    DeviceTensor aligned_route_ids;
    DeviceTensor aligned_expert_ids;
    DeviceTensor aligned_expert_in;
    DeviceTensor aligned_gate_up;
    DeviceTensor aligned_act;
    DeviceTensor aligned_out;
    DeviceTensor a_gu_ptrs;
    DeviceTensor b_gu_ptrs;
    DeviceTensor c_gu_ptrs;
    DeviceTensor a_dn_ptrs;
    DeviceTensor b_dn_ptrs;
    DeviceTensor c_dn_ptrs;
    int aligned_block_size = 0;
    int aligned_max_blocks = 0;
    // flashinfer CUTLASS fused-MoE scratch. The runner permutes, runs both
    // grouped GEMMs, applies SwiGLU and finalises the top-k weighted sum in one
    // call, replacing the whole gather/batched-GEMM/scatter chain above.
    DeviceTensor cutlass_ws;        // opaque runner workspace (uint8)
    DeviceTensor cutlass_row_map;   // [cutlass_max_rows * top_k] int32
    int cutlass_max_rows = 0;
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

std::size_t kimi_workspace_bytes(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int tp_size);

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
    const std::uint8_t* row_valid_d = nullptr,
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0);

}  // namespace pie_cuda_driver::model
