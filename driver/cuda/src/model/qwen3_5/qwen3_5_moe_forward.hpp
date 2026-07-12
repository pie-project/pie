#pragma once

// Qwen3.6-MoE forward driver. Reuses the linear-attn / full-attn /
// rs_cache plumbing from Qwen3.5; only the MLP becomes a sparse-MoE
// block (routed experts + shared expert with sigmoid gate).

#include <cstdint>

#include "ops/attention_workspace.hpp"
#include "device_buffer.hpp"
#include "store/kv_cache.hpp"
#include "model/llama_like/qwen3.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"  // reuse Qwen3_5LinearAttnWorkspace
#include "model/qwen3_5/qwen3_5_moe.hpp"
#include "ops/gemm.hpp"
#include "store/recurrent_state_cache.hpp"

namespace pie_cuda_driver::model {

// MoE-side workspace. Reuses Qwen3.5's la_ws for the linear-attn
// staging tensors.
struct Qwen3_5MoeMlpWorkspace {
    // Routing.
    DeviceBuffer<std::uint16_t> router_logits;     // [N, E] bf16
    DeviceBuffer<std::int32_t>  topk_idx;          // [N, K] i32
    DeviceBuffer<float>         topk_weights;      // [N, K] fp32

    // Per-expert scratch (worst case: all N*K rows go to one expert).
    DeviceBuffer<std::uint16_t> expert_in;         // [N*K, H] bf16
    DeviceBuffer<std::uint16_t> expert_gate_up;    // [N*K, 2*I_moe] bf16
    DeviceBuffer<std::uint16_t> expert_act;        // [N*K, I_moe] bf16 (post SwiGLU)
    DeviceBuffer<std::uint16_t> expert_out;        // [N*K, H] bf16
    DeviceBuffer<std::int32_t>  expert_idx;        // [N*K] i32 (dst row)
    DeviceBuffer<float>         expert_w;          // [N*K] fp32

    // Shared expert scratch.
    DeviceBuffer<std::uint16_t> shared_gate;       // [N, I_shared]
    DeviceBuffer<std::uint16_t> shared_up;         // [N, I_shared]
    DeviceBuffer<std::uint16_t> shared_gate_up;    // [N, 2*I_shared(+scalar gate)]
    DeviceBuffer<std::uint16_t> shared_act;        // [N, I_shared]
    DeviceBuffer<std::uint16_t> shared_out;        // [N, H]
    DeviceBuffer<std::uint16_t> shared_gate_logit; // [N, 1]

    // Final MoE+shared sum buffer (pre residual).
    DeviceBuffer<std::uint16_t> moe_out;           // [N, H]

    // Decode fast-path scratch — separate pointer arrays for gate_up
    // and down_proj GEMMs (cuBLAS reads them at launch time, so the
    // two GEMMs cannot share buffers). Sized for N*K routed rows and
    // populated on-device so batched decode avoids host routing syncs.
    DeviceBuffer<const std::uint16_t*> a_gu_ptrs;  // [N*K]
    DeviceBuffer<const std::uint16_t*> b_gu_ptrs;
    DeviceBuffer<std::uint16_t*>       c_gu_ptrs;
    DeviceBuffer<const std::uint16_t*> a_dn_ptrs;
    DeviceBuffer<const std::uint16_t*> b_dn_ptrs;
    DeviceBuffer<std::uint16_t*>       c_dn_ptrs;
    DeviceBuffer<float>                batch_weights;  // [N*K] fp32

    // vLLM/SGL-style aligned decode scratch. Decode routes are grouped into
    // fixed-size expert blocks so cuBLAS sees M=block_size GEMMs instead of
    // M=1 GEMVs. PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK=0 disables it for
    // isolation experiments.
    int aligned_block_size = 0;
    std::size_t aligned_rows_capacity = 0;
    DeviceBuffer<std::int32_t>  aligned_route_ids;   // [aligned_rows]
    DeviceBuffer<std::int32_t>  aligned_expert_ids;  // [aligned_rows / block]
    DeviceBuffer<std::uint16_t> aligned_expert_in;   // [aligned_rows, H]
    DeviceBuffer<std::uint16_t> aligned_gate_up;     // [aligned_rows, 2*I_moe]
    DeviceBuffer<std::uint16_t> aligned_act;         // [aligned_rows, I_moe]
    DeviceBuffer<std::uint16_t> aligned_out;         // [aligned_rows, H]

    static Qwen3_5MoeMlpWorkspace allocate(
        int max_tokens, int hidden, int num_experts, int top_k,
        int moe_intermediate, int shared_intermediate);
};

void qwen3_5_moe_forward_paged(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3_5PlanState& plan_state,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
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
    int total_tokens, int num_requests,
    bool is_pure_decode,
    const std::uint8_t* mask_d,
    const std::int32_t* mask_indptr_d,
    const std::int32_t* slot_ids_h = nullptr,
    const std::uint8_t* is_fresh_h = nullptr,
    const std::int32_t* slot_ids_d = nullptr,
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0,
    const std::int32_t* commit_advance_gather = nullptr);

void qwen3_5_moe_mtp_process_cache(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::int32_t* slot_ids_d,
    const std::int32_t* source_row_indices,
    int total_tokens,
    int num_requests);

// Run one NEXTN step of Qwen3.6-MoE's one-layer MTP head. The executor can
// recursively feed this step's hidden rows back in to return multiple drafts.
void qwen3_5_moe_mtp_forward(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* position_ids,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    std::int32_t* sampled_token_ids,
    int num_tokens,
    int draft_step,
    int max_global_tokens);

// MoE-side workspace byte budget. Includes the optional aligned-decode
// rebatch arena (sized by PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK).
std::size_t qwen3_5_moe_workspace_bytes(const HfConfig& cfg,
                                        int N,
                                        int tp_size = 1);

}  // namespace pie_cuda_driver::model
