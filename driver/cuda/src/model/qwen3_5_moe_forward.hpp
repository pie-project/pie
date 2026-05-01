#pragma once

// Qwen3.6-MoE forward driver. Reuses the linear-attn / full-attn /
// state-cache plumbing from Qwen3.5; only the MLP becomes a sparse-MoE
// block (routed experts + shared expert with sigmoid gate).

#include <cstdint>

#include "attention_workspace.hpp"
#include "device_buffer.hpp"
#include "kv_cache.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5_forward.hpp"  // reuse Qwen3_5LinearAttnWorkspace
#include "model/qwen3_5_moe.hpp"
#include "ops/gemm.hpp"
#include "qwen3_5_state_cache.hpp"

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
    DeviceBuffer<std::uint16_t> shared_act;        // [N, I_shared]
    DeviceBuffer<std::uint16_t> shared_out;        // [N, H]
    DeviceBuffer<std::uint16_t> shared_gate_logit; // [N, 1]

    // Final MoE+shared sum buffer (pre residual).
    DeviceBuffer<std::uint16_t> moe_out;           // [N, H]

    static Qwen3_5MoeMlpWorkspace allocate(
        int max_tokens, int hidden, int num_experts, int top_k,
        int moe_intermediate, int shared_intermediate);
};

void qwen3_5_moe_forward_paged(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    Qwen3_5StateCache& state_cache,
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
    const std::int32_t* mask_indptr_d);

}  // namespace pie_cuda_driver::model
