#pragma once

// Mixtral (sparse top-k MoE) decoder. Identical to llama_like in
// attention; the FFN block is replaced by:
//
//     router_logits = norm_y @ moe_gate.T            # [N, E]
//     probs         = softmax(router_logits)
//     topk_w, topk_idx = topk(probs, K)
//     topk_w       /= topk_w.sum(dim=-1, keepdim=True)
//     out           = sum_e (top_k_w * Expert_e(norm_y))
//
// We loop the experts on the host: for each expert e, gather the rows
// routed to it (via `launch_gather_bf16_rows`), run the expert's
// gate/up/down GEMMs through cuBLAS, then `launch_scatter_add_weighted`
// the result back into the residual stream. Top-K and renormalization
// happen on-device via `launch_topk_softmax_bf16`.

#include <cstdint>
#include <vector>

#include "engine.hpp"
#include "kv_cache.hpp"
#include "model/llama_like.hpp"
#include "model/qwen3.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

struct MixtralExpertWeights {
    const DeviceTensor* w_gate = nullptr;  // [intermediate, hidden] (HF: w1)
    const DeviceTensor* w_up   = nullptr;  // [intermediate, hidden] (HF: w3)
    const DeviceTensor* w_down = nullptr;  // [hidden, intermediate] (HF: w2)

    // Optional per-expert biases. Mixtral's reference release ships
    // bias-free MLPs; GPT-OSS adds them. Nullptr → bias-add step is
    // skipped at runtime (no overhead for plain Mixtral).
    const DeviceTensor* b_gate = nullptr;  // [intermediate] (gpt-oss)
    const DeviceTensor* b_up   = nullptr;  // [intermediate] (gpt-oss)
    const DeviceTensor* b_down = nullptr;  // [hidden]       (gpt-oss)
};

struct MixtralLayerWeights {
    // Attention weights — same set as Qwen3LayerWeights' attention half.
    const DeviceTensor* attn_norm = nullptr;
    const DeviceTensor* mlp_norm  = nullptr;
    const DeviceTensor* q_proj    = nullptr;
    const DeviceTensor* k_proj    = nullptr;
    const DeviceTensor* v_proj    = nullptr;
    const DeviceTensor* o_proj    = nullptr;

    // Optional QKV/O biases. GPT-OSS has them on every linear; plain
    // Mixtral / Qwen-3 have none. Nullptr → step skipped.
    const DeviceTensor* q_bias    = nullptr;
    const DeviceTensor* k_bias    = nullptr;
    const DeviceTensor* v_bias    = nullptr;
    const DeviceTensor* o_bias    = nullptr;

    // Per-head learnable attention sink. GPT-OSS only. Shape
    // [num_attention_heads], bf16. Nullptr → skip post-attention
    // sink rescale.
    const DeviceTensor* attn_sinks = nullptr;

    // Sparse-MoE block.
    const DeviceTensor* router      = nullptr;   // [num_experts, hidden]
    const DeviceTensor* router_bias = nullptr;   // [num_experts] (gpt-oss)
    std::vector<MixtralExpertWeights> experts;   // size = num_experts
};

struct MixtralWeights {
    const DeviceTensor* embed       = nullptr;
    const DeviceTensor* final_norm  = nullptr;
    const DeviceTensor* lm_head     = nullptr;
    std::vector<MixtralLayerWeights> layers;

    // Owns the bf16 expert tensors that bind_gpt_oss synthesises by
    // dequantising MXFP4 fused weights. Plain bind_mixtral leaves this
    // empty (per-expert tensors live in the Engine). Kept here so they
    // outlive any lookup-by-name path; pointers in `experts[*]` index
    // into this vector.
    std::vector<DeviceTensor> owned_expert_buffers;
};

MixtralWeights bind_mixtral(const Engine& engine);

void mixtral_forward_paged(
    const MixtralWeights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    int num_experts,
    int top_k,
    Qwen3Workspace& ws,
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
    const std::uint8_t* custom_mask_d = nullptr,
    const std::int32_t* custom_mask_indptr_d = nullptr);

}  // namespace pie_cuda_driver::model
