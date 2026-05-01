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
};

struct MixtralLayerWeights {
    // Attention weights — same set as Qwen3LayerWeights' attention half.
    const DeviceTensor* attn_norm = nullptr;
    const DeviceTensor* mlp_norm  = nullptr;
    const DeviceTensor* q_proj    = nullptr;
    const DeviceTensor* k_proj    = nullptr;
    const DeviceTensor* v_proj    = nullptr;
    const DeviceTensor* o_proj    = nullptr;

    // Sparse-MoE block.
    const DeviceTensor* router    = nullptr;     // [num_experts, hidden]
    std::vector<MixtralExpertWeights> experts;   // size = num_experts
};

struct MixtralWeights {
    const DeviceTensor* embed       = nullptr;
    const DeviceTensor* final_norm  = nullptr;
    const DeviceTensor* lm_head     = nullptr;
    std::vector<MixtralLayerWeights> layers;
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
