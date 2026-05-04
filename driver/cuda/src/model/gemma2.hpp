#pragma once

// Gemma-2 decoder. Differs from Llama-like in five ways the forward
// graph has to know about:
//   1. Four RMSNorms per layer — input + post-attn + pre-ffn + post-ffn —
//      with Gemma's `(1 + w) * x_hat` shift.
//   2. Embedding scale: `y *= sqrt(hidden_size)` after the lookup.
//   3. Query pre-attention scale: `q *= 1/sqrt(query_pre_attn_scalar)`.
//      flashinfer always applies its own `1/sqrt(head_dim)` inside the
//      kernel; we leave that intact and pre-scale `q` by
//      `sqrt(head_dim) / sqrt(query_pre_attn_scalar)` so the effective
//      scale matches Gemma's spec.
//   4. GeGLU(tanh) MLP instead of SwiGLU.
//   5. Final-logit soft-cap: `logits = cap * tanh(logits / cap)`.
//
// Out of scope here (Gemma-3 / Gemma-4 territory):
//   * QK norm (Gemma-3+).
//   * Dual RoPE bases — global vs sliding.
//   * Per-layer alternating sliding-window attention.
//   * KV-cache sharing across layers, Per-Layer Embeddings.

#include <cstdint>
#include <vector>

#include "distributed.hpp"
#include "engine.hpp"
#include "kv_cache.hpp"
#include "model/llama_like.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

struct Gemma2LayerWeights {
    // Four RMSNorms — see header comment for placement.
    const DeviceTensor* attn_norm_pre  = nullptr;  // input_layernorm
    const DeviceTensor* attn_norm_post = nullptr;  // post_attention_layernorm
    const DeviceTensor* mlp_norm_pre   = nullptr;  // pre_feedforward_layernorm
    const DeviceTensor* mlp_norm_post  = nullptr;  // post_feedforward_layernorm

    const DeviceTensor* q_proj = nullptr;
    const DeviceTensor* k_proj = nullptr;
    const DeviceTensor* v_proj = nullptr;
    const DeviceTensor* o_proj = nullptr;

    // Per-head q/k RMSNorm (Gemma-3+; null on Gemma-2).
    const DeviceTensor* q_norm = nullptr;  // [head_dim]
    const DeviceTensor* k_norm = nullptr;  // [head_dim]

    const DeviceTensor* gate_proj = nullptr;
    const DeviceTensor* up_proj   = nullptr;
    const DeviceTensor* down_proj = nullptr;
};

struct Gemma2Weights {
    const DeviceTensor* embed       = nullptr;
    const DeviceTensor* final_norm  = nullptr;  // model.norm
    const DeviceTensor* lm_head     = nullptr;
    std::vector<Gemma2LayerWeights> layers;
};

struct Gemma2ForwardCfg {
    // `query_pre_attn_scalar` from HF config (typically `head_dim`, but
    // Gemma-2 9B uses 224 with head_dim=256). We pre-scale `q` so the
    // effective attention scale becomes `1/sqrt(query_pre_attn_scalar)`.
    float query_pre_attn_scalar = 256.f;

    // Final logit soft-cap (HF: `final_logit_softcapping`). Zero / negative
    // disables. Gemma-2-2B/9B/27B all set this to 30.
    float final_logit_softcap = 0.f;

    // Per-attention-call soft-cap (HF: `attn_logit_softcapping`). Zero
    // disables. Gemma-2 sets this to 50; Gemma-3 doesn't use it. Routes
    // to the `AttnVariantSoftcap` flashinfer template inside
    // `dispatch_attention_flashinfer_*_bf16`.
    float attn_logit_softcap = 0.f;

    // Per-head q/k RMSNorm (Gemma-3+). Null weights on Gemma-2.
    bool use_qk_norm = false;

    // Force the prefill kernel for batches whose GQA group size isn't in
    // flashinfer's decode dispatch table — same role as in
    // `LlamaLikeForwardCfg`.
    bool force_prefill_path = false;

    // Per-layer sliding-window left context (-1 = full causal). Empty
    // when the model uses a single attention type across all layers
    // (homogeneous Gemma-2 with sliding everywhere, or one-off tests).
    std::vector<int> per_layer_window_left;

    // Per-layer rope_theta. Empty when the model uses a single rope
    // base across all layers; populated for Gemma-3 (full-attention
    // layers use ~1M, sliding layers use ~10K).
    std::vector<float> per_layer_rope_theta;

    // Tensor-parallel state. tp_size == 1 keeps the original
    // single-GPU forward; tp_size > 1 activates sharded GEMM dims
    // and drops in two NCCL all-reduces per layer (after o_proj and
    // after down_proj). `tp_comm` must be non-null whenever tp_size > 1.
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
};

Gemma2Weights bind_gemma2(const Engine& engine);

// Gemma-3 reuses the same `Gemma2Weights` shape — the only schema delta
// is that `q_norm` / `k_norm` are non-null per layer.
Gemma2Weights bind_gemma3(const Engine& engine);

void gemma2_forward_paged(
    const Gemma2Weights& w,
    const HfConfig& cfg,
    const Gemma2ForwardCfg& fwd_cfg,
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
