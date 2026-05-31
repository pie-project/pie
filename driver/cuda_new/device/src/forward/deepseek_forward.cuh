#pragma once

// Full DeepSeek-V3 / V4 decoder forward (prefill, bf16). The "MLA + MoE"
// frontier shape: each layer is an MLA (multi-head latent attention) attention
// sublayer followed by a feed-forward sublayer that is EITHER a dense SwiGLU
// MLP (the first `first_k_dense` layers) OR a top-K sparse-MoE MLP (all later
// layers). It is the MoE generalization of mla_forward.cuh — it chains the same
// VALIDATED per-layer attention block `mla_block_bf16` but adds an FFN sublayer
// after each attention block, sandwiched by embed (front) and final-RMSNorm ->
// lm_head -> greedy-argmax (back):
//
//   embed(token_ids)
//     -> for L in [0, n_layers):
//          mla_block_bf16(hidden, layers[L].attn, <cache slice L>)   // attn + residual
//          h = rmsnorm(hidden, layers[L].ffn_norm)
//          ffn = (L < first_k_dense) ? dense_swiglu(h, w_gate/w_up/w_down)
//                                    : moe_mlp_block_bf16(h, router_w/wgu/wdown)
//          hidden += ffn                                             // FFN residual
//     -> rmsnorm(hidden, final_norm)
//     -> lm_head gemm  (logits)
//     -> argmax        (next-token ids)
//
// Each layer runs IN PLACE on the residual stream `hidden` [num_tokens, H]. The
// MLA attention sublayer (`mla_block_bf16`) already does its OWN attn_norm +
// MLA attention + o_proj + residual internally; this entry only adds the FFN
// sublayer (its own ffn_norm + MLP + residual) on top.
//
// MLA FORMULATION: ABSORBED, inherited verbatim from mla_block_bf16 (NoPE key
// up-proj folded into the query W_uk, value up-proj into the latent output
// W_uv; paged attention runs in the compressed kv_lora_rank latent space). See
// mla_block.cuh for the full per-layer attention math and the MlaLayerWeights
// layouts.
//
// MoE FORMULATION: dense reference (moe_mlp.cuh): softmax over all E router
// logits, take top-K (renormalized), compute ALL E experts and combine the
// top-K per token (O(E) work, correct semantics). Dense FFN: standard SwiGLU
// (gate/up GEMMs -> silu(gate)*up -> down GEMM), matching llama_layer.cu.
//
// ===========================================================================
// MODEL-LEVEL WEIGHT LAYOUTS (all device bf16, row-major). V = vocab, H = hidden.
//   embed       [V, H]   token-id -> hidden lookup (y[n,:] = embed[token[n],:])
//   layers      host array of `n_layers` DeepseekLayerWeights (each holding
//               device pointers; see per-field layouts in the struct below and
//               in mla_block.cuh for the 9 MLA attention pointers)
//   final_norm  [H]      pre-lm_head RMSNorm gain
//   lm_head     [V, H]   output projection (HF storage; logits = normed @ lm_head^T)
// ===========================================================================
//
// PER-LAYER FFN WEIGHT LAYOUTS (device bf16, row-major):
//   ffn_norm   [H]                         pre-FFN RMSNorm gain
//   dense FFN (used iff layer index L < first_k_dense):
//     w_gate   [dense_inter, H]            HF: h @ w_gate^T -> [T, dense_inter]
//     w_up     [dense_inter, H]            HF: h @ w_up^T   -> [T, dense_inter]
//     w_down   [H, dense_inter]            HF: mlp @ w_down^T -> [T, H]
//   MoE FFN (used iff L >= first_k_dense; see moe_mlp.cuh):
//     router_w [E, H]                      router logits = h @ router_w^T -> [T, E]
//     wgu      [E, 2*moe_inter, H]         per expert gate||up (gate first)
//     wdown    [E, H, moe_inter]           per expert down-proj
//
// PAGED MLA CACHE (matches ops/mla_paged.cuh & kernels/mla_write.cuh):
//   ckv_pages [n_layers, num_pages, page_size, kv_lora_rank]      bf16
//   kpe_pages [n_layers, num_pages, page_size, qk_rope_head_dim]  bf16
//
// PER-LAYER MLA CACHE STRIDE (same convention as mla_forward.cuh):
//   ckv_pages is a single contiguous device allocation whose layer-L slice
//   starts at element offset  L * (num_pages * page_size * kv_lora_rank).
//   kpe_pages likewise at      L * (num_pages * page_size * qk_rope_head_dim).
//   i.e. layer L is handed   ckv_pages + L*ckv_layer_stride_elems  and
//                            kpe_pages + L*kpe_layer_stride_elems.
//
// All work is enqueued on `stream`; the per-layer sublayers (mla_block_bf16,
// moe_mlp_block_bf16) synchronize internally (they allocate/free their own
// activation scratch), and this entry synchronizes once more before returning.

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "mla_block.cuh"  // MlaLayerWeights

namespace pie_cuda_device::forward {

// Per-layer DeepSeek weight pointers (device bf16). `attn` carries the 9 MLA
// attention pointers (see mla_block.cuh). The dense-FFN pointers are used when
// the layer index < first_k_dense; the MoE pointers otherwise. (The unused set
// may be left null by the loader.) See header block for layouts.
struct DeepseekLayerWeights {
    MlaLayerWeights attn;            // the 9 MLA attention pointers (mla_block.cuh)
    const void* ffn_norm;            // [H]
    // dense FFN (used when layer index < first_k_dense):
    const void* w_gate;              // [dense_inter, H]
    const void* w_up;                // [dense_inter, H]
    const void* w_down;              // [H, dense_inter]
    // MoE FFN (used otherwise):
    const void* router_w;            // [E, H]
    const void* wgu;                 // [E, 2*moe_inter, H]
    const void* wdown;               // [E, H, moe_inter]
};

// Whole-model DeepSeek weights. `layers` is a host array of `dims.n_layers`
// per-layer DeepseekLayerWeights (each holding device pointers). embed /
// lm_head are [vocab, hidden], final_norm is [hidden] — all device bf16.
struct DeepseekWeights {
    const void* embed;                     // [vocab, H]
    const DeepseekLayerWeights* layers;    // host array, length dims.n_layers
    const void* final_norm;                // [H]
    const void* lm_head;                   // [vocab, H]
};

// DeepSeek model dims. Mirrors MlaForwardDims (the MLA attention scalars +
// model-level n_layers / vocab / num_pages / numerical knobs) and adds the FFN
// dims: first_k_dense (count of leading dense layers), dense_inter (dense MLP
// intermediate), moe_inter (per-expert MLP intermediate), num_experts, top_k.
// See mla_block.cuh for the per-head attention dims and the
// kv_lora_rank % 128 == 0 attention constraint.
struct DeepseekDims {
    int n_layers;
    int first_k_dense;       // layers [0, first_k_dense) are dense; rest are MoE
    int hidden;              // H
    int num_heads;           // nh
    int q_lora_rank;
    int kv_lora_rank;        // ckv  (must be a multiple of 128, /128 <= 8)
    int qk_nope_head_dim;    // nope
    int qk_rope_head_dim;    // rope
    int v_head_dim;
    int dense_inter;         // dense SwiGLU MLP intermediate size
    int moe_inter;           // per-expert MLP intermediate size
    int num_experts;         // E
    int top_k;               // experts routed per token
    int vocab;
    int page_size;
    int num_pages;           // pages PER LAYER (sets the per-layer cache stride)
    float rms_eps;
    float sm_scale;          // softmax scale (e.g. 1/sqrt(kv_lora_rank+qk_rope))
    float rope_theta;
};

// Runs the full DeepSeek forward. Inputs:
//   token_ids   [num_tokens]            int32 device  (greedy decode input)
//   positions   [num_tokens]            int32 device  (RoPE positions)
//   ckv_pages   [n_layers, num_pages, page_size, kv_lora_rank]      bf16 device
//   kpe_pages   [n_layers, num_pages, page_size, qk_rope_head_dim]  bf16 device
//   qo_indptr / kv_page_indices / kv_page_indptr / kv_last_page_lens
//               flashinfer-style CSR page bookkeeping (uint32 device), SHARED
//               across all layers.
// Outputs:
//   out_logits     [num_tokens, vocab]  bf16  device  (lm_head logits)
//   out_token_ids  [num_tokens]         int32 device  (greedy argmax)
//
// Returns the first CUDA error encountered (block-launch failure or a cuda API
// error), else cudaSuccess.
cudaError_t deepseek_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const DeepseekWeights& w,
    const std::int32_t* positions,
    void* ckv_pages, void* kpe_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, const DeepseekDims& dims);

}  // namespace pie_cuda_device::forward
