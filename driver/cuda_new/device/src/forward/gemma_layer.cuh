#pragma once

// Composed Gemma-2 decoder layer (prefill). Demonstrates a *second arch* as a
// directory-local addition: a new forward file reusing the shared primitives
// (gemm, rope, paged attention, residual, KV scatter) plus Gemma's own lifted
// kernels (`rmsnorm_gemma` (1+w), `geglu_tanh`).
//
// Gemma-2's defining shape vs Llama is the "sandwich" norm — every sublayer
// output is normed BEFORE the residual add — giving four (1+w) RMSNorms per
// layer:
//   residual = hidden
//   h = rmsnorm_gemma(hidden, input_ln);  h = attn(h);  h = rmsnorm_gemma(h, post_attn_ln)
//   hidden = residual + h
//   residual = hidden
//   h = rmsnorm_gemma(hidden, pre_ffn_ln); h = geglu_mlp(h); h = rmsnorm_gemma(h, post_ffn_ln)
//   hidden = residual + h
//
// Attention may carry a logit softcap + sliding window (both supported by the
// naive paged attention); the embedding ×√d scale and the final logit softcap
// are forward-level concerns added when the full gemma forward lands.

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "llama_layer.cuh"  // reuse forward::LayerScratch

namespace pie_cuda_device::forward {

// Gemma-2 per-layer weights: 4 sandwich norms + the standard projections.
struct GemmaLayerWeights {
    const void* input_ln;
    const void* post_attn_ln;
    const void* pre_ffn_ln;
    const void* post_ffn_ln;
    const void* wq;
    const void* wk;
    const void* wv;
    const void* wo;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
};

// One Gemma-2 layer in place on `hidden`, caller-provided scratch.
// `attn_logit_softcap <= 0` disables the attention softcap; `window_left < 0`
// disables the sliding window.
void decoder_layer_gemma_inplace(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const GemmaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    const LayerScratch& s,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta,
    int window_left, float attn_logit_softcap,
    const AttnPlan& attn);

// Convenience: allocates scratch per call, runs one layer, syncs, frees.
cudaError_t gemma_layer_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const GemmaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta,
    int window_left, float attn_logit_softcap);

}  // namespace pie_cuda_device::forward
