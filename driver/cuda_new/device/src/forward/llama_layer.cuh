#pragma once

// Composed llama-like decoder layer (prefill), chaining the lifted
// primitives in the standard pre-norm order:
//
//   normed   = rmsnorm(hidden, attn_norm)
//   q,k,v    = normed @ {Wq,Wk,Wv}^T
//   rope(q, k)
//   <<write k,v into the paged KV cache>>
//   attn     = attention(q, kv_pages)
//   hidden  += attn @ Wo^T
//   normed   = rmsnorm(hidden, ffn_norm)
//   hidden  += (silu(normed @ Wgate^T) * (normed @ Wup^T)) @ Wdown^T
//
// bf16; no biases / no qk-norm (base Llama/Mistral/Qwen2 shape).
//
// SLICE SIMPLIFICATION (documented): KV append is a contiguous device-to-
// device copy assuming a single fresh request occupying page
// `kv_page_indices[0]` from slot 0. The general paged append is a scatter
// kernel, lifted later.

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

// Per-layer weight pointers (device bf16). HF row-major storage.
struct LlamaLayerWeights {
    const void* attn_norm;
    const void* wq;
    const void* wk;
    const void* wv;
    const void* wo;
    const void* ffn_norm;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
    // Per-head q/k RMSNorm gains [head_dim] (Qwen3 / OLMo-3). Both null = no
    // qk-norm (Llama); when present, applied to q,k after the qkv proj and
    // before RoPE.
    const void* q_norm;
    const void* k_norm;
    // Additive q/k/v projection biases (Qwen2): q_bias [Hq], k_bias/v_bias [Hkv].
    // Null = no bias (Llama/Qwen3); when present, added after the qkv proj.
    const void* q_bias;
    const void* k_bias;
    const void* v_bias;
};

// Pre-allocated bf16 activation scratch, sized for the forward token
// capacity and reused across layers (no per-call alloc). Provided by a
// PieWorkspace; the standalone layer entry below mallocs its own.
struct LayerScratch {
    void* normed;
    void* q;
    void* k;
    void* v;
    void* attn;
    void* o;
    void* gate;
    void* up;
    void* mlp;
    void* mlp_out;
};

// Runs one decoder layer in place on `hidden` [num_tokens, hidden_size]
// bf16, using caller-provided scratch. All work is enqueued on `stream`;
// the caller synchronizes.
void decoder_layer_inplace(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const LlamaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    const LayerScratch& s,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta);

// Convenience: allocates scratch per call, runs one layer, syncs, frees.
// Kept as the standalone single-layer entry. Returns the first CUDA error.
cudaError_t llama_layer_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const LlamaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta);

}  // namespace pie_cuda_device::forward
