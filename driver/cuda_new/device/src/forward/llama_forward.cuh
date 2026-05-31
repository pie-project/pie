#pragma once

// Full llama-like forward (prefill): embed → N decoder layers → final
// rmsnorm → lm_head → argmax. The capstone of the phase-1 hot path — a
// complete forward producing next-token ids through the seam, composed
// from the lifted primitives + decoder_layer_inplace.
//
// SLICE SIMPLIFICATIONS (documented): single fresh request, contiguous
// per-layer KV append (see llama_layer.cuh), and synthetic weights at the
// ABI (the real weight loader is phase 2). KV is one page per layer.

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "llama_layer.cuh"

struct PieWorkspace;  // workspace.hpp

namespace pie_cuda_device::forward {

// Whole-model weights. `layers` is a host array of `n_layers` per-layer
// weight structs (each holding device pointers). embed/lm_head [vocab,
// hidden], final_norm [hidden] — all device bf16.
struct LlamaWeights {
    const void* embed;
    const LlamaLayerWeights* layers;
    int n_layers;
    const void* final_norm;
    const void* lm_head;
};

// Runs the full forward. `token_ids` [num_tokens] int32 device →
// `out_token_ids` [num_tokens] int32 device (greedy argmax). `out_logits`
// [num_tokens, vocab] bf16 device receives the lm_head logits. KV for layer
// L lives at `kv_k`/`kv_v` + L*(page_size*n_kv_heads*head_dim) bf16 elems.
// Returns the first CUDA error.
cudaError_t llama_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream, PieWorkspace* ws,
    const std::int32_t* token_ids, const LlamaWeights& w, const std::int32_t* positions,
    void* kv_k, void* kv_v,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, int num_kv_pages, int vocab, float rms_eps, float rope_theta);

}  // namespace pie_cuda_device::forward
