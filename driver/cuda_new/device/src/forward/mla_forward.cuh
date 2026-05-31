#pragma once

// Full MLA (multi-head latent attention) decoder forward (prefill, bf16) — the
// DeepSeek-V4 / Kimi / GLM frontier shape. The MLA analogue of
// llama_forward.cuh: it chains the VALIDATED per-layer block `mla_block_bf16`
// across N layers, sandwiched by embed (front) and final-RMSNorm → lm_head →
// greedy-argmax (back):
//
//   embed(token_ids)
//     -> for L in [0, n_layers): mla_block_bf16(hidden, layers[L], <cache slice L>)
//     -> rmsnorm(hidden, final_norm)
//     -> lm_head gemm  (logits)
//     -> argmax        (next-token ids)
//
// Each layer runs IN PLACE on the residual stream `hidden` [num_tokens, H], and
// each layer owns its own MlaLayerWeights (per-layer weight pointers) and its
// own slice of the per-layer paged MLA cache (see "PER-LAYER MLA CACHE STRIDE"
// below). The CSR page bookkeeping (qo_indptr / kv_page_indices /
// kv_page_indptr / kv_last_page_lens) is shared across all layers: the paging
// (which physical page holds which token) is identical for every layer; only
// the cache *base* changes per layer.
//
// FORMULATION: ABSORBED, inherited verbatim from mla_block_bf16 — the NoPE key
// up-proj is folded into the query (W_uk) and the value up-proj into the latent
// output (W_uv), so the paged attention runs entirely in the compressed
// kv_lora_rank latent space. See mla_block.cuh for the full per-layer math and
// the per-layer weight layouts (MlaLayerWeights).
//
// ===========================================================================
// MODEL-LEVEL WEIGHT LAYOUTS (all device bf16, row-major). V = vocab, H = hidden.
//   embed       [V, H]   token-id -> hidden lookup (y[n,:] = embed[token[n],:])
//   layers      host array of `n_layers` MlaLayerWeights (each holding device
//               pointers; see mla_block.cuh for per-field layouts)
//   final_norm  [H]      pre-lm_head RMSNorm gain
//   lm_head     [V, H]   output projection (HF storage; logits = normed @ lm_head^T)
// ===========================================================================
//
// PAGED MLA CACHE (matches ops/mla_paged.cuh & kernels/mla_write.cuh):
//   ckv_pages [n_layers, num_pages, page_size, kv_lora_rank]      bf16
//   kpe_pages [n_layers, num_pages, page_size, qk_rope_head_dim]  bf16
//
// PER-LAYER MLA CACHE STRIDE (the one model-level addition over the block):
//   ckv_pages is a single contiguous device allocation whose layer-L slice
//   starts at element offset  L * (num_pages * page_size * kv_lora_rank).
//   kpe_pages likewise at      L * (num_pages * page_size * qk_rope_head_dim).
//   i.e. layer L is handed   ckv_pages + L*ckv_layer_stride_elems  and
//                            kpe_pages + L*kpe_layer_stride_elems
//   exactly as llama_forward offsets kv_k/kv_v by L*(num_pages*page_size*Hkv).
//
// All work is enqueued on `stream`; mla_block_bf16 synchronizes per layer (it
// allocates/frees its own activation scratch), and this entry synchronizes once
// more before returning.

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "mla_block.cuh"  // MlaLayerWeights

namespace pie_cuda_device::forward {

// Whole-model MLA weights. `layers` is a host array of `dims.n_layers`
// per-layer MlaLayerWeights (each holding device pointers). embed / lm_head are
// [vocab, hidden], final_norm is [hidden] — all device bf16.
struct MlaForwardWeights {
    const void* embed;                  // [vocab, hidden]
    const MlaLayerWeights* layers;      // host array, length dims.n_layers
    const void* final_norm;             // [hidden]
    const void* lm_head;                // [vocab, hidden]
};

// MLA model dims (mirrors the per-block scalar args, plus model-level n_layers /
// vocab / num_pages and the numerical knobs). See mla_block.cuh for the meaning
// of the per-head dims and the kv_lora_rank % 128 == 0 attention constraint.
struct MlaForwardDims {
    int n_layers;
    int hidden;             // H
    int num_heads;          // nh
    int q_lora_rank;
    int kv_lora_rank;       // ckv  (must be a multiple of 128, /128 <= 8)
    int qk_nope_head_dim;   // nope
    int qk_rope_head_dim;   // rope
    int v_head_dim;
    int vocab;
    int page_size;
    int num_pages;          // pages PER LAYER (sets the per-layer cache stride)
    float rms_eps;
    float sm_scale;         // softmax scale (e.g. 1/sqrt(kv_lora_rank+qk_rope))
    float rope_theta;
};

// Runs the full MLA forward. Inputs:
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
cudaError_t mla_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const MlaForwardWeights& w,
    const std::int32_t* positions,
    void* ckv_pages, void* kpe_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, const MlaForwardDims& dims);

}  // namespace pie_cuda_device::forward
