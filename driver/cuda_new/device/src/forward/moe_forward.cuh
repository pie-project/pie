#pragma once

// Dense-MoE transformer FORWARD (prefill, bf16) — the Qwen3.5-MoE / GPT-OSS
// frontier shape, first-pass composition.
//
//   embed(tokens)
//   for L in 0..n_layers:
//       h       = rmsnorm(hidden, attn_norm[L])
//       q,k,v   = h @ {Wq,Wk,Wv}^T  ; rope(q,k) ; paged-KV append
//       attn    = paged_attention(q, kv_pages)          # GQA, causal
//       hidden += attn @ Wo^T
//       h       = rmsnorm(hidden, ffn_norm[L])
//       hidden += moe_mlp_block(h, router[L], wgu[L], wdown[L])   # top-K MoE
//   logits = rmsnorm(hidden, final_norm) @ lm_head^T
//   next   = argmax(logits)                              # greedy
//
// A frontier MoE model = llama-style attention (GQA + RoPE + paged KV) with the
// dense swiglu MLP replaced by the validated dense top-K MoE FFN sublayer
// (`moe_mlp_block_bf16`). This is the FIRST PASS: standard llama attention.
//
// REFINEMENTS DEFERRED (note, do not block on first pass), and where each slots:
//   * qk-norm (Qwen3): an extra rmsnorm on q/k per head right after the qkv
//     gemms and before rope, in the attention half of moe_decoder_layer_inplace.
//   * shared expert (DeepSeek-V2/Qwen2-MoE): one always-on dense FFN added to
//     the MoE output; would live inside / alongside the moe_mlp_block call.
//   * sliding-window attention (GPT-OSS alternating layers): pass a non-negative
//     `window_left` per layer to attention_naive_paged_bf16 in the attn half.
//
// All tensors bf16, HF row-major storage. See struct docs for layouts.

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

// Per-layer weight pointers (device bf16). HF row-major storage.
//
// Attention (Hq = n_q_heads*head_dim, Hkv = n_kv_heads*head_dim, H = hidden):
//   attn_norm [H]            rmsnorm gain before qkv
//   wq        [Hq,  H]       q proj    (y = x @ wq^T)
//   wk        [Hkv, H]       k proj
//   wv        [Hkv, H]       v proj
//   wo        [H,   Hq]      output proj
// MoE FFN (E = num_experts, I = intermediate):
//   ffn_norm  [H]            rmsnorm gain before the router
//   router_w  [E,  H]        router logits proj (y = x @ router_w^T -> [T,E])
//   wgu       [E, 2I, H]     per-expert gate||up (gate rows [0,I), up rows [I,2I))
//   wdown     [E,  H, I]     per-expert down proj
struct MoeLayerWeights {
    const void* attn_norm;
    const void* wq;
    const void* wk;
    const void* wv;
    const void* wo;
    const void* ffn_norm;
    const void* router_w;
    const void* wgu;
    const void* wdown;
};

// Whole-model weights. `layers` is a host array of `n_layers` per-layer
// structs (each holding device pointers). embed/lm_head [vocab, hidden],
// final_norm [hidden] — all device bf16.
struct MoeForwardWeights {
    const void* embed;
    const MoeLayerWeights* layers;
    int n_layers;
    const void* final_norm;
    const void* lm_head;
};

// Static model dims, mirroring LlamaForwardDims style. All ints.
struct MoeForwardDims {
    int hidden_size;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int intermediate;   // per-expert FFN width I
    int num_experts;    // E
    int top_k;          // K experts selected per token
    int n_layers;
    int vocab;
    int page_size;
    float rms_eps;
    float rope_theta;
};

// Runs the full MoE forward (prefill). Allocates its own activation scratch
// (like the block entries); the caller provides cublas + stream and owns the
// KV pool and CSR page bookkeeping.
//
//   token_ids   [num_tokens] i32 device          input ids
//   positions   [num_tokens] i32 device          rope positions
//   kv_k/kv_v   per-layer paged KV pools, NHD layout
//               [n_layers][num_kv_pages, page_size, n_kv_heads, head_dim] bf16;
//               layer L lives at base + L*(num_kv_pages*page_size*Hkv) elems.
//   qo_indptr        [num_requests+1] u32 device  query rows per request
//   kv_page_indices  [..]            u32 device   concatenated page ids
//   kv_page_indptr   [num_requests+1] u32 device  per-request page-list bounds
//   kv_last_page_lens[num_requests]   u32 device  valid tokens in last page
//   out_logits   [num_tokens, vocab] bf16 device  receives lm_head logits
//   out_token_ids[num_tokens]        i32 device   greedy argmax next tokens
//
// Returns the first CUDA error encountered (cudaSuccess on success).
cudaError_t moe_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const MoeForwardWeights& w,
    const std::int32_t* positions,
    void* kv_k, void* kv_v,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, int num_kv_pages,
    const MoeForwardDims& d);

}  // namespace pie_cuda_device::forward
