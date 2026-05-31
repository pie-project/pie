#pragma once

// Composed DeepSeek-V3/V4-style MLA (multi-head latent attention) transformer
// block forward (prefill, bf16). Chains the banked + validated primitives:
//   rmsnorm  ->  Q lora (down/up + qk-norm)  ->  KV lora (down + kv-norm)
//   ->  RoPE (partial, on q_pe / k_pe)  ->  ABSORB q_nope into latent space
//   ->  write compressed KV to paged cache  ->  latent paged MLA attention
//   ->  ABSORB latent output up to V  ->  o_proj  ->  residual add.
//
// FORMULATION: ABSORBED (a.k.a. "weight absorption"). The banked attention
// kernel `ops::mla_naive_paged` runs entirely in the compressed kv_lora_rank
// latent space (it takes q_nope already projected to [T,nh,kv_lora_rank] and
// emits a latent output [T,nh,kv_lora_rank]); to match that contract we fold
// the per-head NoPE key up-projection into the query (W_uk) and the per-head
// value up-projection into the latent output (W_uv). This is the standard MLA
// inference trick and is what the kernel's header comment describes.
//
// ===========================================================================
// WEIGHT LAYOUTS (all device bf16, row-major). H = hidden_size, nh = num_heads.
//   attn_norm        [H]                              input RMSNorm gain
//   W_q_a            [q_lora_rank, H]                 q down-proj   (HF: q@W^T)
//   q_a_ln           [q_lora_rank]                    q_a RMSNorm gain
//   W_q_b            [nh*(qk_nope_head_dim+qk_rope_head_dim), q_lora_rank]
//                                                     q up-proj     (HF: q_a@W^T)
//   W_kv_a           [kv_lora_rank+qk_rope_head_dim, H] kv down-proj (HF: hn@W^T)
//   kv_a_ln          [kv_lora_rank]                   compressed-kv RMSNorm gain
//   W_uk             [nh, kv_lora_rank, qk_nope_head_dim]
//                       PRE-TRANSPOSED absorbed NoPE key up-proj. For head h it
//                       is used directly as the gemm `w` arg (act@w^T) so that
//                       q_nope_latent[t,h,:] = q_nope[t,h,:] @ W_uk_orig[h]
//                       where W_uk_orig[h] is [qk_nope_head_dim, kv_lora_rank].
//                       i.e. W_uk[h] == transpose(W_uk_orig[h]).
//   W_uv             [nh, v_head_dim, kv_lora_rank]
//                       PRE-TRANSPOSED absorbed value up-proj. For head h used
//                       directly as gemm `w` so that
//                       o_v[t,h,:] = o_latent[t,h,:] @ W_uv_orig[h]
//                       where W_uv_orig[h] is [kv_lora_rank, v_head_dim].
//                       i.e. W_uv[h] == transpose(W_uv_orig[h]).
//   W_o              [H, nh*v_head_dim]               output proj  (HF: ov@W^T)
// ===========================================================================
//
// PAGED MLA CACHE (matches ops/mla_paged.cuh & kernels/mla_write.cuh):
//   ckv_pages [num_pages, page_size, kv_lora_rank]      bf16  (NoPE latent)
//   kpe_pages [num_pages, page_size, qk_rope_head_dim]  bf16  (RoPE key)
// CSR page bookkeeping is the same flashinfer-style layout the banked kernels
// consume: qo_indptr [R+1], kv_page_indices, kv_page_indptr [R+1],
// kv_last_page_lens [R] (all uint32, device).
//
// RoPE is applied via kernels::rope_partial_bf16 over the full qk_rope_head_dim
// slice (rotary_dim == qk_rope_head_dim, NeoX pairing), to q_pe per head and to
// k_pe as a single shared head (num_kv_heads == 1).

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

// Per-layer MLA weight pointers (device bf16). See header block for layouts.
struct MlaLayerWeights {
    const void* attn_norm;  // [H]
    const void* W_q_a;      // [q_lora_rank, H]
    const void* q_a_ln;     // [q_lora_rank]
    const void* W_q_b;      // [nh*(qk_nope+qk_rope), q_lora_rank]
    const void* W_kv_a;     // [kv_lora_rank+qk_rope, H]
    const void* kv_a_ln;    // [kv_lora_rank]
    const void* W_uk;       // [nh, kv_lora_rank, qk_nope_head_dim]  (transposed)
    const void* W_uv;       // [nh, v_head_dim, kv_lora_rank]        (transposed)
    const void* W_o;        // [H, nh*v_head_dim]
};

// Runs one MLA block in place on `hidden` [num_tokens, H] bf16. All work is
// enqueued on `stream`; this entry synchronizes before returning. Allocates its
// own activation scratch (mirrors llama_layer_bf16 / moe_mlp_block_bf16).
// Returns the first CUDA error encountered (or attention launch failure).
cudaError_t mla_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const MlaLayerWeights& w, const std::int32_t* positions,
    void* ckv_pages, void* kpe_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    int num_tokens, int num_requests, int H, int num_heads,
    int q_lora_rank, int kv_lora_rank,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim,
    int page_size, float rms_eps, float sm_scale, float rope_theta);

}  // namespace pie_cuda_device::forward
