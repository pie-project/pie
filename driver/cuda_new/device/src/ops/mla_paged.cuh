#pragma once

// Naive paged MLA (multi-head latent attention) — the self-contained,
// arch-agnostic fallback that needs NO flashinfer/cutlass. Streams over each
// request's compressed-latent KV pages with a flash-style online softmax (no
// tensor cores, no cooperative scheduler, no flashinfer template machinery).
// This is the path the upstream driver routes to on Blackwell (sm_100+),
// where flashinfer's FA2 BatchMLAPagedAttention zero-outputs.
//
// Lifted from driver/cuda/src/ops/attention_mla.cu: ONLY the raw-pointer bf16
// naive path (`launch_mla_naive_paged` plus its `mla_naive_paged_kernel`). The
// flashinfer FA2 path (`plan_attention_mla_bf16`, `dispatch_mla_512_64`,
// `MlaPlanCache`, `mla_use_naive_backend`, `dispatch_attention_mla_bf16`) is
// dropped, so this TU never includes flashinfer headers. The `MlaCacheLayerView`
// argument is replaced by raw device pointers + scalar dims (mirroring how
// attention_naive_paged.{cuh,cu} replaced KvCacheLayerView with raw pointers),
// so this never includes mla_cache.hpp / tensor.hpp either.
//
// MLA cache layout (compressed latent KV — one latent vector plus one rotary
// key vector per token, shared across all query heads):
//   ckv_pages: [num_pages, page_size, kv_lora_rank]      bf16  (NoPE latent)
//   kpe_pages: [num_pages, page_size, qk_rope_head_dim]  bf16  (RoPE key)
// The query is likewise split into a NoPE part absorbed into the latent space
// and a RoPE part:
//   q_nope:    [total_tokens, num_heads, kv_lora_rank]   bf16
//   q_pe:      [total_tokens, num_heads, qk_rope_head_dim] bf16
//
// The per-head score is the sum over BOTH splits:
//   score = (q_nope · ckv_j  +  q_pe · kpe_j) * sm_scale
// i.e. the NoPE dot runs over the full kv_lora_rank latent dims and the RoPE
// dot runs over the qk_rope_head_dim rotary dims; they are added before the
// softmax. The attention output lives in the kv_lora latent space (it is the
// softmax-weighted sum of the ckv latent vectors, NOT a separate V), so the
// output shape matches q_nope: [total_tokens, num_heads, kv_lora_rank]. The
// downstream MLA forward (latent_to_v / o_proj) consumes that latent output.
//
// One block per (query_token, head). Threads cover the latent (kv_lora_rank)
// axis. Each block runs a single-pass online (flash) softmax over the
// request's causal KV span, rescaling the running max / denominator / latent
// accumulator per KV row.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::ops {

// `q_nope`              [total_tokens, num_heads, kv_lora_rank]      bf16
// `q_pe`                [total_tokens, num_heads, qk_rope_head_dim]  bf16
// `ckv_pages`           [num_pages, page_size, kv_lora_rank]         bf16
// `kpe_pages`           [num_pages, page_size, qk_rope_head_dim]     bf16
// `o`                   [total_tokens, num_heads, kv_lora_rank]      bf16
// `qo_indptr_d`         [R+1] device  — start row of each request in q/o
// `kv_page_indices_d`   device        — concatenated page-id list
// `kv_page_indptr_d`    [R+1] device  — request page-list bounds
// `kv_last_page_lens_d` [R] device    — last-page valid token count
// `sm_scale`            softmax scale (already resolved; the upstream default
//                       is 1/sqrt(kv_lora_rank + qk_rope_head_dim))
// `causal`              true enables the causal mask
// `index_mask`          optional DSA top-k mask [num_query_tokens,
//                       index_mask_stride] uint8 (1=attend), applied to
//                       in-batch keys (j < index_mask_stride); null = dense
//
// Constraint (verbatim from upstream): kv_lora_rank must be a multiple of the
// internal block (128) and kv_lora_rank/128 <= 8.
void mla_naive_paged(
    const void* q_nope,
    const void* q_pe,
    const void* ckv_pages,
    const void* kpe_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_heads,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int page_size,
    float sm_scale,
    bool causal,
    cudaStream_t stream,
    const std::uint8_t* index_mask = nullptr,
    int index_mask_stride = 0);

}  // namespace pie_cuda_device::ops
