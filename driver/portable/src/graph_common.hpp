#pragma once

// Shared graph-building helpers used by the per-arch graph builders
// (graph_qwen3.cpp, graph_gemma4.cpp).
//
// These are layered on top of ggml's primitives — `norm_scale` collapses
// the L4MA-vs-Gemma RMSNorm weighting variants, `build_moe_ffn` is the
// shared MoE block, etc. The graph-input/output structs `GraphInputs`
// and `GraphResult` are produced by every per-arch builder and consumed
// by `Executor::compute_`.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <ggml.h>

#include "executor/executor.hpp"

namespace pie_portable_driver {

// Per-call ggml graph node budget. Sized to comfortably fit MoE + spec
// decode batches (every layer's MoE op count is ~10-20 nodes).
inline constexpr std::size_t GRAPH_MAX_NODES = 1ull << 14;  // reduced from 1<<19 (524288) to 1<<14 (16384) — the original value requests a ~10GB scheduler buffer regardless of model size, which fails malloc on memory-constrained systems for small models like Qwen3-0.6B.

// ggml's CUDA flash_attn supports head_dim ≤ 256 across all GQA ratios.
// Gemma 4's full-attention layers use head_dim=512 — those layers fall
// back to manual SDPA in graph_gemma4.cpp until ggml ships a dkq=512
// kernel for GQA-8.
inline constexpr std::int32_t kFlashAttnMaxHeadDim = 256;

// MoE activation variants. SiLU/GeLU are the standard SwiGLU/GeGLU paths.
// SwigluOai is gpt-oss's `silu(α·gate) * (up + β)` with clamps on gate
// (max=limit) and up (±limit) — matches ggml_swiglu_oai.
enum class MoeActivation { Silu, Gelu, SwigluOai };

// RMSNorm scale step. Applies `x * w` for L4MA, or `x * (1 + w)` for the
// Gemma family (their RMSNorm is centered at 1; weights are stored as
// `actual_weight - 1`). Cast handles BF16 / F16 weights against F32
// activations.
ggml_tensor* norm_scale(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                        bool plus_one);

// Add `w` to `x`, casting `w` to `x`'s dtype if they differ. Used for
// optional QKV / o_proj / sinks biases stored at non-F32 dtype.
ggml_tensor* add_with_cast(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w);

// Mixture-of-Experts FFN block.
//
// Inputs:
//   `cur`            : [hidden, n_total]                    (post-norm activation)
//   `gate_inp`       : [hidden, n_experts]                  (router weight)
//   `gate_exps`      : [hidden, ff,    n_experts]           (stacked SwiGLU gate)
//   `up_exps`        : [hidden, ff,    n_experts]           (stacked SwiGLU up)
//   `down_exps`      : [ff,     hidden, n_experts]          (stacked output proj)
//   `n_experts`      : total expert count
//   `n_used`         : top-k routing
//   `act`            : SiLU / GeLU / SwigluOai
//   `norm_topk`      : renormalize selected weights to sum to 1
//
// Optional bias tensors:
//   `gate_inp_b`     : [n_experts]               (gpt-oss router bias)
//   `gate_exps_b`    : [ff, n_experts]           (gpt-oss per-expert gate bias)
//   `up_exps_b`      : [ff, n_experts]           (gpt-oss per-expert up bias)
//   `down_exps_b`    : [hidden, n_experts]       (gpt-oss per-expert down bias)
//
// `oai_alpha` / `oai_limit` are the SwigluOai parameters (only used when
// act == SwigluOai).
//
// Returns: [hidden, n_total] — the per-token weighted sum over selected
// experts. Mirrors `src/llama-graph.cpp::llm_graph_context::build_moe_ffn`,
// pared down to the SwiGLU/GeGLU softmax-routing common case.
ggml_tensor* build_moe_ffn(ggml_context* ctx,
                           ggml_tensor* cur,
                           ggml_tensor* gate_inp,
                           ggml_tensor* gate_exps,
                           ggml_tensor* up_exps,
                           ggml_tensor* down_exps,
                           std::int32_t n_experts,
                           std::int32_t n_used,
                           MoeActivation act,
                           bool norm_topk,
                           ggml_tensor* gate_inp_b   = nullptr,
                           ggml_tensor* gate_exps_b  = nullptr,
                           ggml_tensor* up_exps_b    = nullptr,
                           ggml_tensor* down_exps_b  = nullptr,
                           float        oai_alpha    = 1.702f,
                           float        oai_limit    = 7.0f);

// Inputs the per-arch graph builder allocates and returns alongside the
// graph. The engine uploads host-side per-batch arrays into these
// tensors before calling `ggml_backend_graph_compute`.
struct GraphInputs {
    ggml_tensor*              tok_input;   // I32 [total_n_tokens]
    ggml_tensor*              pos_input;   // I32 [total_n_tokens]
    ggml_tensor*              kv_idxs;     // I64 [total_n_tokens] (write idxs)
    ggml_tensor*              out_idx;     // I32 [n_request]
    // Slow path (per-request): one mask + gather tensor per request.
    std::vector<ggml_tensor*> masks;       // F16 [n_kv_r, n_tokens_pad_r]
    // Phi-3-small only: per-request blocksparse-clipped mask used by
    // blocksparse layers (every layer where (il+1) % dense_every_n != 0).
    // Empty for other archs.
    std::vector<ggml_tensor*> masks_blocksparse;
    std::vector<ggml_tensor*> gather_idxs; // I32 [n_kv_r] gather idxs per req
    // Fast path (pure-decode, M11): packed gather + mask, single attn call.
    ggml_tensor*              packed_gather = nullptr; // I32 [n_req * max_n_kv]
    ggml_tensor*              packed_mask   = nullptr; // F16 [max_n_kv, 64, 1, n_req] — SWA-clipped
    // Companion "no-SWA" mask for archs with mixed sliding+full
    // attention patterns (Gemma 4). Null unless the engine populated
    // packed_mask_full_f16 in the BatchPlan.
    ggml_tensor*              packed_mask_full = nullptr;
    // Paged-attn fast-path inputs. Allocated only when both
    // `plan.pure_decode` AND `Model::supports_paged_attn_ext()` hold;
    // null otherwise (the materialize+flash_attn_ext fallback reads
    // from `packed_gather` / `packed_mask` instead). The graph builder
    // tests `page_indices != nullptr` to choose between paths.
    // Layout matches FlashInfer / vLLM convention.
    ggml_tensor*              page_indices   = nullptr;  // I32 [total_pages_in_batch]
    ggml_tensor*              page_indptr    = nullptr;  // I32 [n_req + 1]
    ggml_tensor*              last_page_lens = nullptr;  // I32 [n_req]

    // ── Vision (multimodal) side-inputs. Allocated only when the batch
    // carries image spans (plan.has_images); null for text-only passes.
    // Filled host-side by upload_graph_inputs from the BatchPlan's vis_*
    // arrays. The per-arch builder runs the vision encoder over these and
    // scatters the result into the token-embedding rows at vis_img_rows.
    ggml_tensor*              vis_pixels    = nullptr;  // F32 [patch_dim, n_patch]
    ggml_tensor*              vis_pos_embed = nullptr;  // F32 [hidden, n_patch]
    ggml_tensor*              vis_rope_cos  = nullptr;  // F32 [head_dim, n_patch]
    ggml_tensor*              vis_rope_sin  = nullptr;  // F32 [head_dim, n_patch]
    ggml_tensor*              vis_img_rows  = nullptr;  // I64 [n_token] scatter rows
    ggml_tensor*              vis_img_rows_i32 = nullptr;  // I32 [n_token] (get_rows)
    ggml_tensor*              vis_pool_matrix = nullptr;  // F32 [n_patch, n_token] (Gemma-4)
    ggml_tensor*              vis_attn_mask   = nullptr;  // F32 [n_patch, n_patch] block-diagonal (multi-image)

    // Audio (Gemma-4 Conformer) side-inputs; filled from the BatchPlan aud_*
    // arrays. The encoder runs over these and scatters into aud_rows.
    // Per-clip (multiple audio clips per forward are encoded independently).
    ggml_tensor*              aud_pe       = nullptr;  // F32 [hidden, P] (shared)
    std::vector<ggml_tensor*> aud_features;            // per clip F32 [n_mel, n_frame]
    std::vector<ggml_tensor*> aud_win_mask;            // per clip F32 [dwin, n_token]
    std::vector<ggml_tensor*> aud_rows;                // per clip I64 [n_token]
};

// Per-arch graph builders return one of these. Exactly one of the three
// output tensors is non-null per call, matched against the corresponding
// BatchPlan flag:
//   all_greedy        → tokens_out
//   uniform_top_sample → top_k_idx + top_k_probs
//   else              → logits
struct GraphResult {
    ggml_cgraph* gf;
    ggml_tensor* logits      = nullptr;  // F32 [vocab, n_slots]
    ggml_tensor* tokens_out  = nullptr;  // I32 [n_slots]
    ggml_tensor* top_k_idx   = nullptr;  // I32 [K, n_slots]
    ggml_tensor* top_k_probs = nullptr;  // F32 [K, n_slots]
    GraphInputs  in;
};

// Stage all per-batch host arrays into the graph's input tensors. Picks
// between the slow (per-request) path and the M11 packed-decode fast path
// based on `plan.pure_decode`.
void upload_graph_inputs(const GraphResult& g,
                         const Executor::BatchPlan& plan);

// Allocate the standard set of per-batch graph input tensors
// (`tok_input`, `pos_input`, `kv_idxs`, `out_idx`) plus either the
// packed-decode pair (`packed_gather` + `packed_mask`) or per-request
// (`masks[r]` + `gather_idxs[r]`) inputs based on `plan.pure_decode`.
// Returns the populated GraphInputs; per-arch builders chain layer
// computation off these tensors.
GraphInputs declare_graph_inputs(ggml_context* ctx,
                                 const Executor::BatchPlan& plan,
                                 std::int32_t n_total,
                                 std::int32_t n_req,
                                 bool         supports_paged_attn_ext);

// Per-request attention via `ggml_flash_attn_ext`. Slices the request's
// Q view, gathers its K/V from the paged cache, permutes for flash_attn
// (axes 0,2,1,3), and applies optional gpt-oss attention sinks. Returns
// `[head_dim, n_q_heads, n_tokens, 1]` (already `ggml_cont`). Caller
// concatenates per-request outputs along ne[2] then reshapes to
// `[head_dim * n_q_heads, n_total]` via `concat_per_request_attn`.
ggml_tensor* build_request_flash_attn(
    ggml_context* ctx,
    ggml_tensor*  Q,                // [head_dim, n_q_heads, n_total]
    ggml_tensor*  k_cached,         // [n_embd_gqa, total_slots] post-set_rows
    ggml_tensor*  v_cached,
    ggml_tensor*  gather_idx_r,     // I32 [n_kv]
    ggml_tensor*  mask_r,           // F16 [n_kv, n_tokens_pad, 1, 1]
    std::int32_t  qo_start,
    std::int32_t  n_tokens,
    std::int32_t  n_kv,
    std::int32_t  head_dim,
    std::int32_t  n_kv_heads,
    std::int32_t  n_q_heads,
    float         kq_scale,
    float         attn_softcap,
    ggml_tensor*  sinks /* nullable; F32 [n_q_heads] when present */);

// Concatenate per-request attention outputs (each
// `[head_dim, n_q_heads, n_tokens, 1]`) along ne[2] and reshape to
// `[head_dim * n_q_heads, n_total]` for the o_proj matmul.
ggml_tensor* concat_per_request_attn(
    ggml_context* ctx,
    const std::vector<ggml_tensor*>& per_req,
    std::int32_t  head_dim,
    std::int32_t  n_q_heads,
    std::int32_t  n_total);

// LoRA delta. Returns `y + scale * (b @ (a @ x))`. No-op (returns y) when
// either a or b is null — adapters can target some projections without
// targeting all of them. Used for both QKV and o_proj in the qwen3 graph
// builder.
ggml_tensor* apply_lora_delta(ggml_context* ctx,
                              ggml_tensor*  y,
                              ggml_tensor*  a,
                              ggml_tensor*  b,
                              ggml_tensor*  x,
                              float         scale);

}  // namespace pie_portable_driver
