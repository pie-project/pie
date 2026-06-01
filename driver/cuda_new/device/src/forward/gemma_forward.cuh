#pragma once

// =============================================================================
// Gemma-3 / Gemma-4 family decoder FORWARD (prefill, bf16) — first pass.
//
// This is the model-level analogue of mla_forward.cuh / moe_forward.cuh for the
// Gemma sandwich-norm architecture. It chains the BANKED, validated per-layer
// block `decoder_layer_gemma_inplace` (gemma_layer.cuh) across N layers, wrapped
// by Gemma's forward-level concerns:
//
//   embed(token_ids) * embed_scale            (Gemma scales embeddings by √H)
//     -> for L in [0, n_layers):
//          decoder_layer_gemma_inplace(hidden, layers[L], <cache slice L>,
//                                      window_left[L], attn_logit_softcap)
//     -> rmsnorm_gemma(hidden, final_norm)     ((1 + w) convention)
//     -> lm_head gemm                          (logits = normed @ lm_head^T)
//     -> logit_softcap(logits, final_logit_softcap)   (Gemma final tanh cap)
//     -> argmax                                (greedy next-token ids)
//
// Each layer runs IN PLACE on the residual stream `hidden` [num_tokens, H] and
// owns its own GemmaLayerWeights plus its own slice of the per-layer paged KV
// cache (see "PER-LAYER KV CACHE STRIDE"). The CSR page bookkeeping (qo_indptr /
// kv_page_indices / kv_page_indptr / kv_last_page_lens) is shared across layers;
// only the cache *base* and the per-layer sliding `window_left` change.
//
// =============================================================================
// PROJECT STRATEGY NOTE: "implement frontiers upfront, then validate/fix".
// Gemma-4's exact spec is NOT externally pinned here. The accompanying
// self-test (gemma_forward_selftest.cu) is a SELF-CONSISTENCY test: a CPU
// reference reproduces the EXACT formulation implemented below (same kernels,
// same bf16 rounding boundaries). It validates the WIRING — shapes, per-layer
// cache strides, sliding/full alternation, softcap placement, embed scaling —
// NOT agreement with an external Gemma checkpoint. Every architectural choice
// that is uncertain for Gemma-3/4 is enumerated in "ARCHITECTURAL ASSUMPTIONS"
// so a later validation pass can confirm or correct each one in isolation.
//
// =============================================================================
// ARCHITECTURAL ASSUMPTIONS (read before trusting against a real checkpoint):
//
//   A1. EMBED SCALE. Gemma multiplies the token embedding by a constant
//       (√hidden in the public checkpoints). We apply it as an explicit
//       elementwise scale right after the embed lookup, using the caller-
//       supplied `embed_scale` (so √H, √H rounded to bf16, or any normalizer
//       can be passed without changing code). The scale is applied in fp32 and
//       re-rounded to bf16, matching how the CPU reference scales.
//
//   A2. SANDWICH NORMS = (1 + w). Inherited verbatim from the banked
//       decoder_layer_gemma_inplace: four rmsnorm_gemma per layer
//       (input / post_attn / pre_ffn / post_ffn), each using the (1 + weight)
//       gain convention. The final pre-lm_head norm is ALSO rmsnorm_gemma
//       (1 + w) — Gemma's final norm is the same family, NOT the plain
//       rmsnorm used by llama/mla/moe forwards. If a checkpoint stores the
//       final-norm gain already offset by −1, subtract 1 before upload.
//
//   A3. qk-norm: DEFERRED (OFF in this pass). Real Gemma-2/3 apply a per-head
//       RMSNorm to Q (and Gemma-3 also to K) BEFORE RoPE. The banked
//       decoder_layer_gemma_inplace does NOT do this — it ropes the raw
//       projections directly. We therefore DO NOT apply qk-norm here, and the
//       CPU reference matches (no qk-norm) so the self-consistency test still
//       passes. `dims.qk_norm` is carried as a flag and is asserted/expected to
//       be 0 (OFF); wiring per-head qk-norm is a follow-up that touches the
//       banked layer, which this deliverable must not modify. DOCUMENTED GAP.
//
//   A4. RoPE: full-head NeoX, single global theta. The banked layer calls the
//       full-slice NeoX rope_bf16 over the entire head_dim with one rope_theta.
//       Real Gemma-3/4 use a *dual* RoPE base (a large "global" theta on
//       full-attention layers, a small "local" theta on sliding layers) and
//       Gemma-4 full layers may use PARTIAL rope (rope_partial.cuh exists for
//       exactly this). This pass uses ONE theta for ALL layers and full-slice
//       rope, because that is what the banked block does. Per-layer theta and
//       partial rope are DEFERRED (would require either a banked-layer change
//       or a parallel layer entry). DOCUMENTED GAP.
//
//   A5. SLIDING-WINDOW ALTERNATION. Gemma interleaves sliding-window-attention
//       layers with full-attention layers (e.g. 5:1 in Gemma-2/3). We do NOT
//       hard-code the pattern: the caller supplies a per-layer
//       `window_left[L]` array (length n_layers). `window_left[L] < 0` ⇒ full
//       attention for layer L; `>= 0` ⇒ sliding window of that many *prior*
//       tokens (the banked attention's `kv < kv_lim - 1 - window_left` test, so
//       window_left = W keeps the current token + W tokens to its left = W+1
//       total). This makes the 5:1 (or any) pattern a pure caller concern.
//
//   A6. ATTENTION SOFTCAP. A single scalar `attn_logit_softcap` is applied to
//       every layer's attention logits (Gemma-2 used 50.0; Gemma-3 dropped it,
//       so pass <= 0 to disable). FINAL logit softcap is a separate scalar
//       `final_logit_softcap` (Gemma-2 used 30.0) applied to the lm_head output
//       before argmax. Both flow through the banked kernels
//       (transform_logit / logit_softcap_bf16). `<= 0` disables either.
//
//   A7. ATTENTION sm_scale. The banked attention is called with sm_scale = -1
//       (⇒ 1/√head_dim) by decoder_layer_gemma_inplace; we cannot override it
//       without editing the banked layer. Real Gemma uses a query_pre_attn
//       scalar that for most configs equals 1/√head_dim but for some Gemma-3
//       sizes is 1/√hidden_per_head_override. If a checkpoint needs a custom
//       scale, that is a banked-layer change → DEFERRED. DOCUMENTED GAP.
//
//   A8. AltUp (Gemma-3n / Gemma-4 alternating residual streams): DEFERRED
//       (OFF). altup_predict/altup_correct kernels are banked (altup.cuh) but
//       are NOT wired here. We run a SINGLE residual stream, i.e.
//       altup_num_inputs == 1 ≡ "no AltUp". `dims.altup_num_inputs` is carried
//       and expected to be 1; any other value is rejected by the host entry
//       (returns cudaErrorInvalidValue) so a caller cannot silently believe
//       AltUp is active. Wiring K>1 streams (predict before / correct after
//       each layer, with router/correction coefficients) is a clean follow-up
//       that does not require touching the banked layer. DOCUMENTED GAP.
//
//   A9. KV CACHE LAYOUT. NHD: [num_pages, page_size, n_kv_heads, head_dim] bf16,
//       matching write_kv_to_pages_bf16(hnd_layout=false) and the naive paged
//       attention. Per-layer stride below.
//
// =============================================================================
// MODEL-LEVEL WEIGHT LAYOUTS (all device bf16, row-major; V = vocab, H = hidden):
//   embed       [V, H]   token-id -> hidden lookup, then * embed_scale
//   layers      host array of `n_layers` GemmaLayerWeights (device pointers;
//               per-field layout per gemma_layer.cuh: 4 norms [H], wq [Hq,H],
//               wk/wv [Hkv,H], wo [H,Hq], w_gate/w_up [I,H], w_down [H,I])
//   final_norm  [H]      pre-lm_head rmsnorm_gemma gain ((1 + w) convention)
//   lm_head     [V, H]   output projection (logits = normed @ lm_head^T)
//
// PER-LAYER KV CACHE STRIDE (the model-level addition over the block):
//   k_pages / v_pages are each a single contiguous device allocation whose
//   layer-L slice starts at element offset
//       L * (num_pages * page_size * n_kv_heads * head_dim).
//   Identical to llama_forward / moe_forward's L*(num_pages*page_size*Hkv).
// =============================================================================

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemma_layer.cuh"  // GemmaLayerWeights, decoder_layer_gemma_inplace, LayerScratch

struct PieWorkspace;  // workspace.hpp — persistent activation scratch + attn plan

namespace pie_cuda_device::forward {

// Whole-model Gemma weights. `layers` is a host array of `dims.n_layers`
// per-layer GemmaLayerWeights (each holding device pointers). embed / lm_head
// are [vocab, hidden], final_norm is [hidden] — all device bf16.
struct GemmaForwardWeights {
    const void* embed;                    // [vocab, hidden]
    const GemmaLayerWeights* layers;      // host array, length dims.n_layers
    const void* final_norm;               // [hidden]   (rmsnorm_gemma gain)
    const void* lm_head;                  // [vocab, hidden]
};

// Gemma model dims + numerical knobs. See ARCHITECTURAL ASSUMPTIONS above for
// the meaning / status of each uncertain field.
struct GemmaForwardDims {
    int n_layers;
    int hidden;                 // H
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int intermediate;           // I (FFN width)
    int vocab;                  // V
    int page_size;
    int num_pages;              // pages PER LAYER (sets the per-layer KV stride)

    // Sliding-window alternation (A5). If `window_left` is non-null it must
    // point to a host array of length `n_layers`; entry L is passed verbatim to
    // layer L (<0 = full attention, >=0 = sliding window of that many prior
    // tokens). If `window_left` is null, the scalar `window_left_all` is used
    // for every layer (set to -1 for an all-full-attention model).
    const int* window_left;     // host array [n_layers], or nullptr
    int window_left_all;        // used iff window_left == nullptr

    float attn_logit_softcap;   // A6: per-layer attention logit cap (<=0 = off)
    float final_logit_softcap;  // A6: lm_head output cap before argmax (<=0 = off)
    float embed_scale;          // A1: embedding multiplier (e.g. sqrtf(hidden))
    float rms_eps;
    float rope_theta;           // A4: single global RoPE base for ALL layers

    int qk_norm;                // A3: must be 0 (per-head qk-norm DEFERRED/off)
    int altup_num_inputs;       // A8: must be 1 (AltUp DEFERRED/off)
};

// Runs the full Gemma forward (prefill, bf16). Owns its own activation scratch
// (residual stream + per-layer LayerScratch + normed buffer), allocated once and
// reused across layers. Enqueues all work on `stream`, synchronizes once before
// returning. NO ABI entry — the parent wires this into abi.cpp.
//
// Inputs:
//   token_ids   [num_tokens]            int32 device  (greedy decode input)
//   positions   [num_tokens]            int32 device  (RoPE positions)
//   k_pages /   [n_layers, num_pages, page_size, n_kv_heads, head_dim] bf16 dev
//     v_pages   (NHD; written by each layer for all T tokens)
//   qo_indptr / kv_page_indices / kv_page_indptr / kv_last_page_lens
//               flashinfer-style CSR page bookkeeping (uint32 device), SHARED
//               across all layers.
// Outputs:
//   out_logits     [num_tokens, vocab]  bf16  device  (post final-softcap)
//   out_token_ids  [num_tokens]         int32 device  (greedy argmax)
//
// Returns cudaSuccess, or the first CUDA error (an alloc/launch failure), or
// cudaErrorInvalidValue if a DEFERRED feature is requested (qk_norm != 0 or
// altup_num_inputs != 1).
cudaError_t gemma_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream, PieWorkspace* ws,
    const std::int32_t* token_ids, const GemmaForwardWeights& w,
    const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, const GemmaForwardDims& dims);

}  // namespace pie_cuda_device::forward
