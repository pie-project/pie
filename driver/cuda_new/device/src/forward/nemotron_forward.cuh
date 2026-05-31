#pragma once

// Nemotron-H whole-model decoder FORWARD (prefill, bf16) — the HYBRID stack
// that interleaves Mamba-2 mixer, attention, and MLP/FFN layers per a
// caller-provided per-layer schedule. The Nemotron-H analogue of
// mla_forward.cuh / llama_forward.cuh: it chains VALIDATED per-layer pieces
// across N layers, sandwiched by embed (front) and final-RMSNorm -> lm_head ->
// greedy-argmax (back):
//
//   embed(token_ids)
//     -> for L in [0, n_layers):  dispatch on kinds[L]
//          'M' (Mamba)     : nemotron_mamba_block_bf16(hidden, mamba_w)   [in place,
//                            fuses pre-norm + residual internally? NO — see below]
//          'A' (Attention) : attention sublayer (pre-norm -> qkv -> rope ->
//                            paged-KV append -> naive paged attn -> o_proj ->
//                            residual add)  [attention-ONLY, no FFN]
//          'F' (FFN/MLP)   : MLP sublayer (pre-norm -> SwiGLU MLP -> residual add)
//     -> rmsnorm(hidden, final_norm)
//     -> lm_head gemm   (logits)
//     -> argmax         (next-token ids)
//
// ===========================================================================
// UPSTREAM STRUCTURE (confirmed against driver/cuda/src/model/nemotron_h.hpp
// {NemotronHLayerWeights} and nemotron_h_forward.cpp {the per-layer loop +
// attention_layer / mamba_layer / moe_layer}):
//
//   * Nemotron-H has exactly THREE layer kinds: Mamba, Attention, MoE — each a
//     SINGLE mixer per layer. There is ONE norm weight per layer (Lw.norm),
//     applied pre-mixer (residual stream `ws.y` -> rmsnorm -> `ws.norm_x`), and
//     the mixer output is added back into the residual stream. Attention layers
//     are ATTENTION-ONLY (NO fused FFN, unlike Llama where one layer does
//     attn+MLP). The FFN/MoE is always its own separate layer.
//   * Upstream single-GPU (TP==1) residual flow per layer:
//       norm_x = rmsnorm(y, layer.norm)
//       Mamba :  y += out_proj( mamba_core(norm_x) )        (gemm beta=1)
//       Attn  :  y += o_proj( attn(qkv(norm_x)) )           (gemm beta=1)
//       MoE   :  norm_y = moe(norm_x); y += norm_y          (residual_add)
//   * Final:  norm_y = rmsnorm(y, final_norm); logits = norm_y @ lm_head^T.
//
// ===========================================================================
// ASSUMPTIONS / SIMPLIFICATIONS (DOCUMENTED — this is a first-pass, self-test
// driven composition; the goal is wiring/layout correctness vs a CPU reference
// of the SAME formulation, not bit-exact upstream parity):
//
//   A1. FFN ('F') kind is a DENSE SwiGLU MLP (rmsnorm -> [silu(gate)*up] ->
//       down -> residual), mirroring llama_layer's MLP sublayer. Upstream
//       Nemotron-H's non-attention/non-mamba layers are actually SPARSE MoE
//       with squared-ReLU (relu2) experts and sigmoid+correction-bias routing.
//       That exact path needs relu2 / topk-sigmoid-bias kernels not yet lifted
//       into cuda_new; the dense SwiGLU FFN is used as the FFN stand-in here.
//       The 'F' weights carry {ffn_norm, w_gate, w_up, w_down}; swapping in the
//       real MoE block later is localized to nemotron_ffn_layer() in the .cu.
//       (If a model truly has no FFN layers, just omit 'F' from `kinds`.)
//
//   A2. The Mamba block (nemotron_mamba_block_bf16) is reused VERBATIM. It runs
//       in place on `hidden` and fuses its in_proj pre-norm? — NO: the banked
//       Mamba block does NOT apply the pre-norm; its in_proj reads `hidden`
//       directly. Upstream applies layer.norm BEFORE in_proj. Therefore for a
//       Mamba layer we rmsnorm `hidden` into a scratch buffer, hand THAT to the
//       block as its working residual, then add the block's delta back into the
//       real residual. Concretely: the block does `hidden += out_proj(mixer)`,
//       so we run it on a COPY = rmsnorm(hidden) and add (copy - rmsnorm(hidden))
//       back. Implemented as: pre-norm into `normed`; snapshot `normed` is the
//       block input; block updates `normed` in place to `normed + delta`; we
//       then residual_add the delta into `hidden`. See nemotron_mamba_layer().
//
//   A3. Single fresh prefill request (R=1), state caches (Mamba SSM/conv + the
//       attention paged KV) start empty and are allocated INTERNALLY for the
//       duration of the forward. The Mamba block already zero-inits its own SSM
//       + conv state per call (fresh prompt). The attention KV cache is one
//       page-pool per attention layer; CSR page bookkeeping is built internally
//       for the single contiguous request over [0, num_tokens).
//
//   A4. Attention: no biases, no QK-norm (base Llama/Qwen2 shape), NeoX RoPE
//       (interleaved=false), naive paged attention (sm_scale = 1/sqrt(head_dim)),
//       full causal mask, no sliding window / soft-cap. Per-attention-layer KV
//       pool is sized for a single request; pages are addressed [0, n_pages).
//
//   A5. lm_head and embed are separate weights (NOT tied) — caller supplies
//       both; pass the same pointer for both to tie. No embed scaling.
//
//   A6. All weights device bf16, row-major (HF storage: y = act @ W^T).
//
// ===========================================================================
// MODEL-LEVEL WEIGHT LAYOUTS (all device bf16, row-major). V=vocab, H=hidden.
//   embed       [V, H]
//   layers      host array of `n_layers` NemotronLayerWeights (union over the
//               three kinds; only the fields for the layer's kind are read)
//   final_norm  [H]
//   lm_head     [V, H]
//
// Per-layer Mamba weight layouts: see nemotron_block.cuh (NemotronMambaWeights).
// Per-layer attention weight layouts (HF): attn_norm [H]; wq [n_q*hd, H];
//   wk/wv [n_kv*hd, H]; wo [H, n_q*hd].
// Per-layer FFN weight layouts (HF): ffn_norm [H]; w_gate/w_up [I, H];
//   w_down [H, I].

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "nemotron_block.cuh"  // NemotronMambaWeights

namespace pie_cuda_device::forward {

// Per-layer attention weights (device bf16, HF row-major). Used when
// kinds[L] == 'A'.
struct NemotronAttnWeights {
    const void* attn_norm;  // [H]
    const void* wq;         // [n_q_heads*head_dim, H]
    const void* wk;         // [n_kv_heads*head_dim, H]
    const void* wv;         // [n_kv_heads*head_dim, H]
    const void* wo;         // [H, n_q_heads*head_dim]
};

// Per-layer dense-SwiGLU FFN weights (device bf16, HF row-major). Used when
// kinds[L] == 'F'. See assumption A1 (FFN stands in for the upstream MoE layer).
struct NemotronFfnWeights {
    const void* ffn_norm;  // [H]
    const void* w_gate;    // [intermediate, H]
    const void* w_up;      // [intermediate, H]
    const void* w_down;    // [H, intermediate]
};

// Per-layer weight union. Only the member matching `kind` is read. `kind` is
// the same char the caller places in the `kinds` array ('M'/'A'/'F').
//
// `mamba_pre_norm` is the LAYER pre-norm gain for a Mamba layer ([H], device
// bf16). It is distinct from `mamba.norm_weight`, which is the gated-RMSNorm
// gain INSIDE the mixer ([intermediate]). Upstream Nemotron-H applies the
// layer pre-norm (Lw.norm) before in_proj; the banked Mamba block does NOT, so
// we pass it separately here. (Attention/FFN carry their pre-norm inside their
// own weight struct as attn_norm / ffn_norm.)
struct NemotronLayerWeights {
    char kind;                    // 'M' Mamba | 'A' attention | 'F' FFN/MLP
    const void* mamba_pre_norm;   // [H] layer pre-norm gain, valid iff kind=='M'
    NemotronMambaWeights mamba;   // valid iff kind=='M'
    NemotronAttnWeights  attn;    // valid iff kind=='A'
    NemotronFfnWeights   ffn;     // valid iff kind=='F'
};

// Whole-model Nemotron-H weights. `layers` is a host array of `dims.n_layers`
// per-layer NemotronLayerWeights (each holding device pointers).
struct NemotronForwardWeights {
    const void* embed;                       // [vocab, hidden]
    const NemotronLayerWeights* layers;      // host array, length dims.n_layers
    const void* final_norm;                  // [hidden]
    const void* lm_head;                     // [vocab, hidden]
};

// Model dims. The per-layer `kinds` schedule plus the Mamba and attention
// shape knobs and the global numerical knobs.
struct NemotronForwardDims {
    int n_layers;
    const char* kinds;     // host array [n_layers] of 'M'/'A'/'F' (layer schedule)

    int hidden;            // H
    int vocab;             // V

    // Mamba-2 mixer dims (shared across all Mamba layers).
    int mamba_num_heads;   // num_heads
    int mamba_head_dim;    // head_dim
    int mamba_state_size;  // state_size (SSM state dim per channel)
    int mamba_n_groups;    // n_groups   (B/C shared within a group)
    int mamba_conv_kernel; // conv_kernel (K)
    float time_step_min;   // softplus(dt) lower clamp (upstream 0.f)

    // Attention dims (shared across all attention layers).
    int attn_n_q_heads;    // n_q_heads
    int attn_n_kv_heads;   // n_kv_heads
    int attn_head_dim;     // head_dim
    int page_size;         // KV page size
    float rope_theta;      // RoPE base

    // FFN dims (shared across all FFN layers).
    int ffn_intermediate;  // I

    float rms_eps;
};

// Runs the full Nemotron-H hybrid forward for a single fresh prefill request.
// Inputs:
//   token_ids   [num_tokens]   int32 device  (prompt token ids)
//   positions   [num_tokens]   int32 device  (RoPE positions; 0..num_tokens-1)
// Outputs:
//   out_logits     [num_tokens, vocab]  bf16  device  (lm_head logits)
//   out_token_ids  [num_tokens]         int32 device  (greedy argmax)
//
// All work enqueued on `stream`; allocates its own residual / scratch / state
// caches and synchronizes before returning. Returns the first CUDA error.
cudaError_t nemotron_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const NemotronForwardWeights& w,
    const std::int32_t* positions,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, const NemotronForwardDims& dims);

}  // namespace pie_cuda_device::forward
