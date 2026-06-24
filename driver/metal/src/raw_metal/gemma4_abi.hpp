#pragma once
// ── gemma4 raw-Metal decode ABI (alpha) ──────────────────────────────────────
// The gemma4-E2B/E4B analog of delta's qwen3.6 `decode_abi.hpp`. Pure C++ (no
// Metal dep) so every lane includes it freely. It REUSES the shared, locked
// per-kernel bind layouts from `decode_abi.hpp` (Rms / Rope / Sdpa / Embed /
// KvAppend / Qmv / Argmax — same .metal, same arg order) and adds ONLY the
// gemma4-specific surface: geometry, the new bind enums (GeGLU-tanh + the PLE
// pipeline), and the gemma4 dispatch-DAG `Kernel` kinds.
//
// gemma4 vs qwen3.6 (charlie's source-derived surface, wiki mac-gemma4-kernel-surface):
//   + Per-Layer Embeddings (PLE): a layer-less precompute + a per-layer block.
//   + GeGLU-tanh activation (FFN + PLE) — replaces qwen3.6's silu_mul.
//   + 4-norm sandwich (adds post_attn_norm + post_ffn_norm).
//   + sliding-window SDPA(512) on the sliding layers + cross-layer KV-share.
//   + final logit softcap 30*tanh(x/30), embed * sqrt(hidden), weightless v_norm,
//     per-layer layer_scalar mul.
//   + RMSNorm WEIGHT_PLUS_ONE=false (plain rms; opposite of qwen3.6) — same kernel.
//   + TIED with embed_tokens CANONICAL (no lm_head; logits = embed^T).
//   - Dropped: all GDN/linear-attn, QSplit, AttnGate, 2x-wide q_proj.

#include <cstdint>

#include "decode_abi.hpp"  // shared Region / IoSlot / bind::{Rms,Rope,Sdpa,Embed,KvAppend,Qmv,Argmax}

namespace pie_metal_driver::raw_metal::gemma4 {

// ── Geometry (gemma4-E2B; E4B differs only in the scalar fields) ──────────────
struct Gemma4Geometry {
    // global
    int   hidden          = 1536;
    int   n_layers        = 35;
    int   vocab           = 262144;
    float eps             = 1e-6f;
    bool  tied_embeddings = true;    // embed_tokens is canonical; logits = embed^T
    bool  norm_plus_one   = false;   // gemma4 stores plain-rms weights (NOT 1+w)

    // attention
    int   n_q_heads       = 8;
    int   n_kv_heads      = 1;       // GQA factor = n_q/n_kv = 8
    int   head_dim        = 256;     // q_dim = 2048, kv_dim = 256
    int   rotary_dims     = 256;     // partial_rotary_factor = 1.0 -> full rotation
    float attn_scale      = 1.0f;    // Q/K-norm absorbs 1/sqrt(d)

    // RoPE: per-attention-type base frequency
    float rope_theta_global = 1.0e6f;   // full-attn layers (config rope_theta)
    float rope_theta_local  = 1.0e4f;   // sliding layers (config rope_local_base_freq)

    // sliding-window attention
    int   sliding_window  = 512;     // sliding layers attend the last 512 positions

    // mlp (GeGLU-tanh)
    int   intermediate    = 6144;

    // PLE (Per-Layer Embeddings)
    int   per_layer_emb_dim = 256;   // hidden_size_per_layer_input; ple table width = n_layers*this
    // per_layer_model_projection: [hidden] -> [n_layers*per_layer_emb_dim]

    // cross-layer KV sharing: the last `num_kv_shared_layers` layers re-attend the
    // most-recent earlier layer of the SAME attention type (no own k/v_proj/append).
    int   num_kv_shared_layers = 20;

    // final logit softcap: out = cap * tanh(logits / cap). 0 = none.
    float final_softcap   = 30.0f;

    // quantization (only when the target checkpoint is 4-bit; bf16 E2B leaves bits=0).
    int   q_group         = 64;
    int   q_bits          = 0;       // 0 = bf16 dense GEMV; 4 = affine_qmv

    // ── Per-layer attention-type schedule ────────────────────────────────────
    // full-attention every `full_attn_interval`-th layer: (il+1) % 5 == 0.
    static constexpr int full_attn_interval = 5;
    static constexpr bool is_full_attn(int layer) {
        return ((layer + 1) % full_attn_interval) == 0;
    }
    static constexpr bool is_sliding(int layer) { return !is_full_attn(layer); }

    int first_kv_shared() const { return n_layers - num_kv_shared_layers; }
    bool is_kv_shared(int layer) const {
        return num_kv_shared_layers > 0 && layer >= first_kv_shared();
    }
    // KV source layer for `layer`: itself when not shared, else the most-recent
    // earlier non-shared layer of the same attention type (-1 = config error).
    int kv_source(int layer) const {
        if (!is_kv_shared(layer)) return layer;
        const bool want_sliding = is_sliding(layer);
        for (int j = first_kv_shared() - 1; j >= 0; --j) {
            if (is_sliding(j) == want_sliding) return j;
        }
        return -1;
    }
    int n_full_attn() const {
        int n = 0;
        for (int L = 0; L < n_layers; ++L) n += is_full_attn(L) ? 1 : 0;
        return n;
    }
};

// ── gemma4-specific per-kernel binding indices ───────────────────────────────
// (Shared kernels reuse decode_abi.hpp's bind:: enums verbatim — same layouts.)
namespace bind {

// Dense bf16 GEMV (M=1 decode matvec): out[N] = W[N,K] @ x[K].  The gemma4 linear
// kernel (q/k/v/o/gate/up/down + tied embed^T logits) — no dequant (cf. qwen3.6's
// affine_qmv). group=(32,2,1), grid=(1, ceil(N/8), 1); float accumulation.
enum class Gemv : uint8_t { W = 0, X = 1, Out = 2, K = 3, N = 4 };

// GeGLU-tanh: out = gelu_tanh(gate) * up.  gate/up are the two GEMV outputs.
//   gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
enum class Geglu : uint8_t { Gate = 0, Up = 1, Out = 2, N = 3 };

// Weightless RMSNorm (gemma4 V-norm before the KV write): out = x / rms(x).
enum class VNorm : uint8_t { X = 0, Out = 1, Axis = 2, Eps = 3 };

// PLE precompute — projection norm over each per_layer_emb_dim row.
// (The gather + GEMV + combine reuse Embed / Qmv; this is the per-256-row rms.)
enum class PleProjNorm : uint8_t { X = 0, W = 1, Out = 2, RowDim = 3, Eps = 4 };

// PLE combine: ple = (proj + token) * (1/sqrt(2)).  Elementwise over [L*ple_dim].
enum class PleCombine : uint8_t { Proj = 0, Token = 1, Out = 2, InvSqrt2 = 3, N = 4 };

// final logit softcap: out = cap * tanh(logits / cap).  In-place over [vocab].
enum class Softcap : uint8_t { Logits = 0, Out = 1, Cap = 2, N = 3 };

// layer_scalar broadcast multiply: hidden *= scalar[0]  (over [hidden]).
enum class LayerScalar : uint8_t { X = 0, Scalar = 1, Out = 2, N = 3 };

}  // namespace bind

// ── gemma4 dispatch-DAG kernel kinds (per-step encode order) ──────────────────
// Layer-less PLE precompute, then ×35 decoder layers (attn norm-sandwich +
// GeGLU FFN + PLE-residual + layer_scalar), then final norm + tied lm_head +
// softcap + argmax. Sliding layers use Sdpa with a 512-window; KV-shared layers
// (15..34) skip Qmv{K,V}/VNorm/RopeK/KvAppend and read the source layer's pages.
enum class Kernel : uint8_t {
    // ── PLE precompute (layer-less, once per step) ──
    EmbedGather,       // embed_tokens gather (x sqrt(hidden))
    PleTokenGather,    // embed_tokens_per_layer gather (x sqrt(ple_dim))
    PleProjGemv,       // per_layer_model_projection GEMV (x 1/sqrt(hidden))
    PleProjNorm,       // rms over each ple_dim row
    PleCombine,        // (proj + token) * 1/sqrt(2)

    // ── per-layer attention (norm sandwich; qk-norm; sliding/full; kv-share) ──
    AttnNorm,          // input_layernorm (rms)
    QmvQ, QmvK, QmvV,  // q/k/v projections
    QNorm, KNorm,      // per-head q/k RMSNorm (weighted)
    VNorm,             // weightless V-norm before KV write
    RopeQ, RopeK,      // partial/full rope (per-layer theta)
    KvAppend,          // in-place KV write (skipped on shared layers)
    Sdpa,              // decode attention (sliding-window or full)
    QmvO,              // o_proj
    PostAttnNorm,      // post_attention_layernorm (rms)
    AttnResidual,      // hidden += attn_out

    // ── per-layer FFN (GeGLU-tanh) ──
    FfnNorm,           // pre_feedforward_layernorm (rms)
    QmvGate, QmvUp,    // gate/up projections
    GegluTanh,         // gelu_tanh(gate) * up
    QmvDown,           // down projection
    PostFfnNorm,       // post_feedforward_layernorm (rms)
    FfnResidual,       // hidden += ffn_out

    // ── per-layer PLE residual + layer scalar ──
    PleGateGemv,       // per_layer_input_gate GEMV
    PleGeglu,          // gelu_tanh(gate) * ple_signal
    PleProjLayerGemv,  // per_layer_projection GEMV
    PleNorm,           // post_per_layer_input_norm (rms)
    PleResidual,       // hidden += ple_out
    LayerScalar,       // hidden *= layer_scalar[0]

    // ── tail (once per step) ──
    FinalRms,          // model.norm (rms)
    LmHead,            // logits = embed^T @ hidden (tied)
    FinalSoftcap,      // 30 * tanh(logits / 30)
    Argmax,            // optional device argmax (I3)
};

}  // namespace pie_metal_driver::raw_metal::gemma4
