#pragma once

// Qwen3.5 hybrid: linear-attention (Gated DeltaNet) on most layers,
// full-attention on every Nth layer. Layer mix is stored in
// HfConfig::layer_types as "linear_attention" / "full_attention".
//
// ── Linear-attention layer ────────────────────────────────────────
//   in_proj_qkv : H → 2*K_dim + V_dim   (K_dim = num_k_heads * head_k_dim,
//                                        V_dim = num_v_heads * head_v_dim)
//   in_proj_z   : H → V_dim             (gate, used by RMSNormGated)
//   in_proj_b   : H → num_v_heads       (β source)
//   in_proj_a   : H → num_v_heads       (g source, with dt_bias + A_log)
//   conv1d      : depthwise causal conv, kernel=4, channels = 2*K_dim+V_dim
//   norm        : RMSNormGated on head_v_dim
//   out_proj    : V_dim → H
//   dt_bias, A_log : per-v-head learned scalars
//
// ── Full-attention layer ──────────────────────────────────────────
//   Mostly Qwen3-style (q_norm, k_norm) with two twists:
//     • q_proj output is 2× wide; second half is the per-token gate
//       used as `attn_out *= sigmoid(gate)` before o_proj.
//     • RoPE has `partial_rotary_factor=0.25` — only the first 25%
//       of head_dim is rotated.

#include <cstdint>
#include <optional>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

// Forward-decl HfConfig members we touch. Using direct dependency for
// brevity rather than threading individual ints through the API.
struct Qwen3_5LayerWeights {
    enum class Kind { LinearAttn, FullAttn };
    Kind kind;

    // ── Norms (both kinds) ─────────────────────────────────────────
    const DeviceTensor* attn_norm_pre  = nullptr;  // input_layernorm
    const DeviceTensor* mlp_norm_pre   = nullptr;  // post_attention_layernorm

    // ── Linear-attn weights (kind == LinearAttn) ───────────────────
    const DeviceTensor* la_in_proj_qkv = nullptr;  // [2*K + V, H] bf16
    const DeviceTensor* la_in_proj_z   = nullptr;  // [V, H]       bf16
    const DeviceTensor* la_in_proj_b   = nullptr;  // [V_heads, H] bf16
    const DeviceTensor* la_in_proj_a   = nullptr;  // [V_heads, H] bf16
    const DeviceTensor* la_conv1d_w    = nullptr;  // [conv_dim, 1, K] bf16
    const DeviceTensor* la_conv1d_b    = nullptr;  // [conv_dim] bf16 (may be null)
    const DeviceTensor* la_dt_bias     = nullptr;  // [V_heads] bf16
    // The recurrent + RMSNormGated kernels consume these in fp32. HF
    // ships them as fp32 on Qwen3.5-4B and as bf16 on Qwen3.6-35B-A3B;
    // bind materialises a single fp32 copy either way, owned in
    // `Qwen3_5Weights::owned_fp32_buffers`.
    const float* la_A_log_fp32   = nullptr;        // [V_heads]
    const float* la_norm_w_fp32  = nullptr;        // [head_v_dim]
    const DeviceTensor* la_out_proj    = nullptr;  // [H, V] bf16

    // ── Full-attn weights (kind == FullAttn) ───────────────────────
    const DeviceTensor* fa_q_proj      = nullptr;  // [2*Hq*d, H] bf16
    const DeviceTensor* fa_k_proj      = nullptr;  // [Hkv*d, H] bf16
    const DeviceTensor* fa_v_proj      = nullptr;  // [Hkv*d, H] bf16
    const DeviceTensor* fa_o_proj      = nullptr;  // [H, Hq*d] bf16
    const DeviceTensor* fa_q_norm      = nullptr;  // [d] bf16 ((1+w) gemma-style)
    const DeviceTensor* fa_k_norm      = nullptr;  // [d] bf16

    // ── MLP (both kinds) ───────────────────────────────────────────
    const DeviceTensor* gate_proj = nullptr;  // [I, H] bf16
    const DeviceTensor* up_proj   = nullptr;  // [I, H] bf16
    const DeviceTensor* down_proj = nullptr;  // [H, I] bf16

    // Optional QuantMeta companions for the GEMM-fed projections —
    // populated when runtime_quant or an offline-quantized checkpoint
    // tags these weights via `LoadedModel::set_quant_meta`. Linear-attn
    // weights stay bf16 for now (their fused [K1|K2|V] block layout
    // needs per-block scale handling that isn't wired yet).
    std::optional<QuantMeta> fa_q_proj_quant;
    std::optional<QuantMeta> fa_k_proj_quant;
    std::optional<QuantMeta> fa_v_proj_quant;
    std::optional<QuantMeta> fa_o_proj_quant;
    std::optional<QuantMeta> gate_proj_quant;
    std::optional<QuantMeta> up_proj_quant;
    std::optional<QuantMeta> down_proj_quant;

    // KV cache slot for full-attn layers; -1 for linear-attn layers
    // (their state lives in the recurrent/conv state caches).
    int kv_layer = -1;
};

struct Qwen3_5Weights {
    const DeviceTensor* embed   = nullptr;      // [vocab, H] bf16
    const DeviceTensor* lm_head = nullptr;      // [vocab, H] bf16 (or alias of embed)
    const DeviceTensor* final_norm = nullptr;   // [H] bf16 ((1+w) gemma-style)

    std::vector<Qwen3_5LayerWeights> layers;

    // Owned fp32 copies of A_log + RMSNormGated.weight materialised at
    // bind time so the kernel signature stays uniform regardless of
    // whether HF ships them as fp32 (Qwen3.5-4B) or bf16 (Qwen3.6-MoE,
    // and likely other variants).
    std::vector<DeviceBuffer<float>> owned_fp32_buffers;

    // Owned bf16 copies of per-rank-sliced linear-attn weights (fused
    // in_proj_qkv [conv_dim_local, H], conv1d_w [conv_dim_local, 1, K],
    // conv1d_b [conv_dim_local]). Only populated when tp_size > 1 — the
    // engine loader stores these tensors replicated because the
    // [K1 | K2 | V] block layout doesn't shard cleanly under uniform
    // axis-0 partitioning, so we slice per-block here.
    std::vector<DeviceTensor> owned_bf16_buffers;
};

Qwen3_5Weights bind_qwen3_5(LoadedModel& engine);

}  // namespace pie_cuda_driver::model
