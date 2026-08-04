#pragma once

// Kimi-K3 — a hybrid KDA/MLA decoder with a latent MoE and attention
// residuals. Four things make it its own family rather than a Kimi-K2 or a
// Qwen3.5-MoE variant:
//
//   * Hybrid attention. Three of every four layers run **KDA** (Kimi Delta
//     Attention): a gated delta-rule recurrence whose decay is *per key
//     channel*, not per head like Qwen3.5's Gated DeltaNet. The fourth runs
//     **MLA with no rotation at all** (`mla_use_nope`) plus a sigmoid output
//     gate. The two kinds keep different state: KDA a [H, V, K] recurrent
//     slab plus three conv windows, MLA a paged compressed-latent KV cache.
//
//   * Latent MoE. The routed experts do not run at `hidden_size`: the block
//     projects 7168 -> `routed_expert_hidden_size` (3584), runs the experts
//     there, RMSNorms, and projects back. The router and the shared expert
//     still read the full-width hidden.
//
//   * Attention residuals. Every layer blends its running residual with a
//     stack of saved "block" residuals through a softmax; a new block is
//     pushed every `attn_res_block_size` layers. See `kimi_k3_forward.hpp`.
//
//   * SiTU activation instead of SwiGLU, in both the dense MLP and the
//     experts.
//
// HF spelling (`language_model.model.layers.{L}.`):
//
//   KDA layer   : self_attn.{q,k,v}_proj, self_attn.{q,k,v}_conv1d,
//                 self_attn.{A_log,dt_bias,b_proj,f_a_proj,f_b_proj,o_norm,
//                            g_proj,o_proj}
//   MLA layer   : self_attn.{q_a_proj,q_a_layernorm,q_b_proj,
//                            kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,
//                            g_proj,o_proj}
//   Dense MLP   : mlp.{gate,up,down}_proj                  (layer 0 only)
//   MoE         : block_sparse_moe.{gate.weight,
//                                   gate.e_score_correction_bias,
//                                   routed_expert_down_proj,
//                                   routed_expert_norm,
//                                   routed_expert_up_proj,
//                                   experts.{e}.w{1,2,3},
//                                   shared_experts.{gate,up,down}_proj}
//   AttnRes     : self_attention_res_{norm,proj}, mlp_res_{norm,proj}
//                 and model.output_attn_res_{norm,proj} at the tail
//
// The routed experts ship MXFP4 (`weight_packed` U8 + `weight_scale` E8M0
// over groups of 32); everything else is bf16.

#include <cstdint>
#include <optional>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

/// True when the routed `gate_up` stack is materialised as `[up | gate]`
/// instead of HF's `[gate | up]` -- the order flashinfer's grouped GEMM reads
/// fc1 in. The contract picks the order and every consumer asks here.
bool kimi_k3_moe_gate_up_swapped();

struct KimiK3LayerWeights {
    enum class Kind { Kda, Mla };
    Kind kind = Kind::Kda;

    // ── Norms ───────────────────────────────────────────────────────
    const DeviceTensor* attn_norm = nullptr;   // input_layernorm [H]
    const DeviceTensor* mlp_norm  = nullptr;   // post_attention_layernorm [H]

    // ── Attention residuals (present on every layer) ───────────────
    // `*_res_proj` is [1, H]: one score row, so it is replicated across TP
    // ranks and the blend runs on the full-width residual.
    const DeviceTensor* attn_res_norm = nullptr;   // [H]
    const DeviceTensor* attn_res_proj = nullptr;   // [1, H]
    const DeviceTensor* mlp_res_norm  = nullptr;   // [H]
    const DeviceTensor* mlp_res_proj  = nullptr;   // [1, H]

    // ── KDA (Kind::Kda) ────────────────────────────────────────────
    // Widths are per rank: `local_heads = num_heads / tp_size`.
    const DeviceTensor* kda_q_proj = nullptr;   // [local_heads*D, H]
    const DeviceTensor* kda_k_proj = nullptr;   // [local_heads*D, H]
    const DeviceTensor* kda_v_proj = nullptr;   // [local_heads*D, H]
    const DeviceTensor* kda_q_conv1d = nullptr; // [local_heads*D, 1, conv_k]
    const DeviceTensor* kda_k_conv1d = nullptr;
    const DeviceTensor* kda_v_conv1d = nullptr;
    // Decay gate: g = f_b_proj(f_a_proj(x)), then
    //   gate = lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias))
    // `f_a_proj` is a rank-`head_dim` bottleneck shared by every head, so it
    // stays replicated; `f_b_proj`/`dt_bias` shard with the heads.
    const DeviceTensor* kda_f_a_proj = nullptr; // [D, H] (replicated)
    const DeviceTensor* kda_f_b_proj = nullptr; // [local_heads*D, D]
    const DeviceTensor* kda_b_proj   = nullptr; // [local_heads, H]
    const DeviceTensor* kda_g_proj   = nullptr; // [local_heads*D, H]
    const DeviceTensor* kda_o_proj   = nullptr; // [H, local_heads*D]
    // fp32 companions the recurrence and the gated RMSNorm consume directly.
    // `A_log` ships as F32[128] even though only the first `num_heads`=96
    // entries mean anything -- the tail is storage padding, and reading it
    // as the head count is what makes a TP split land on the right heads.
    const float* kda_A_log_fp32  = nullptr;     // [local_heads]
    const float* kda_dt_bias_fp32 = nullptr;    // [local_heads*D]
    const float* kda_o_norm_fp32 = nullptr;     // [D] (replicated)

    // ── MLA (Kind::Mla) ────────────────────────────────────────────
    const DeviceTensor* q_a_proj = nullptr;             // [q_lora, H]
    const DeviceTensor* q_a_norm = nullptr;             // [q_lora]
    const DeviceTensor* q_b_proj = nullptr;             // [local_heads*(nope+rope), q_lora]
    const DeviceTensor* kv_a_proj_with_mqa = nullptr;   // [kv_lora+rope, H]
    const DeviceTensor* kv_a_norm = nullptr;            // [kv_lora]
    const DeviceTensor* kv_b_proj = nullptr;            // [local_heads*(nope+v), kv_lora]
    const DeviceTensor* mla_g_proj = nullptr;           // [local_heads*v, H]
    const DeviceTensor* o_proj = nullptr;               // [H, local_heads*v]

    // ── MLP ────────────────────────────────────────────────────────
    bool is_moe = false;

    // Dense SiTU MLP (layer 0 only on K3).
    const DeviceTensor* dense_gate_proj = nullptr;   // [I, H]
    const DeviceTensor* dense_up_proj   = nullptr;   // [I, H]
    const DeviceTensor* dense_down_proj = nullptr;   // [H, I]

    // Latent MoE. The router reads the *full-width* hidden, so it is
    // replicated; the latent projections shard on the latent axis.
    const DeviceTensor* router = nullptr;                   // [E, H]
    const DeviceTensor* e_score_correction_bias = nullptr;  // [E]
    const DeviceTensor* routed_down_proj = nullptr;         // [L, H]
    const DeviceTensor* routed_norm = nullptr;              // [L] or null
    const DeviceTensor* routed_up_proj = nullptr;           // [H, L]
    // Routed experts, dequantised and stacked per layer. Required: the binder
    // refuses a checkpoint the contract could not stack, because the prefill
    // GEMM has no per-expert fallback.
    const DeviceTensor* moe_gate_up_bf16 = nullptr;  // [E, 2*I_moe, L] bf16
    const DeviceTensor* moe_down_bf16    = nullptr;  // [E, L, I_moe] bf16
    // The MXFP4 originals, per expert, for the decode GEMV. Decode reads the
    // whole bank for one token, so the four-bit form is four times cheaper
    // there; the bf16 stacks above are what the prefill GEMMs want.
    DeviceBuffer<const std::uint8_t*> expert_gate_up_packed_ptrs;
    DeviceBuffer<const std::uint8_t*> expert_gate_up_scale_ptrs;
    DeviceBuffer<const std::uint8_t*> expert_down_packed_ptrs;
    DeviceBuffer<const std::uint8_t*> expert_down_scale_ptrs;

    // Shared expert: a full-width SiTU MLP over the pre-down-proj hidden.
    const DeviceTensor* shared_gate_proj = nullptr;      // [I_shared, H]
    const DeviceTensor* shared_up_proj   = nullptr;      // [I_shared, H]
    const DeviceTensor* shared_down_proj = nullptr;      // [H, I_shared]

    int kv_layer = -1;   // index into the MLA cache; -1 on KDA layers
    int kda_layer = -1;  // index into the recurrent state cache; -1 on MLA
};

struct KimiK3Weights {
    const DeviceTensor* embed      = nullptr;
    const DeviceTensor* lm_head    = nullptr;
    const DeviceTensor* final_norm = nullptr;

    // The tail AttnRes blend, applied to the last layer's output before
    // `final_norm`.
    const DeviceTensor* output_res_norm = nullptr;   // [H]
    const DeviceTensor* output_res_proj = nullptr;   // [1, H]

    int embed_tp_vocab_offset = 0;
    bool embed_tp_sharded = false;
    bool lm_head_tp_sharded = false;

    std::vector<KimiK3LayerWeights> layers;

    // fp32 copies materialised at bind time (A_log / dt_bias / o_norm ship
    // as fp32 on disk today, but the TP slice still needs its own storage).
    std::vector<DeviceBuffer<float>> owned_fp32_buffers;
    // Hand-sliced bf16 tensors whose layout does not shard under the
    // contract's uniform axis policy.
    std::vector<DeviceTensor> owned_bf16_buffers;
};

KimiK3Weights bind_kimi_k3(const LoadedModel& engine);

}  // namespace pie_cuda_driver::model
