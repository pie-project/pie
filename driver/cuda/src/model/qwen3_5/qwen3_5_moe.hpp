#pragma once

// Qwen3.6-MoE = Qwen3.5 hybrid (linear-attn + full-attn) + sparse-MoE
// MLP block on every layer. The MoE block has both routed experts and
// an always-on shared expert with its own per-token sigmoid gate.
//
// HF spelling: `model.language_model.layers.{L}.mlp` contains:
//   * `gate.weight`               : router [E, H]
//   * `experts.gate_up_proj`      : fused [E, 2*I_moe, H] (gate first half,
//                                   up second half along dim 1)
//   * `experts.down_proj`         : [E, H, I_moe]
//   * `shared_expert.{gate,up,down}_proj.weight` : standard SwiGLU MLP
//   * `shared_expert_gate.weight` : [1, H] — sigmoid gate for shared
//
// Forward (per token):
//     y_moe    = sum_k topk_w * silu(gate_e(x)) * up_e(x), then down_e(...)
//     y_shared = sigmoid(W_g x) * (down_s(silu(gate_s x) * up_s x))
//     y        = y_moe + y_shared
//
// The linear-attn / full-attn weights are identical to Qwen3_5; only the
// MLP block differs. Keeping a parallel struct rather than adding flags
// to Qwen3_5LayerWeights keeps each arch's invariants local.

#include <cstdint>
#include <optional>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

// True when the routed gate/up weights are stored in flashinfer's
// [linear|gate] order instead of HuggingFace's [gate|up]. Decided once from
// the environment; the bind reorders the weight and the swiglu kernels are
// told which order they read.
bool qwen35_moe_gate_up_swapped();

/// True when the speculative head reads `lm_head` as int8 rather than bf16.
///
/// Opt-in (`PIE_QWEN35_MTP_INT8_LM_HEAD`): the draft step's argmax GEMV is
/// memory-bound over the whole vocabulary, so halving the head pays for the
/// accuracy the draft can afford to lose -- the verifier re-scores every token
/// anyway. Read by the contract, which is what publishes the int8 view, and by
/// the bind, which reads whichever views the contract produced.
bool qwen35_mtp_int8_lm_head_enabled();

/// True when the shared expert's scalar gate is folded into the fused gate/up
/// slab as one extra row, giving `[2*Is+1, H]` instead of `[2*Is, H]`.
///
/// Opt-in (`PIE_QWEN35_FUSED_SHARED_SCALAR_GATE`): measured a wash to a loss --
/// the split path wins despite the extra launch, see the definition. Read by
/// the contract, which is what decides the slab's shape, and by the bind, which
/// reads whichever slab the contract produced.
bool qwen35_fused_shared_scalar_gate_enabled();

struct Qwen3_5MoeLayerWeights {
    enum class Kind { LinearAttn, FullAttn };
    Kind kind;

    // ── Norms ──────────────────────────────────────────────────────
    const DeviceTensor* attn_norm_pre = nullptr;
    const DeviceTensor* mlp_norm_pre  = nullptr;

    // ── Linear-attn weights ────────────────────────────────────────
    const DeviceTensor* la_in_proj_qkv = nullptr;
    const DeviceTensor* la_in_proj_z   = nullptr;
    const DeviceTensor* la_in_proj_b   = nullptr;
    const DeviceTensor* la_in_proj_a   = nullptr;
    const DeviceTensor* la_conv1d_w    = nullptr;
    const DeviceTensor* la_conv1d_b    = nullptr;
    const DeviceTensor* la_dt_bias     = nullptr;
    // The recurrent kernel and RMSNormGated need fp32 inputs for these
    // two tensors. Qwen3.5-4B ships them as fp32 on disk but
    // Qwen3.6-35B-A3B ships them as bf16; `gdn_fp32_parameters` states the
    // widening on the contract, so these point straight at the arena.
    const float* la_A_log_fp32  = nullptr;  // [V_h]
    const float* la_norm_w_fp32 = nullptr;  // [head_v_dim]
    const DeviceTensor* la_out_proj    = nullptr;

    // ── Full-attn weights ──────────────────────────────────────────
    const DeviceTensor* fa_q_proj = nullptr;
    const DeviceTensor* fa_k_proj = nullptr;
    const DeviceTensor* fa_v_proj = nullptr;
    const DeviceTensor* fa_o_proj = nullptr;
    const DeviceTensor* fa_q_norm = nullptr;
    const DeviceTensor* fa_k_norm = nullptr;

    // ── Sparse-MoE block ───────────────────────────────────────────
    const DeviceTensor* moe_router        = nullptr;  // [E, H] bf16
    const DeviceTensor* moe_gate_up_proj  = nullptr;  // [E, 2*I_moe, H] bf16
    const DeviceTensor* moe_down_proj     = nullptr;  // [E, H, I_moe] bf16

    // Streamed routed experts. Set instead of the two pointers above when the
    // contract declared the experts as a group: there is no fused slab to
    // index, so the per-expert path asks the cache for one expert at a time.
    GroupStreamCache* expert_cache = nullptr;
    std::size_t expert_group = 0;

    // Shared expert (standard SwiGLU MLP, intermediate = shared_I)
    const DeviceTensor* shared_gate_proj  = nullptr;  // [I_shared, H]
    const DeviceTensor* shared_up_proj    = nullptr;  // [I_shared, H]
    const DeviceTensor* shared_gate_up_proj = nullptr;  // [2*I_shared, H]
    const DeviceTensor* shared_gate_up_gate_proj = nullptr;  // [2*I_shared + 1, H]
    const DeviceTensor* shared_down_proj  = nullptr;  // [H, I_shared]
    const DeviceTensor* shared_gate       = nullptr;  // [1, H]

    // Optional QuantMeta companions for runtime-quantized 2-D projections.
    // Routed experts are fused 3-D tables and stay bf16 on this path.
    std::optional<QuantMeta> fa_q_proj_quant;
    std::optional<QuantMeta> fa_k_proj_quant;
    std::optional<QuantMeta> fa_v_proj_quant;
    std::optional<QuantMeta> fa_o_proj_quant;
    std::optional<QuantMeta> shared_gate_proj_quant;
    std::optional<QuantMeta> shared_up_proj_quant;
    std::optional<QuantMeta> shared_down_proj_quant;
    std::optional<QuantMeta> shared_gate_quant;

    int kv_layer = -1;  // -1 on linear-attn layers
};

struct Qwen3_5MoeWeights {
    const DeviceTensor* embed      = nullptr;
    const DeviceTensor* lm_head    = nullptr;
    const DeviceTensor* final_norm = nullptr;

    std::vector<Qwen3_5MoeLayerWeights> layers;

    // Owned bf16 copies of per-rank-sliced linear-attn weights and
    // routed-expert weights. Same role as in Qwen3_5Weights — these
    // tensors have block / fused layouts that don't shard cleanly under
    // uniform axis-0 partitioning, so we slice them by hand at bind time.

    struct MtpWeights {
        const DeviceTensor* pre_fc_norm_embedding = nullptr;
        const DeviceTensor* pre_fc_norm_hidden = nullptr;
        const DeviceTensor* fc = nullptr;
        const DeviceTensor* norm = nullptr;
        const DeviceTensor* embed = nullptr;
        Qwen3_5MoeLayerWeights layer;
    };
    std::optional<MtpWeights> mtp;
};

Qwen3_5MoeWeights bind_qwen3_5_moe(const LoadedModel& engine);

}  // namespace pie_cuda_driver::model
