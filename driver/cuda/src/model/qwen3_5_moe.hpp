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
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

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
    // Qwen3.6-35B-A3B ships them as bf16; bind materialises fp32 copies
    // (owned in `Qwen3_5MoeWeights::owned_fp32_buffers`) so the kernel
    // signature stays uniform.
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

    // Shared expert (standard SwiGLU MLP, intermediate = shared_I)
    const DeviceTensor* shared_gate_proj  = nullptr;  // [I_shared, H]
    const DeviceTensor* shared_up_proj    = nullptr;  // [I_shared, H]
    const DeviceTensor* shared_down_proj  = nullptr;  // [H, I_shared]
    const DeviceTensor* shared_gate       = nullptr;  // [1, H]

    int kv_layer = -1;  // -1 on linear-attn layers
};

struct Qwen3_5MoeWeights {
    const DeviceTensor* embed      = nullptr;
    const DeviceTensor* lm_head    = nullptr;
    const DeviceTensor* final_norm = nullptr;

    std::vector<Qwen3_5MoeLayerWeights> layers;

    // Owned fp32 copies of A_log and RMSNormGated.weight materialised at
    // bind time (Qwen3.6-35B-A3B ships them as bf16 even though the FLA
    // path consumes them in fp32). One pair per linear-attn layer.
    std::vector<DeviceBuffer<float>> owned_fp32_buffers;

    // Owned bf16 copies of per-rank-sliced linear-attn weights and
    // routed-expert weights. Same role as in Qwen3_5Weights — these
    // tensors have block / fused layouts that don't shard cleanly under
    // uniform axis-0 partitioning, so we slice them by hand at bind time.
    std::vector<DeviceTensor> owned_bf16_buffers;
};

Qwen3_5MoeWeights bind_qwen3_5_moe(LoadedModel& engine);

}  // namespace pie_cuda_driver::model
