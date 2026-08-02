#pragma once

// GLM-5.1 (`glm_moe_dsa`) decoder weights.
//
// Architecture: DeepSeek-V3 / Kimi-K2 style MLA attention + routed/shared
// MoE, plus a DSA indexer for sparse attention. This first-pass binding
// loads the MLA + MoE weights and skips the DSA indexer entirely — the
// forward pass falls back to full MLA attention, which is acceptable for
// short prompts.
//
// FP8 quantization: the released GLM-5.1 checkpoints store every dense
// projection (MLA Q/KV, routed/shared experts, MoE) as FP8_E4M3 with a
// companion `weight_scale_inv` per-channel tensor. We carry an optional
// `QuantMeta` alongside each weight pointer; the forward path picks the
// right `ops::WeightView` via `make_weight_view` (see qwen3.hpp).

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "model/llama_like/qwen3.hpp"  // for make_weight_view
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct Glm5ExpertWeights {
    const DeviceTensor* gate_proj = nullptr;
    const DeviceTensor* up_proj   = nullptr;
    const DeviceTensor* down_proj = nullptr;
    std::optional<QuantMeta> gate_quant;
    std::optional<QuantMeta> up_quant;
    std::optional<QuantMeta> down_quant;
};

struct Glm5LayerWeights {
    // RMSNorm weights (1D, [hidden] each).
    const DeviceTensor* attn_norm = nullptr;   // input_layernorm
    const DeviceTensor* mlp_norm  = nullptr;   // post_attention_layernorm

    // MLA projections.
    const DeviceTensor* q_a_proj = nullptr;             // [q_lora_rank, H]
    const DeviceTensor* q_a_norm = nullptr;             // [q_lora_rank]
    const DeviceTensor* q_b_proj = nullptr;             // [local_heads*(nope+rope), q_lora_rank]
    const DeviceTensor* kv_a_proj_with_mqa = nullptr;   // [kv_lora_rank+rope, H]
    const DeviceTensor* kv_a_norm = nullptr;            // [kv_lora_rank]
    const DeviceTensor* kv_b_proj = nullptr;            // [local_heads*(nope+v), kv_lora_rank]
    const DeviceTensor* o_proj    = nullptr;            // [H, local_heads*v_dim]

    std::optional<QuantMeta> q_a_proj_quant;
    std::optional<QuantMeta> q_b_proj_quant;
    std::optional<QuantMeta> kv_a_proj_with_mqa_quant;
    std::optional<QuantMeta> o_proj_quant;

    // ── DSA (DeepSeek Sparse Attention) "lightning indexer" ──────────────
    // Selects the top-`index_topk` keys per query that the main MLA attends
    // to. For seq_len <= index_topk it reduces to dense attention.
    //   idx_wq_b:        [index_n_heads*index_head_dim, q_lora_rank] FP8
    //   idx_wk:          [index_head_dim, H]                         FP8
    //   idx_weights_proj:[index_n_heads, H]                          BF16
    //   idx_k_norm:      LayerNorm([index_head_dim]) weight + bias   BF16
    const DeviceTensor* idx_wq_b = nullptr;
    const DeviceTensor* idx_wk = nullptr;
    const DeviceTensor* idx_weights_proj = nullptr;
    const DeviceTensor* idx_k_norm_weight = nullptr;
    const DeviceTensor* idx_k_norm_bias = nullptr;
    std::optional<QuantMeta> idx_wq_b_quant;
    std::optional<QuantMeta> idx_wk_quant;

    bool is_moe = false;

    // Dense MLP (li < first_k_dense_replace, i.e. layers 0..2 on GLM-5.1).
    const DeviceTensor* dense_gate_proj = nullptr;
    const DeviceTensor* dense_up_proj   = nullptr;
    const DeviceTensor* dense_down_proj = nullptr;
    std::optional<QuantMeta> dense_gate_quant;
    std::optional<QuantMeta> dense_up_quant;
    std::optional<QuantMeta> dense_down_quant;

    // When the checkpoint's dense gate/up are BF16 and TP==1, the loader
    // concatenates them into `mlp.gate_up_proj.fused.weight` and drops the
    // originals; `dense_gate_proj` / `dense_up_proj` then point at row views
    // owned here (see `bind_glm5`).
    std::unique_ptr<DeviceTensor> dense_gate_view;
    std::unique_ptr<DeviceTensor> dense_up_view;

    // Routed + shared MoE (layers >= first_k_dense_replace).
    const DeviceTensor* router = nullptr;                  // [E, H]  (typically bf16/fp32)
    const DeviceTensor* e_score_correction_bias = nullptr; // [E] or null
    std::vector<Glm5ExpertWeights> experts;                // size E

    // Contiguous BF16 expert stacks produced by the loader contract when the
    // checkpoint ships plain per-expert BF16 weights. When both are present the
    // forward takes the device-side aligned MoE path (no host round-trip);
    // otherwise it falls back to the per-expert loop over `experts`.
    const DeviceTensor* moe_gate_up_proj = nullptr;  // [E, 2*I_moe, H] bf16
    const DeviceTensor* moe_down_proj    = nullptr;  // [E, H, I_moe]   bf16

    const DeviceTensor* shared_gate_proj = nullptr;        // [I_shared, H]
    const DeviceTensor* shared_up_proj   = nullptr;        // [I_shared, H]
    const DeviceTensor* shared_down_proj = nullptr;        // [H, I_shared]
    std::optional<QuantMeta> shared_gate_quant;
    std::optional<QuantMeta> shared_up_quant;
    std::optional<QuantMeta> shared_down_quant;
};

struct Glm5Weights {
    const DeviceTensor* embed      = nullptr;
    const DeviceTensor* lm_head    = nullptr;
    const DeviceTensor* final_norm = nullptr;
    int embed_tp_vocab_offset = 0;
    bool embed_tp_sharded = false;
    bool lm_head_tp_sharded = false;

    std::vector<Glm5LayerWeights> layers;
};

Glm5Weights bind_glm5(const LoadedModel& engine);

/// True when the routed `gate_up` slab is published as `[up | gate]` -- the
/// order flashinfer's CUTLASS grouped GEMM reads fc1 in (`silu(w[I:]) * w[:I]`),
/// which is the mirror of pie's own `chunked_swiglu`. The contract author and
/// every consumer of `moe_gate_up_proj` must agree, so both ask here.
bool glm5_moe_gate_up_swapped();

}  // namespace pie_cuda_driver::model
