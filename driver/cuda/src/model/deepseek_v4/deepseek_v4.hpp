#pragma once

// DeepSeek-V4 decoder weights.
//
// V4 differs fundamentally from V3/Kimi:
//   * Hypercompressed (HC) mixing: hc_mult parallel streams per layer
//   * Compressed attention: per-layer compress_ratios (0=SWA, 4=C4, 128=C128)
//   * Grouped output LoRA: wo_a with o_groups batched BMM
//   * MXFP4 routed experts, block-scaled FP8 dense projections
//   * sqrtsoftplus MoE routing + hash routing for early layers

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver { class GroupStreamCache; }

namespace pie_cuda_driver::model {

struct DsV4CompressorWeights {
    const DeviceTensor* ape  = nullptr;   // [compress_ratio, coff*head_dim] F32
    const DeviceTensor* norm = nullptr;   // [dim] BF16
    const DeviceTensor* wkv  = nullptr;   // [dim, H] BF16
    const DeviceTensor* wgate = nullptr;  // [dim, H] BF16
};

struct DsV4IndexerWeights {
    const DeviceTensor* wq_b       = nullptr;  // [index_n_heads*index_head_dim, q_lora_rank] FP8
    const DeviceTensor* wq_b_scale = nullptr;  // block scale E8M0
    const DeviceTensor* weights_proj = nullptr; // [index_n_heads, H] BF16
    DsV4CompressorWeights compressor;
};

struct DsV4ExpertWeights {
    const DeviceTensor* w1       = nullptr;  // gate [moe_I, H/2] I8 (MXFP4 packed)
    const DeviceTensor* w1_scale = nullptr;  // E8M0 per-32 scales
    const DeviceTensor* w2       = nullptr;  // down [H, moe_I/2] I8 (MXFP4 packed)
    const DeviceTensor* w2_scale = nullptr;
    const DeviceTensor* w3       = nullptr;  // up [moe_I, H/2] I8 (MXFP4 packed)
    const DeviceTensor* w3_scale = nullptr;
};

struct DsV4LayerWeights {
    const DeviceTensor* attn_norm = nullptr;   // [H]
    const DeviceTensor* ffn_norm  = nullptr;   // [H]

    // Attention projections (block-scaled FP8)
    const DeviceTensor* wq_a       = nullptr;  // [q_lora_rank, H]
    const DeviceTensor* wq_a_scale = nullptr;
    std::optional<QuantMeta> wq_a_quant;
    const DeviceTensor* wq_b       = nullptr;  // [num_heads*head_dim, q_lora_rank]
    const DeviceTensor* wq_b_scale = nullptr;
    std::optional<QuantMeta> wq_b_quant;
    const DeviceTensor* q_norm     = nullptr;  // [q_lora_rank]
    const DeviceTensor* wkv        = nullptr;  // [head_dim, H]
    const DeviceTensor* wkv_scale  = nullptr;
    std::optional<QuantMeta> wkv_quant;
    const DeviceTensor* kv_norm    = nullptr;  // [head_dim]
    const DeviceTensor* wo_a       = nullptr;  // [o_groups*o_lora_rank, H]
    const DeviceTensor* wo_a_scale = nullptr;
    std::optional<QuantMeta> wo_a_quant;
    const DeviceTensor* wo_b       = nullptr;  // [H, o_groups*o_lora_rank]
    const DeviceTensor* wo_b_scale = nullptr;
    std::optional<QuantMeta> wo_b_quant;
    const DeviceTensor* attn_sink  = nullptr;  // [num_heads] F32

    // HC mixing parameters (F32)
    const DeviceTensor* hc_attn_fn    = nullptr;  // [mix_hc, hc_mult*H]
    const DeviceTensor* hc_attn_scale = nullptr;  // [3]
    const DeviceTensor* hc_attn_base  = nullptr;  // [mix_hc]
    const DeviceTensor* hc_ffn_fn     = nullptr;
    const DeviceTensor* hc_ffn_scale  = nullptr;
    const DeviceTensor* hc_ffn_base   = nullptr;

    // Per-layer compression config
    int compress_ratio = 0;

    // Compressor (C4/C128 layers only)
    DsV4CompressorWeights compressor;

    // Indexer (C4 layers only)
    DsV4IndexerWeights indexer;

    // MoE routing
    bool is_hash_layer = false;
    const DeviceTensor* router      = nullptr;  // [E, H] BF16
    const DeviceTensor* router_bias = nullptr;  // [E] F32
    const DeviceTensor* tid2eid     = nullptr;  // [vocab_size, K] I64

    // Routed experts (MXFP4)
    std::vector<DsV4ExpertWeights> experts;

    // BF16 dequant of the routed experts, stacked per layer. The packed MXFP4
    // weights cannot feed cuBLAS directly, and dequantising them every layer of
    // every step costs more bandwidth than the GEMMs themselves. Layout matches
    // the shared MoE kernels: [E, 2*moe_I, H] and [E, H, moe_I].
    //
    // Null when the contract left the experts packed, which is the whole
    // signal: `author_deepseek_v4_contract` decides between the two, and the
    // branches below read that decision off which weights it published rather
    // than off a flag carried alongside them.
    const DeviceTensor* moe_gate_up_bf16 = nullptr;  // [E, 2*moe_I, H] BF16
    const DeviceTensor* moe_down_bf16    = nullptr;  // [E, H, moe_I] BF16

    // Or a third form: not here at all, but in a slab this layer shares with
    // every other, holding whichever experts were routed to recently. Same
    // bf16 layout as the stacks and the same dequantize produced it -- only
    // one expert at a time instead of E, and only when a token asks.
    //
    // Non-null excludes the two above, the way they exclude each other.
    GroupStreamCache* expert_cache = nullptr;
    std::size_t expert_group = 0;

    // Shared expert (block-scaled FP8)
    const DeviceTensor* shared_w1       = nullptr;  // gate
    const DeviceTensor* shared_w1_scale = nullptr;
    std::optional<QuantMeta> shared_w1_quant;
    const DeviceTensor* shared_w2       = nullptr;  // down
    const DeviceTensor* shared_w2_scale = nullptr;
    std::optional<QuantMeta> shared_w2_quant;
    const DeviceTensor* shared_w3       = nullptr;  // up
    const DeviceTensor* shared_w3_scale = nullptr;
    std::optional<QuantMeta> shared_w3_quant;
};

struct DsV4Weights {
    const DeviceTensor* embed      = nullptr;
    const DeviceTensor* lm_head    = nullptr;
    const DeviceTensor* final_norm = nullptr;

    int embed_tp_vocab_offset = 0;
    bool embed_tp_sharded = false;
    bool lm_head_tp_sharded = false;

    // Global HC head
    const DeviceTensor* hc_head_fn    = nullptr;  // [hc_mult, hc_mult*H] F32
    const DeviceTensor* hc_head_scale = nullptr;  // [1] F32
    const DeviceTensor* hc_head_base  = nullptr;  // [hc_mult] F32

    std::vector<DsV4LayerWeights> layers;
};

DsV4Weights bind_deepseek_v4(const LoadedModel& engine);

}  // namespace pie_cuda_driver::model
