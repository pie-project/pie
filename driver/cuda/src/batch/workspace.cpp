#include "batch/workspace.hpp"

#include <algorithm>

#include <cuda_runtime.h>

#include "config.hpp"
#include "model/config.hpp"

namespace pie_cuda_driver {

bool has_non_full_attention_layers(const HfConfig& hf) {
    for (const auto& kind : hf.layer_types) {
        if (kind != "full_attention") return true;
    }
    return false;
}

std::size_t attention_float_workspace_bytes(const HfConfig& hf,
                                            const Config& cfg,
                                            const cudaDeviceProp& prop,
                                            int /*max_requests*/) {
    const bool qwen_hybrid =
        hf.model_type == "qwen3_5" ||
        hf.model_type == "qwen3_5_text" ||
        hf.model_type == "qwen3_5_moe" ||
        hf.model_type == "qwen3_5_moe_text";
    const std::size_t base =
        qwen_hybrid ? 128ull * 1024 * 1024 : 80ull * 1024 * 1024;

    // FlashInfer's `PrefillPlan` (scheduler.cuh) carves the split-KV partial
    // outputs out of this float buffer:
    //
    //   tmp_v = num_qo_heads * padded_batch_size * cta_tile_q * head_dim * 4
    //   tmp_s = num_qo_heads * padded_batch_size * cta_tile_q     * 4
    //
    // `padded_batch_size` is the number of (q_tile, kv_chunk) work items.
    // PrefillBinarySearchKVChunkSize bounds it by `max_batch_size_if_split`,
    // and PrefillPlan hardcodes `num_blocks_per_sm = 2`, so per rank
    //
    //   max_batch_size_if_split = ceil(2 * num_sm / num_kv_heads_local).
    //
    // `cta_tile_q` (FA2DetermineCtaTileQ) tops out at 128 for head_dim < 256
    // (64 otherwise). This term scales with SM count and the model's head
    // config — both of which a flat default can't track — and is what
    // overflows the buffer on larger GPUs / low-KV-head models
    // (pie-project/pie#414). It is needed for *every* arch that hits the
    // prefill kernel, including GQA ratios outside the decode fast-path
    // (force_prefill_path) and sliding-window layouts, so it is sized
    // unconditionally rather than gated on the decode fast-path.
    const bool supported_head_dim =
        hf.head_dim_kernel == 64 || hf.head_dim_kernel == 128 ||
        hf.head_dim_kernel == 256 || hf.head_dim_kernel == 512;
    if (hf.num_attention_heads <= 0 || hf.num_key_value_heads <= 0 ||
        !supported_head_dim || prop.multiProcessorCount <= 0) {
        return base;
    }

    auto align_up = [](std::size_t n, std::size_t a) {
        return (n + (a - 1)) & ~(a - 1);
    };
    auto ceil_div = [](std::size_t n, std::size_t d) {
        return (n + d - 1) / d;
    };

    const int tp_size = std::max(1, cfg.distributed.tp_size);
    // Per-rank head counts. tp cancels in tmp_v (qo_heads shrinks, padded_batch
    // grows as num_kv_heads shrinks), so this is tp-invariant — but compute it
    // honestly so a non-divisible split still produces a safe (larger) bound.
    const std::size_t qo_heads =
        std::max<std::size_t>(1, hf.num_attention_heads / tp_size);
    const std::size_t kv_heads =
        std::max<std::size_t>(1, hf.num_key_value_heads / tp_size);
    const std::size_t head_dim = static_cast<std::size_t>(hf.head_dim_kernel);
    const std::size_t num_sm =
        static_cast<std::size_t>(prop.multiProcessorCount);
    const std::size_t cta_tile_q = (head_dim < 256) ? 128 : 64;
    const std::size_t padded_batch = ceil_div(2 * num_sm, kv_heads);

    const std::size_t tmp_v =
        qo_heads * padded_batch * cta_tile_q * head_dim * sizeof(float);
    const std::size_t tmp_s =
        qo_heads * padded_batch * cta_tile_q * sizeof(float);
    // Slack covers alignment padding and the decode plan's (much smaller)
    // tmp buffers, which share this buffer but never coexist with a prefill.
    const std::size_t planned = tmp_v + tmp_s + 16ull * 1024 * 1024;
    return std::max(base, align_up(planned, 16ull * 1024 * 1024));
}

}  // namespace pie_cuda_driver
