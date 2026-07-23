#include "batch/workspace.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

// The float section sized here is carved by flashinfer's `PrefillPlan`
// (split-KV partial outputs); including the scheduler couples this bound to
// the same header that consumes the buffer, so an upstream signature change
// breaks this translation unit instead of silently invalidating the bound
// (pie-project/pie#414 was exactly that drift, against a hand-copied
// formula). `FA2DetermineCtaTileQ` (utils.cuh) is called directly below.
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/utils.cuh>

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
                                            int max_requests,
                                            int max_forward_tokens,
                                            bool cuda_graphs) {
    const bool qwen_hybrid =
        hf.model_type == "qwen3_5" ||
        hf.model_type == "qwen3_5_text" ||
        hf.model_type == "qwen3_5_moe" ||
        hf.model_type == "qwen3_5_moe_text";
    const std::size_t base =
        qwen_hybrid ? 128ull * 1024 * 1024 : 80ull * 1024 * 1024;

    // FlashInfer's `PrefillPlan` (attention/scheduler.cuh) carves the
    // split-KV partial outputs out of this float buffer:
    //
    //   tmp_v = num_qo_heads * padded_batch_size * cta_tile_q * head_dim * 4
    //   tmp_s = num_qo_heads * padded_batch_size * cta_tile_q     * 4
    //
    // Outside CUDA-graph planning (how pie plans prefill today),
    // `PrefillBinarySearchKVChunkSize` only reports `split_kv` — and hence
    // a carve — when the chunked work-item count fits
    // `max_batch_size_if_split`, so that value also ceils the carve's
    // `padded_batch_size`. `PrefillPlan` computes it as
    //
    //   max_batch_size_if_split = (num_blocks_per_sm * num_sm) / num_kv_heads
    //
    // with `num_blocks_per_sm = 2` a literal in its body — the one number
    // this bound still mirrors by hand. `cta_tile_q` comes from
    // flashinfer's own `FA2DetermineCtaTileQ` (worst case over qo lengths)
    // so an upstream tile-policy change reaches this bound by
    // recompilation. The term scales with SM count and the model's head
    // config — what a flat default can't track, and what overflowed the
    // buffer on larger GPUs / low-KV-head models. It is needed for *every*
    // arch that hits the prefill kernel, including GQA ratios outside the
    // decode fast-path (force_prefill_path) and sliding-window layouts, so
    // it is sized unconditionally rather than gated on the decode
    // fast-path.
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
    // Worst case over a batch's qo lengths (a long-prompt prefill).
    const std::size_t cta_tile_q = ::flashinfer::FA2DetermineCtaTileQ(
        std::numeric_limits<std::int64_t>::max(),
        static_cast<std::uint32_t>(head_dim));
    constexpr std::size_t num_blocks_per_sm = 2;  // literal in PrefillPlan
    const std::size_t padded_batch =
        std::max<std::size_t>(1, (num_blocks_per_sm * num_sm) / kv_heads);

    // Graph-mode prefill planning (`enable_cuda_graph`) sizes the carve by
    //
    //   padded_batch_size = max(max_batch_size_if_split, total_num_tiles_q)
    //   total_num_tiles_q = ceil(total_num_rows * gqa / cta_tile_q)
    //                       + batch_size - 1
    //
    // (PrefillSplitQOKVIndptr) and forces split_kv, so the carve always
    // happens and scales with the token budget, not the CTA budget.
    // Budgeting it for the full (max_forward_tokens, max_requests)
    // envelope buys the graphed-prefill coverage; a wave whose plan would
    // not fit the granted buffer plans non-graph instead and runs eager.
    std::size_t graph_padded_batch = 0;
    if (cuda_graphs && max_forward_tokens > 0 && max_requests > 0) {
        auto ceil_div = [](std::size_t n, std::size_t d) {
            return (n + d - 1) / d;
        };
        const std::size_t gqa = ceil_div(qo_heads, kv_heads);
        const std::size_t total_tiles =
            ceil_div(static_cast<std::size_t>(max_forward_tokens) * gqa,
                     cta_tile_q) +
            static_cast<std::size_t>(max_requests) - 1;
        graph_padded_batch = std::max(padded_batch, total_tiles);
    }
    const std::size_t sized_batch = std::max(padded_batch, graph_padded_batch);

    const std::size_t tmp_v =
        qo_heads * sized_batch * cta_tile_q * head_dim * sizeof(float);
    const std::size_t tmp_s =
        qo_heads * sized_batch * cta_tile_q * sizeof(float);
    // Slack covers the plan's 16-byte allocation alignment and the decode
    // plan's (much smaller) tmp buffers, which share this buffer but never
    // coexist with a prefill.
    const std::size_t planned = tmp_v + tmp_s + 16ull * 1024 * 1024;
    return std::max(base, align_up(planned, 16ull * 1024 * 1024));
}

}  // namespace pie_cuda_driver
