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
#include "kernels_manifest.hpp"
#include "model/config.hpp"
#include "batch/forward_graph.hpp"

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
                                            int max_tokens,
                                            int max_requests) {
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
        attn_head_dim_instantiated(hf.head_dim_kernel);
    if (hf.num_attention_heads <= 0 || hf.num_key_value_heads <= 0 ||
        !supported_head_dim || prop.multiProcessorCount <= 0) {
        return base;
    }

    auto align_up = [](std::size_t n, std::size_t a) {
        return (n + (a - 1)) & ~(a - 1);
    };

    const int tp_size = std::max(1, cfg.distributed.tp_size);
    // DeepSeek-V4 REPLICATES its attention across TP ranks — `dsv4_shard_axis`
    // shards the FFN and nothing else, so every rank runs every head and
    // `DsV4Model::prepare` plans with `num_attention_heads` verbatim. Dividing
    // here undersized the float workspace by exactly `tp_size`, and the plan
    // then threw mid-decode ("Buffer overflow ... batch_prefill_tmp_v") the
    // moment a sequence grew long enough to need a second KV chunk. That
    // throw kills the TP followers while rank 0 keeps publishing fires, which
    // presents as an unrecoverable hang rather than an error.
    const bool attention_replicated_across_tp =
        hf.model_type == "deepseek_v4";
    const int attn_tp = attention_replicated_across_tp ? 1 : tp_size;
    // Per-rank head counts. tp cancels in tmp_v (qo_heads shrinks, padded_batch
    // grows as num_kv_heads shrinks), so this is tp-invariant — but compute it
    // honestly so a non-divisible split still produces a safe (larger) bound.
    const std::size_t qo_heads =
        std::max<std::size_t>(1, hf.num_attention_heads / attn_tp);
    const std::size_t kv_heads =
        std::max<std::size_t>(1, hf.num_key_value_heads / attn_tp);
    const std::size_t head_dim = static_cast<std::size_t>(hf.head_dim_kernel);
    const std::size_t num_sm =
        static_cast<std::size_t>(prop.multiProcessorCount);
    // Worst case over a batch's qo lengths (a long-prompt prefill).
    const std::size_t cta_tile_q = ::flashinfer::FA2DetermineCtaTileQ(
        std::numeric_limits<std::int64_t>::max(),
        static_cast<std::uint32_t>(head_dim));
    constexpr std::size_t num_blocks_per_sm = 2;  // literal in PrefillPlan
    std::size_t padded_batch =
        std::max<std::size_t>(1, (num_blocks_per_sm * num_sm) / kv_heads);

    // The floor above is `max_batch_size_if_split`, and it is the right bound
    // for the EAGER path only: outside graph-mode planning the chunk search
    // reports a carve only when the work-item count fits it, so it ceils
    // `padded_batch_size`. Graph-mode planning does not go through that search
    // -- it takes `padded_batch = max(floor, total_tiles)` with
    //
    //     total_tiles = (N*gqa + cta_tile_q - 1)/cta_tile_q + (R - 1)
    //
    // (`ops/attention_flashinfer.cu`), so the REQUEST COUNT dominates. At
    // R=256 that is ~7-9x this floor, the carve check fails, and the plan
    // demotes itself out of graph mode -- measured at 233 of 241
    // prefill-carrying waves on the S cell. The buffer, not the geometry, is
    // what refuses the graph.
    //
    // Sized here only when the prefill graph is armed, so the OFF arm of any
    // A/B allocates exactly what it allocated before.
    if (prefill_graph_enabled() && max_tokens > 0 && max_requests > 0) {
        const std::size_t gqa = std::max<std::size_t>(1, qo_heads / kv_heads);
        const std::size_t max_r = static_cast<std::size_t>(max_requests);
        const std::size_t max_n = static_cast<std::size_t>(max_tokens);
        const std::size_t total_tiles =
            (max_n * gqa + cta_tile_q - 1) / cta_tile_q + (max_r - 1);
        padded_batch = std::max(padded_batch, total_tiles);
    }

    const std::size_t tmp_v =
        qo_heads * padded_batch * cta_tile_q * head_dim * sizeof(float);
    const std::size_t tmp_s =
        qo_heads * padded_batch * cta_tile_q * sizeof(float);
    // Slack covers the plan's 16-byte allocation alignment and the decode
    // plan's (much smaller) tmp buffers, which share this buffer but never
    // coexist with a prefill.
    const std::size_t planned = tmp_v + tmp_s + 16ull * 1024 * 1024;
    return std::max(base, align_up(planned, 16ull * 1024 * 1024));
}

}  // namespace pie_cuda_driver
