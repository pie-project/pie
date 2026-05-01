#pragma once

// Forward engine: builds a per-call ggml graph for Qwen3 (multi-request,
// page-table-aware), computes it on the model's backend, samples one token
// per sampler slot (greedy/argmax for now), and writes a `BPIS` flat
// response.
//
// M2 status:
//   - multiple requests per batch (each with its own context_id)
//   - paged KV pool honoring `kv_page_indices` / `kv_page_indptr` /
//     `kv_last_page_lens` from the BPIQ wire format
//   - one greedy sampler slot per request (full sampler suite lands in M4)
//   - no logit masks, no spec decode, no LoRA, no custom attention masks

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "adapter.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "shmem_schema.hpp"

namespace pie_portable_driver {

class ForwardEngine {
public:
    ForwardEngine(Model& model,
                  std::int32_t total_pages,
                  std::int32_t page_size);
    ~ForwardEngine();

    ForwardEngine(const ForwardEngine&) = delete;
    ForwardEngine& operator=(const ForwardEngine&) = delete;

    std::size_t run(const schema::DecodedRequest& req,
                    std::span<std::uint8_t> response);

    // Greedy generation harness — single context. Internally allocates a
    // contiguous block of pages starting at `page_offset` for this run.
    std::vector<std::uint32_t> generate(std::span<const std::uint32_t> prompt_tokens,
                                        std::int32_t max_new_tokens,
                                        std::uint64_t context_id,
                                        std::int32_t page_offset = 0);

    // Multi-context greedy generation. Allocates non-overlapping page
    // ranges per context.
    std::vector<std::vector<std::uint32_t>> generate_multi(
        std::vector<std::vector<std::uint32_t>>& prompts,
        std::int32_t max_new_tokens,
        std::vector<std::uint64_t> context_ids);

    std::int32_t total_pages() const noexcept { return kv_.total_pages(); }
    std::int32_t page_size()   const noexcept { return kv_.page_size(); }
    std::size_t  kv_buffer_size() const noexcept { return kv_.buffer_size(); }

    // Access the KV cache (for the M7 aux IPC server's page copies).
    KvCachePaged& kv() noexcept { return kv_; }

    // Wire in the adapter pool (M9). The engine looks up active adapters
    // per batch from this pool.
    void set_adapters(AdapterPool* pool) noexcept { adapters_ = pool; }

    // Per-batch plan exposed for the graph builder. One Req per request.
    struct ReqPlan {
        std::int32_t  qo_start;       // offset into the (expanded) flat token arrays
        std::int32_t  n_tokens;       // pending + drafts (after spec splice)
        std::int32_t  n_tokens_pad;   // GGML_PAD(n_tokens, 64)
        std::int32_t  n_kv;           // total KV positions to attend (= seq_len)
        std::vector<std::uint16_t> mask_f16;     // [n_kv, n_tokens_pad] F16
        std::vector<std::int32_t>  gather_idxs;  // [n_kv] physical KV row indices
        SamplerParams sampler;        // per-request sampler config (shared across slots)
        std::vector<std::uint32_t> logit_mask_runs;  // BRLE; empty = no mask

        // Per-slot sampling positions (offsets into the expanded batch).
        // Length 1 for plain sampling. For M8 spec decode: length = 1 +
        // n_drafts (slot 0 predicts the first draft position; subsequent
        // slots predict each draft's successor — last slot is the bonus).
        std::vector<std::int32_t> sampling_positions;
        // Draft tokens for verification (length = n_drafts; empty = no spec).
        std::vector<std::uint32_t> draft_tokens;
    };

    struct BatchPlan {
        std::int32_t                total_n_tokens = 0;     // includes drafts
        std::vector<std::int32_t>   tokens_i32;       // [total]
        std::vector<std::int32_t>   positions_i32;    // [total]
        std::vector<std::int64_t>   kv_idxs_i64;      // [total] write idxs
        // Flat list across all requests' sampler slots — size matches
        // the graph's `out_idx` input.
        std::vector<std::int32_t>   sampling_pos_i32;
        std::vector<ReqPlan>        reqs;

        // Multi-stream attention packing (M11 fast path). When every
        // request has exactly 1 query token AND no custom attention
        // masks are in play, attention can be expressed as a single
        // `ggml_flash_attn_ext` call per layer with `ne3 = n_request`.
        // We pre-build the packed gather idxs and packed mask host-side.
        bool                        pure_decode = false;
        std::int32_t                max_n_kv = 0;
        std::vector<std::int32_t>   packed_gather_idxs;  // [n_req * max_n_kv]
        std::vector<std::uint16_t>  packed_mask_f16;     // [max_n_kv, 64, 1, n_req]

        // M9 LoRA: active adapter for this whole batch (single-adapter-
        // per-batch v1 restriction). nullptr = no adapter.
        const Adapter*              active_adapter = nullptr;

        // GPU-greedy fast path. True iff every request's sampler is a
        // token-producing type at temperature ≤ 1e-5 (i.e. argmax) AND
        // there are no per-request logit masks. When set, the graph
        // builder emits a `tokens_out` int32 tensor via ggml_argmax and
        // compute_() downloads only those n_slots * 4 bytes instead of
        // the full [vocab_size, n_slots] F32 logits block.
        bool                        all_greedy = false;

        // GPU non-greedy fast path (uniform top-K). True iff:
        //   - not all_greedy (greedy takes precedence)
        //   - every slot has a token-producing sampler != Multinomial
        //   - every slot uses the same temperature (>1e-5)
        //   - no slot has a logit mask
        // When set, the graph emits top-K sorted [probs, indices] per
        // slot (size K * n_slots) and compute_() downloads ~K * n_slots
        // small tensors. Per-slot top-p / min-p / top-k cuts and the
        // categorical sample run on host over the tiny K-sized list.
        bool                        uniform_top_sample = false;
        // K used when uniform_top_sample is set (max of any per-slot
        // top_k, clamped against vocab_size; default 256 if no slot
        // specifies a top_k).
        std::int32_t                uniform_top_k = 0;
    };

    // Per-stage timing accumulators. Used both for offline benchmarks
    // and to validate optimization changes. Each counter accumulates
    // microseconds; `n_calls` counts compute_() invocations. Print on
    // demand (or at engine destruction when calls > 0).
    struct PhaseTimings {
        std::uint64_t plan_us         = 0;
        std::uint64_t graph_build_us  = 0;
        std::uint64_t graph_alloc_us  = 0;
        std::uint64_t upload_us       = 0;
        std::uint64_t compute_us      = 0;     // backend graph compute
        std::uint64_t logits_dl_us    = 0;
        std::uint64_t sample_us       = 0;
        std::uint64_t response_pack_us = 0;
        std::uint64_t total_us        = 0;     // wall time of run()
        std::uint64_t n_calls         = 0;
    };
    const PhaseTimings& timings() const noexcept { return timings_; }
    void reset_timings() noexcept { timings_ = {}; }
    void log_timings(const char* label) const;

private:
    BatchPlan plan_(const schema::DecodedRequest& req);
    BatchPlan plan_test_simple_(std::span<const std::uint32_t> token_ids,
                                std::span<const std::uint32_t> position_ids,
                                std::int32_t sampling_pos,
                                std::int32_t page_offset);
    std::vector<SamplerOutput> compute_(const BatchPlan& plan);

    Model&         model_;
    KvCachePaged   kv_;
    ggml_gallocr_t galloc_ = nullptr;
    AdapterPool*   adapters_ = nullptr;
    mutable PhaseTimings timings_;
};

}  // namespace pie_portable_driver
