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

namespace pie_ggml_driver {

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
        std::int32_t  sampling_pos;   // legacy single-slot index (== sampling_positions[0])
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
    };

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
};

}  // namespace pie_ggml_driver
