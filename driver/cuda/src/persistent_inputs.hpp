#pragma once

// Pre-allocated input device buffers, sized once at startup for the
// configured worst case and reused across `fire_batch` invocations.
//
// Why: the previous design called `DeviceBuffer<T>::from_host(...)` per
// fire — six fresh `cudaMalloc`s each time the BPIQ control arrays were
// uploaded. That cost ~50 µs/fire on its own (small), but more
// importantly the freshly-malloc'd device addresses changed every fire
// → kernels couldn't be CUDA-graph-captured because the captured graph
// hardcodes pointer arguments at capture time. Replay with new
// pointers would read stale memory and produce gibberish.
//
// With this struct, the 9 BPIQ control arrays + sampled-token output
// live at stable addresses. The request handler refreshes their
// *contents* via `copy_from_host` / `copy_from_bytes` per fire and
// hands raw pointers to the forward + sampling kernels.
//
// All sizes are upper bounds derived from the configured workspace
// limits. Allocation is one-shot at engine init; deallocation is
// scope-driven (`DeviceBuffer<T>` is RAII).

#include <cstddef>
#include <cstdint>

#include "device_buffer.hpp"

namespace pie_cuda_driver {

struct PersistentInputs {
    // Per-token arrays (capacity = max_workspace_tokens).
    DeviceBuffer<std::uint32_t> tokens;
    DeviceBuffer<std::uint32_t> positions;
    DeviceBuffer<std::int32_t>  sampled;

    // Per-request arrays (capacity = max_requests + 1 for indptrs,
    // max_requests for last_page_lens).
    DeviceBuffer<std::uint32_t> qo_indptr;        // R+1
    DeviceBuffer<std::uint32_t> kv_page_indptr;   // R+1
    DeviceBuffer<std::uint32_t> kv_last_page_lens;// R

    // Page-id list (capacity = total kv-cache pages — the absolute
    // upper bound on how many pages a single fire could reference).
    DeviceBuffer<std::uint32_t> kv_page_indices;

    // Custom mask. `mask` is a packed bitmap; capacity is the worst
    // case ceil(max_qo * max_kv / 8). `indptr` has R+1 byte offsets.
    DeviceBuffer<std::uint8_t>  custom_mask;
    DeviceBuffer<std::int32_t>  custom_mask_indptr;

    // Per-request linear-attention state-cache slot ids (Qwen3.5 / 3.6).
    // Capacity = max_requests. Inert (zero-length writes) on archs that
    // don't use a state cache. Lives in PersistentInputs so the TP
    // broadcast from rank 0 reaches followers via the same NCCL op
    // pattern as the rest of the per-fire payload.
    DeviceBuffer<std::int32_t>  slot_ids;
    DeviceBuffer<std::uint8_t>  is_fresh;

    // Token-sampler scratch (rank 0 only — TP followers don't sample).
    // Replaces the per-fire `DeviceBuffer::from_host` allocations in
    // `sampling_dispatch.cpp`. Pre-allocated to `max_workspace_tokens`
    // so any fire's plan arrays (length N ≤ max_workspace_tokens) fit.
    // Per-slot scratch buffers (`sample_idx`, `per_sample_tok`, `valid`)
    // are sized identically — `num_sampling ≤ N` for the token-sampler
    // path. Refreshed per fire via `copy_from_host` (async H2D).
    DeviceBuffer<float>         sampling_temp;
    DeviceBuffer<float>         sampling_min_p;
    DeviceBuffer<float>         sampling_top_p;
    DeviceBuffer<std::int32_t>  sampling_top_k;
    DeviceBuffer<std::uint32_t> sampling_seed_u32;   // temp/min-p path
    DeviceBuffer<std::uint64_t> sampling_seed_u64;   // topk/topp path
    DeviceBuffer<std::int32_t>  sampling_sample_idx;
    DeviceBuffer<std::int32_t>  sampling_per_sample_tok;
    DeviceBuffer<bool>          sampling_valid;      // kernel scratch

    // Msgpack-subpass scratch (rank 0 only). The four subpasses
    // (entropy/logprob/dist/raw-logits) each previously did up to four
    // `DeviceBuffer::from_host` per fire when their slot type was
    // present. These persistent buffers replace the small per-row
    // arrays. The two genuinely large per-fire allocations stay
    // per-fire because they're conditional and size-proportional to
    // vocab_size: gather-packed bf16 logits (n × V × u16) and dist
    // probs (n × V × f32). Those slot types are absent from typical
    // text-completion bench loads.
    DeviceBuffer<std::int32_t>  subpass_rows;        // gather-rows index
    DeviceBuffer<std::int32_t>  subpass_lp_rows;
    DeviceBuffer<std::int32_t>  subpass_lp_lindptr;  // CSR indptr
    DeviceBuffer<std::int32_t>  subpass_lp_lids;     // label ids
    DeviceBuffer<float>         subpass_lp_out;      // logprob outputs
    DeviceBuffer<std::int32_t>  subpass_ent_rows;
    DeviceBuffer<float>         subpass_ent_out;
    DeviceBuffer<std::int32_t>  subpass_dist_rows;
    DeviceBuffer<float>         subpass_dist_temps;

    // Pinned host staging for D2H reads and the topk/topp scatter.
    // Allocated via cudaMallocHost so async copies bypass the driver's
    // internal pageable-staging path. Sized to max_workspace_tokens.
    std::int32_t* h_per_sample_tok_pinned = nullptr;
    std::int32_t* h_all_sampled_pinned    = nullptr;
    std::int32_t* h_sampled_pinned        = nullptr;  // D2H read of pi.sampled
    // Pinned host staging for entropy/logprob D2H reads (sized to
    // max_workspace_tokens — the row count never exceeds N, and label
    // counts in practice are bounded similarly).
    float*        h_ent_pinned            = nullptr;
    float*        h_lp_pinned             = nullptr;

    static PersistentInputs allocate(
        int max_workspace_tokens,
        int max_requests,
        int max_kv_pages,
        std::size_t max_custom_mask_bytes);

    PersistentInputs() = default;
    PersistentInputs(const PersistentInputs&) = delete;
    PersistentInputs& operator=(const PersistentInputs&) = delete;
    PersistentInputs(PersistentInputs&&) noexcept;
    PersistentInputs& operator=(PersistentInputs&&) noexcept;
    ~PersistentInputs();
};

}  // namespace pie_cuda_driver
