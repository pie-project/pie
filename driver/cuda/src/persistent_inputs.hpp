#pragma once

// Pre-allocated input device buffers, sized once at startup for the
// configured worst case and reused across `fire_batch` invocations.
//
// Why: the previous design called `DeviceBuffer<T>::from_host(...)` per
// fire — six fresh `cudaMalloc`s each time the wire control arrays were
// uploaded. That cost ~50 µs/fire on its own (small), but more
// importantly the freshly-malloc'd device addresses changed every fire
// → kernels couldn't be CUDA-graph-captured because the captured graph
// hardcodes pointer arguments at capture time. Replay with new
// pointers would read stale memory and produce gibberish.
//
// With this struct, the 9 wire control arrays + sampled-token output
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

    // Sampler per-row parameters. Capacity = max_workspace_tokens.
    // Refreshed per fire by `request_handler::handle_fire_batch`.
    // Held here (rather than allocated inside `dispatch_sampling`) so
    // the sampling kernel runs without per-fire `cudaMalloc/cudaFree`
    // churn — the prior path opened 3–6 fresh device allocations per
    // fire just for these scalar arrays, ~10k API calls over a
    // throughput run.
    DeviceBuffer<float>          sample_temp;
    DeviceBuffer<float>          sample_top_p;
    DeviceBuffer<float>          sample_min_p;
    DeviceBuffer<std::int32_t>   sample_top_k;
    DeviceBuffer<std::uint32_t>  sample_seed;
    DeviceBuffer<std::uint64_t>  sample_seed64;
    DeviceBuffer<std::int32_t>   sample_idx;
    DeviceBuffer<std::int32_t>   sample_per_token;
    DeviceBuffer<bool>           sample_valid;

    static PersistentInputs allocate(
        int max_workspace_tokens,
        int max_requests,
        int max_kv_pages,
        std::size_t max_custom_mask_bytes);
};

}  // namespace pie_cuda_driver
