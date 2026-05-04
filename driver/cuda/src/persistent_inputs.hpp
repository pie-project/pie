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

    static PersistentInputs allocate(
        int max_workspace_tokens,
        int max_requests,
        int max_kv_pages,
        std::size_t max_custom_mask_bytes);
};

}  // namespace pie_cuda_driver
