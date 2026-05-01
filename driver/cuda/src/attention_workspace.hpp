#pragma once

// Persistent workspace buffers for flashinfer's plan + dispatch path.
//
// Each fire_batch calls `DecodePlan` (for decode-only batches) or
// `PrefillPlan` (Phase 2). The plan computes per-request work distribution
// and writes scheduling metadata into these buffers — request_indices,
// kv_tile_indices, o_indptr, kv_chunk_size_ptr, and split-kv tmp buffers.
//
// Allocated once at boot; reused across all forward passes.

#include <cstddef>

#include "tensor.hpp"

namespace pie_cuda_driver {

class AttentionWorkspace {
public:
    static AttentionWorkspace allocate(
        std::size_t float_workspace_bytes = 64 * 1024 * 1024,   // 64 MiB
        std::size_t int_workspace_bytes  =  8 * 1024 * 1024);   // 8  MiB

    AttentionWorkspace() = default;
    AttentionWorkspace(const AttentionWorkspace&) = delete;
    AttentionWorkspace& operator=(const AttentionWorkspace&) = delete;
    AttentionWorkspace(AttentionWorkspace&&) noexcept = default;
    AttentionWorkspace& operator=(AttentionWorkspace&&) noexcept = default;

    ~AttentionWorkspace();

    void* float_buffer()      noexcept { return float_buf_.data(); }
    void* int_buffer()        noexcept { return int_buf_.data(); }
    void* page_locked_int()   noexcept { return page_locked_int_; }

    std::size_t float_bytes() const noexcept { return float_buf_.nbytes(); }
    std::size_t int_bytes()   const noexcept { return int_buf_.nbytes(); }

private:
    DeviceTensor float_buf_;       // device
    DeviceTensor int_buf_;         // device
    void* page_locked_int_ = nullptr;  // host pinned, same size as int_buf_
};

}  // namespace pie_cuda_driver
