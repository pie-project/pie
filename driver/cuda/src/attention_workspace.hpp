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

#include <cuda_runtime.h>

#include "tensor.hpp"

namespace pie_cuda_driver {

class AttentionWorkspace {
public:
    static AttentionWorkspace allocate(
        std::size_t float_workspace_bytes = 80 * 1024 * 1024,  // 80 MiB
        std::size_t int_workspace_bytes  =  8 * 1024 * 1024);  // 8  MiB

    AttentionWorkspace() = default;
    AttentionWorkspace(const AttentionWorkspace&) = delete;
    AttentionWorkspace& operator=(const AttentionWorkspace&) = delete;
    AttentionWorkspace(AttentionWorkspace&&) noexcept;
    AttentionWorkspace& operator=(AttentionWorkspace&&) noexcept;

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

// FlashInfer decode is only a win for certain GQA ratios — for the rest
// we route through the prefill kernel as a fallback.
bool flashinfer_decode_supports_gqa(int gqa);

// PIE_CUDA_XQA_DECODE override (defaults to enabled). When false the
// driver forces the FlashInfer paged decode kernel for all archs.
bool xqa_decode_enabled_by_env();

class HfConfig;
class Config;

// True if any layer in the HF config uses a non-full attention shape
// (e.g. sliding window). Cheap test the planner uses to gate the
// FlashInfer fast-path attention budget.
bool has_non_full_attention_layers(const HfConfig& hf);

// Byte budget for FlashInfer's per-fire float scratch on the active arch
// and TP layout. Returns a conservative base when the fast-path budget
// doesn't apply.
std::size_t attention_float_workspace_bytes(const HfConfig& hf,
                                            const Config& cfg,
                                            const cudaDeviceProp& prop,
                                            int max_requests);

}  // namespace pie_cuda_driver
