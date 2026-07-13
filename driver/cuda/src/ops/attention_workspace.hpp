#pragma once

// ops/: reusable attention, GEMM, MoE, and state-space launch wrappers.
// Low-level device/pinned scratch buffers for FlashInfer's plan + dispatch
// path (DecodePlan / PrefillPlan write per-request scheduling metadata —
// request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr, split-kv
// tmp buffers — into these). Ops-owned (every attention kernel wrapper takes
// one by reference), physically filed under ops/ though the type stays in
// `pie_cuda_driver` (not `pie_cuda_driver::ops`) since batch/forward.hpp
// forward-declares it there. Allocated once at boot by batch/ (see
// batch/workspace.hpp, the sizing-policy wrapper around this type) and
// reused across all forward passes.

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

    void begin_plan_update();
    void end_plan_update(cudaStream_t stream);

private:
    DeviceTensor float_buf_;       // device
    DeviceTensor int_buf_;         // device
    void* page_locked_int_ = nullptr;  // host pinned, same size as int_buf_
    cudaEvent_t plan_upload_done_ = nullptr;
    bool plan_upload_pending_ = false;
};

// FlashInfer decode is only a win for certain GQA ratios — for the rest
// we route through the prefill kernel as a fallback.
bool flashinfer_decode_supports_gqa(int gqa);

// PIE_CUDA_XQA_DECODE override (defaults to enabled). When false the
// driver forces the FlashInfer paged decode kernel for all archs.
bool xqa_decode_enabled_by_env();

}  // namespace pie_cuda_driver
