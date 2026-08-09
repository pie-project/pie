#pragma once

// Device/pinned scratch for FlashInfer's plan + dispatch path. DecodePlan /
// PrefillPlan write their per-request scheduling metadata -- request_indices,
// kv_tile_indices, o_indptr, kv_chunk_size_ptr, split-kv tmp buffers -- into
// these buffers. Allocated once at boot and reused across all forward passes
// (see `batch/workspace.hpp`, the sizing-policy wrapper around this type,
// and `Context::create`).
//
// This is the owner. Kernels never see it: they take an
// `AttentionWorkspaceView` (`kernels-cuda`), which is the five values they
// actually read. Everything the class adds on top of those five -- the
// allocation, the move semantics, the pinned plan-staging slots and the
// events that fence them -- is scheduling, sized by the driver's run-ahead
// depth, and so it lives here rather than downhill in the kernels.

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace_view.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

class AttentionWorkspace {
public:
    /// `plan_staging_slots` is the caller's run-ahead depth in STEPS, and it
    /// is a parameter rather than a constant here because the number is the
    /// scheduler's, not a kernel's: one slot is claimed per step
    /// (`begin_plan_update`) and is reusable only after its upload event
    /// retires, so a pool shallower than the in-flight step count blocks
    /// every submit in `cudaEventSynchronize` for ~a full GPU step. The
    /// driver derives it (`runahead.hpp`) and passes it down.
    static AttentionWorkspace allocate(
        std::size_t float_workspace_bytes = 80 * 1024 * 1024,  // 80 MiB
        std::size_t int_workspace_bytes  =  8 * 1024 * 1024,   // 8  MiB
        std::size_t plan_staging_slots   = 1);

    AttentionWorkspace() = default;
    AttentionWorkspace(const AttentionWorkspace&) = delete;
    AttentionWorkspace& operator=(const AttentionWorkspace&) = delete;
    AttentionWorkspace(AttentionWorkspace&&) noexcept;
    AttentionWorkspace& operator=(AttentionWorkspace&&) noexcept;

    ~AttentionWorkspace();

    /// What a kernel is handed. Named rather than an implicit conversion for
    /// the same reason `KvCache::layer_view` is: the crate boundary should be
    /// legible at the call site, not inferred by overload resolution.
    ///
    /// Non-const to match the accessors it is built from -- the buffers it
    /// points at are written by the kernels.
    AttentionWorkspaceView view() noexcept {
        return AttentionWorkspaceView{
            float_buffer(), float_bytes(),
            int_buffer(),   int_bytes(),
            page_locked_int(),
        };
    }

    void* float_buffer()      noexcept { return float_buf_.data(); }
    void* int_buffer()        noexcept { return int_buf_.data(); }
    void* page_locked_int()   noexcept {
        return plan_staging_[active_plan_slot_].host;
    }

    std::size_t float_bytes() const noexcept { return float_buf_.nbytes(); }
    std::size_t int_bytes()   const noexcept { return int_buf_.nbytes(); }

    void begin_plan_update();
    void end_plan_update(cudaStream_t stream);

private:
    struct PlanStaging {
        void* host = nullptr;
        cudaEvent_t upload_done = nullptr;
        bool upload_pending = false;
    };

    void ensure_plan_slot(PlanStaging& slot);


    DeviceTensor float_buf_;       // device
    DeviceTensor int_buf_;         // device
    std::size_t staging_bytes_ = 0;
    // Sized by `allocate`'s `plan_staging_slots`. Slot 0 is pinned there
    // (some ops read `page_locked_int()` without ever rotating); the rest
    // pin lazily on first rotation, so a non-rotating workspace does not
    // hold the full depth's worth of pinned host memory.
    std::vector<PlanStaging> plan_staging_;
    std::size_t active_plan_slot_ = 0;
    std::size_t next_plan_slot_ = 0;
};

}  // namespace pie_cuda_driver
