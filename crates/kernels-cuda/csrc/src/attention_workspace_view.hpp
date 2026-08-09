#pragma once

// The attention scratch, as a kernel sees it.
//
// The buffers themselves are owned by an `AttentionWorkspace` in the driver
// (`driver-cuda/csrc/src/attention_workspace.hpp`), which is where they
// belong: allocating them, rotating the pinned plan-staging slots and
// fencing those slots on CUDA events are scheduler concerns, sized by the
// driver's run-ahead depth. None of that is a kernel's business, and the
// measurement says the kernels agree -- across all of `attn/` they read
// exactly the five values below and call nothing else.
//
// So the class stays home and only this crosses. The same one-way rule as
// `attn/kv_cache_view.hpp`: the driver reaches down, nothing here reaches
// up. It is also what lets a launcher be called from a driver that has no
// C++ objects at all -- this is standard-layout, so a `#[repr(C)]` mirror
// on the Rust side is a provable equivalent rather than a hopeful one.

#include <cstddef>

namespace pie_cuda_driver {

struct AttentionWorkspaceView {
    /// Device scratch FlashInfer accumulates split-KV partials into.
    void* float_buffer;
    /// Size of `float_buffer`. Kernels check their budget against it.
    std::size_t float_bytes;
    /// Device scratch holding per-request scheduling metadata (request
    /// indices, KV tile indices, `o_indptr`, chunk sizes).
    void* int_buffer;
    /// Size of `int_buffer`.
    std::size_t int_bytes;
    /// Pinned host mirror of `int_buffer`, staged by a plan and uploaded by
    /// the driver. This is the active slot: which one that is rotates per
    /// step, and the rotation is not visible from here.
    void* page_locked_int;
};

}  // namespace pie_cuda_driver
