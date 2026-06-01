#pragma once

// Persistent workspace buffers for FlashInfer's plan + dispatch path.
//
// Each fire calls `DecodePlan` (decode-only batches) or `PrefillPlan`. The plan
// computes per-request work distribution and writes scheduling metadata into
// these buffers — request_indices, kv_tile_indices, o_indptr,
// kv_chunk_size_ptr, and split-kv tmp buffers. Allocated once at model build,
// reused across all forward passes (its fixed device addresses are what make
// CUDA-graph replay safe).
//
// De-branded lift of driver/cuda/src/attention_workspace.{hpp,cpp}: the
// `DeviceTensor` allocations become raw cudaMalloc, and the HfConfig/Config-
// dependent sizing helpers are dropped (the Rust control plane sizes the
// buffers and the gqa gate lives in sampler/builder). The buffer trio + the
// FlashInfer decode-GQA support predicate are all the device side needs.

#include <cstddef>

#include <cuda_runtime.h>

namespace pie_cuda_device::ops {

class AttentionWorkspace {
public:
    AttentionWorkspace() = default;
    AttentionWorkspace(const AttentionWorkspace&) = delete;
    AttentionWorkspace& operator=(const AttentionWorkspace&) = delete;
    AttentionWorkspace(AttentionWorkspace&&) noexcept;
    AttentionWorkspace& operator=(AttentionWorkspace&&) noexcept;
    ~AttentionWorkspace();

    // Allocates the device float/int scratch + a pinned host int buffer (same
    // size as the device int buffer — FlashInfer fills it on the host then
    // async-copies it across). Throws std::runtime_error on cudaMalloc
    // failure. Defaults mirror driver/cuda (128 MiB float / 8 MiB int).
    static AttentionWorkspace allocate(
        std::size_t float_workspace_bytes = 128ull * 1024 * 1024,
        std::size_t int_workspace_bytes   =   8ull * 1024 * 1024);

    void* float_buffer()    noexcept { return float_buf_; }
    void* int_buffer()      noexcept { return int_buf_; }
    void* page_locked_int() noexcept { return page_locked_int_; }

    std::size_t float_bytes() const noexcept { return float_bytes_; }
    std::size_t int_bytes()   const noexcept { return int_bytes_; }

private:
    void* float_buf_ = nullptr;        // device
    void* int_buf_ = nullptr;          // device
    void* page_locked_int_ = nullptr;  // host pinned, same size as int_buf_
    std::size_t float_bytes_ = 0;
    std::size_t int_bytes_ = 0;
};

// The tensor-core BatchDecode kernel only instantiates GQA group sizes
// {1,2,3,4,8}; other ratios route through the prefill kernel instead.
bool decode_supports_gqa(int gqa);

}  // namespace pie_cuda_device::ops
