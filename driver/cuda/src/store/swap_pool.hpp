#pragma once

// Pinned host KV pool for swap-out / swap-in.
//
// Layout per layer matches the device-side `KvCache` exactly:
//   [num_pages, page_size, num_kv_heads, head_dim] in `dtype`.
// One K pool + one V pool per transformer layer.
//
// Allocated via `cudaMallocHost` so the binary's CUDA driver can DMA
// directly to/from these buffers. Owned by the C++ binary; the runtime
// never sees the host pointers — it addresses pages by index, and the
// control channel resolves indices to memory.

#include <cstdint>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#include "kv_cache.hpp"
#include "../tensor.hpp"

namespace pie_cuda_driver {

class SwapPool {
public:
    static SwapPool allocate_for_cache(const KvCache& cache, int num_pages);

    static SwapPool allocate(
        int num_layers,
        int num_pages,
        int page_size,
        int num_kv_heads,
        int head_dim,
        DType dtype = DType::BF16);

    SwapPool() = default;
    SwapPool(const SwapPool&) = delete;
    SwapPool& operator=(const SwapPool&) = delete;
    SwapPool(SwapPool&&) noexcept;
    SwapPool& operator=(SwapPool&&) noexcept;
    ~SwapPool();

    int num_pages() const noexcept { return num_pages_; }
    int num_layers() const noexcept { return num_layers_; }
    std::size_t bytes_per_page() const noexcept { return page_bytes_; }
    cudaStream_t stream() const noexcept { return stream_; }
    /// The stream a restore (H2D) is issued on. Restores are on the
    /// critical path — an evicted process cannot run until its pages are
    /// back — while evictions are background work, so sharing one stream
    /// made every restore queue behind every pending eviction. PCIe is
    /// full duplex; separating them lets the two directions proceed at
    /// once instead of FIFO.
    cudaStream_t restore_stream() const noexcept { return restore_stream_; }
    void synchronize() const;

    // Bulk page copies, src/dst expressed as page indices. Pairs are
    // issued via `cudaMemcpyAsync` on a dedicated swap stream; we sync only
    // that stream before returning, so the compute stream that `fire_batch`
    // uses isn't blocked. All copies are applied across **every layer** —
    // the runtime treats KV pages as opaque per-page resources, not
    // per-layer (matches `pie_driver`'s semantics).
    void copy_d2h(KvCache& cache,
                  std::span<const std::uint32_t> src_gpu_pages,
                  std::span<const std::uint32_t> dst_host_slots);
    void copy_d2h_async(KvCache& cache,
                        std::span<const std::uint32_t> src_gpu_pages,
                        std::span<const std::uint32_t> dst_host_slots);

    void copy_h2d(KvCache& cache,
                  std::span<const std::uint32_t> src_host_slots,
                  std::span<const std::uint32_t> dst_gpu_pages);
    void copy_h2d_async(KvCache& cache,
                        std::span<const std::uint32_t> src_host_slots,
                        std::span<const std::uint32_t> dst_gpu_pages);

    void copy_d2d(KvCache& cache,
                  std::span<const std::uint32_t> src_gpu_pages,
                  std::span<const std::uint32_t> dst_gpu_pages);
    void copy_d2d_async(KvCache& cache,
                        std::span<const std::uint32_t> src_gpu_pages,
                        std::span<const std::uint32_t> dst_gpu_pages);

    void copy_h2h(std::span<const std::uint32_t> src_host_slots,
                  std::span<const std::uint32_t> dst_host_slots);
    void copy_h2h_async(std::span<const std::uint32_t> src_host_slots,
                        std::span<const std::uint32_t> dst_host_slots);

private:
    int num_layers_ = 0;
    int num_pages_ = 0;
    int page_size_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    std::size_t page_bytes_ = 0;
    struct HostBuffer {
        void* data = nullptr;
        std::size_t page_bytes = 0;
    };
    std::vector<std::vector<HostBuffer>> host_pools_;
    // Dedicated streams — swap copies don't share the default stream that
    // the forward pass runs on. Synced per-call before responding.
    // `stream_` carries eviction (D2H) and graft (D2D) traffic;
    // `restore_stream_` carries restores (H2D). See `restore_stream()`.
    cudaStream_t stream_ = nullptr;
    cudaStream_t restore_stream_ = nullptr;
};

}  // namespace pie_cuda_driver
