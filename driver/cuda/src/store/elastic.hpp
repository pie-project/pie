#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "pie_driver/elastic.hpp"
#include "../tensor.hpp"

namespace pie_cuda_driver {

class CudaPhysicalPool final : public pie::elastic::PhysicalPool {
  public:
    CudaPhysicalPool(
        int device_ordinal,
        std::size_t budget_bytes,
        std::size_t handle_bytes = 32ull * 1024 * 1024);
    ~CudaPhysicalPool() override;

    bool try_reserve(std::size_t pages) override;
    void unreserve(std::size_t pages) noexcept override;
    std::size_t page_bytes() const noexcept override;
    std::size_t budget_pages() const noexcept override;
    std::size_t committed_pages() const noexcept override;

    int device_ordinal() const noexcept { return device_ordinal_; }
    std::size_t allocation_granularity() const noexcept {
        return allocation_granularity_;
    }
    std::size_t handle_bytes() const noexcept { return handle_bytes_; }
    void mark_committed(std::size_t pages);
    void mark_uncommitted(std::size_t pages) noexcept;
    CUmemGenericAllocationHandle acquire_handle(std::size_t bytes);
    void release_handle(
        CUmemGenericAllocationHandle handle,
        std::size_t bytes,
        bool cache) noexcept;

  private:
    int device_ordinal_ = 0;
    std::size_t allocation_granularity_ = 0;
    std::size_t handle_bytes_ = 0;
    std::size_t budget_pages_ = 0;
    mutable std::mutex mutex_;
    std::size_t reserved_pages_ = 0;
    std::size_t committed_pages_ = 0;
    std::unordered_map<
        std::size_t,
        std::vector<CUmemGenericAllocationHandle>> free_handles_;
};

class CudaArena final : public pie::elastic::Arena {
  public:
    CudaArena(
        std::shared_ptr<CudaPhysicalPool> pool,
        std::size_t max_bytes,
        std::string label);
    ~CudaArena() override;

    CudaArena(const CudaArena&) = delete;
    CudaArena& operator=(const CudaArena&) = delete;

    std::uint64_t base() const noexcept override;
    std::size_t committed_bytes() const noexcept override;
    void ensure_committed(std::size_t bytes) override;
    void trim_committed(std::size_t bytes) override;

    std::size_t max_bytes() const noexcept { return max_bytes_; }

  private:
    static std::size_t align_up(std::size_t value, std::size_t alignment);
    void release_tail(std::size_t target_bytes) noexcept;

    std::shared_ptr<CudaPhysicalPool> pool_;
    std::string label_;
    CUdeviceptr base_ = 0;
    std::size_t max_bytes_ = 0;
    std::size_t virtual_bytes_ = 0;
    std::size_t map_unit_bytes_ = 0;
    std::vector<CUmemGenericAllocationHandle> handles_;
};

class CudaArenaAllocator {
  public:
    CudaArenaAllocator(
        std::shared_ptr<CudaPhysicalPool> pool,
        std::string label,
        bool commit_on_allocate);

    void* allocate(std::size_t bytes, std::size_t alignment);
    void ensure_fraction(std::size_t used, std::size_t capacity);
    void ensure_all();
    void ensure_bytes(std::size_t bytes);
    void trim_fraction(std::size_t used, std::size_t capacity);
    void trim_bytes(std::size_t bytes);
    std::size_t committed_bytes() const noexcept;
    std::size_t allocated_bytes() const noexcept;

  private:
    static void* allocate_callback(
        void* context,
        std::size_t bytes,
        std::size_t alignment);

    friend class ScopedCudaArenaAllocator;

    std::shared_ptr<CudaPhysicalPool> pool_;
    std::string label_;
    bool commit_on_allocate_ = false;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<CudaArena>> arenas_;
    std::size_t allocated_bytes_ = 0;
};

class ScopedCudaArenaAllocator {
  public:
    explicit ScopedCudaArenaAllocator(CudaArenaAllocator& allocator);
    ~ScopedCudaArenaAllocator();

    ScopedCudaArenaAllocator(const ScopedCudaArenaAllocator&) = delete;
    ScopedCudaArenaAllocator& operator=(const ScopedCudaArenaAllocator&) = delete;

  private:
    DeviceMemoryAllocatorBinding previous_{};
};

}  // namespace pie_cuda_driver
