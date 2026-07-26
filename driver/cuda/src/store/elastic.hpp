#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <array>
#include <limits>
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
    std::size_t held_pages() const noexcept;
    std::size_t charged_pages() const noexcept;
    std::size_t hard_budget_pages() const noexcept;
    std::uint64_t generation() const noexcept;

    int device_ordinal() const noexcept { return device_ordinal_; }
    std::size_t allocation_granularity() const noexcept {
        return allocation_granularity_;
    }
    std::size_t handle_bytes() const noexcept { return handle_bytes_; }

    // Grant peer devices read/write access to every arena page backed by this
    // pool. VMM mappings are private to the owning device unless the peer is
    // named in the access descriptor, so without this a same-process tensor-
    // parallel peer faults when it reads this rank's activations directly
    // (which is exactly what the custom P2P all-reduce kernel does). Must be
    // called before any arena grows; pages mapped later pick the list up
    // automatically.
    void set_peer_devices(std::vector<int> peers);
    const std::vector<int>& peer_devices() const noexcept {
        return peer_devices_;
    }
    // Access descriptors for the owning device plus every peer.
    std::vector<CUmemAccessDesc> access_descriptors() const;

    void mark_committed(std::size_t pages);
    void mark_uncommitted(std::size_t pages) noexcept;
    void recalibrate_budget(
        std::size_t available_bytes,
        std::size_t safety_floor_bytes,
        bool reset_hard_ceiling);
    CUmemGenericAllocationHandle acquire_handle(std::size_t bytes);
    void release_handle(CUmemGenericAllocationHandle handle) noexcept;
    bool should_fail_mapping_for_test();
    void fail_mapping_after_for_test(std::size_t successful_mappings);

  private:
    int device_ordinal_ = 0;
    std::vector<int> peer_devices_;
    std::size_t allocation_granularity_ = 0;
    std::size_t handle_bytes_ = 0;
    std::size_t budget_pages_ = 0;
    std::size_t hard_budget_pages_ = 0;
    mutable std::mutex mutex_;
    std::size_t held_pages_ = 0;
    std::size_t committed_pages_ = 0;
    std::uint64_t generation_ = 1;
    std::size_t fail_mapping_after_ = std::numeric_limits<std::size_t>::max();
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
    std::size_t target_committed_bytes(std::size_t bytes) const;
    std::size_t target_pages(std::size_t bytes) const;
    std::size_t physical_growth_pages(std::size_t bytes) const;
    void grow_reserved(std::size_t bytes);
    void rollback_reserved(
        std::size_t bytes,
        std::size_t cached_handle_count) noexcept;
    std::size_t cached_handle_count() const noexcept {
        return cached_handles_.size();
    }

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
    std::vector<CUmemGenericAllocationHandle> cached_handles_;
};

enum class CudaCommitOutcome {
    Committed,
    Exhausted,
    Impossible,
};

struct CudaCommitResult {
    CudaCommitOutcome outcome = CudaCommitOutcome::Committed;
    std::size_t required_pages = 0;
    std::size_t budget_pages = 0;
    std::uint64_t generation = 0;
};

struct CudaAllocatorTarget;

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
    std::size_t target_bytes(
        std::size_t used,
        std::size_t capacity) const noexcept;

    // Base address of every arena backing this allocator. The custom P2P
    // all-reduce registers allocations by base pointer, so this is what has
    // to be handed to it for the workspace the model all-reduces out of.
    std::vector<void*> arena_bases() const;

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

    friend CudaCommitResult commit_cuda_arena_targets_atomically(
        const std::shared_ptr<CudaPhysicalPool>&,
        const std::vector<CudaAllocatorTarget>&);
};

struct CudaAllocatorTarget {
    CudaArenaAllocator* allocator = nullptr;
    std::size_t used = 0;
    std::size_t capacity = 1;
};

CudaCommitResult commit_cuda_arena_targets_atomically(
    const std::shared_ptr<CudaPhysicalPool>& pool,
    const std::vector<CudaAllocatorTarget>& targets);

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
