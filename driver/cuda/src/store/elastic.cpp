#include "elastic.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pie_cuda_driver {
namespace {

void check_cu(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    throw std::runtime_error(
        std::string(operation) + " failed: " +
        (name != nullptr ? name : "unknown") + " (" +
        (description != nullptr ? description : "no description") + ")");
}

}  // namespace

CudaPhysicalPool::CudaPhysicalPool(
    int device_ordinal,
    std::size_t budget_bytes,
    std::size_t handle_bytes)
    : device_ordinal_(device_ordinal) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal_;
    check_cu(
        cuMemGetAllocationGranularity(
            &allocation_granularity_,
            &prop,
            CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
    handle_bytes_ = std::max(handle_bytes, allocation_granularity_);
    handle_bytes_ =
        (handle_bytes_ + allocation_granularity_ - 1) /
        allocation_granularity_ * allocation_granularity_;
    budget_pages_ = pie::elastic::pages_for_bytes(budget_bytes);
}

CudaPhysicalPool::~CudaPhysicalPool() {
    for (auto& [bytes, handles] : free_handles_) {
        static_cast<void>(bytes);
        for (const auto handle : handles) cuMemRelease(handle);
    }
}

bool CudaPhysicalPool::try_reserve(std::size_t pages) {
    std::lock_guard lock(mutex_);
    if (pages > budget_pages_ - std::min(budget_pages_, reserved_pages_)) {
        return false;
    }
    reserved_pages_ += pages;
    return true;
}

void CudaPhysicalPool::unreserve(std::size_t pages) noexcept {
    std::lock_guard lock(mutex_);
    reserved_pages_ -= std::min(reserved_pages_, pages);
}

std::size_t CudaPhysicalPool::page_bytes() const noexcept {
    return pie::elastic::kLogicalPageBytes;
}

std::size_t CudaPhysicalPool::budget_pages() const noexcept {
    return budget_pages_;
}

std::size_t CudaPhysicalPool::committed_pages() const noexcept {
    std::lock_guard lock(mutex_);
    return committed_pages_;
}

void CudaPhysicalPool::mark_committed(std::size_t pages) {
    std::lock_guard lock(mutex_);
    committed_pages_ += pages;
}

void CudaPhysicalPool::mark_uncommitted(std::size_t pages) noexcept {
    std::lock_guard lock(mutex_);
    committed_pages_ -= std::min(committed_pages_, pages);
}

CUmemGenericAllocationHandle CudaPhysicalPool::acquire_handle(
    std::size_t bytes) {
    {
        std::lock_guard lock(mutex_);
        auto found = free_handles_.find(bytes);
        if (found != free_handles_.end() && !found->second.empty()) {
            const auto handle = found->second.back();
            found->second.pop_back();
            return handle;
        }
    }
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal_;
    CUmemGenericAllocationHandle handle = 0;
    check_cu(cuMemCreate(&handle, bytes, &prop, 0), "cuMemCreate");
    return handle;
}

void CudaPhysicalPool::release_handle(
    CUmemGenericAllocationHandle handle,
    std::size_t bytes,
    bool cache) noexcept {
    if (handle == 0) return;
    if (cache) {
        std::lock_guard lock(mutex_);
        auto& handles = free_handles_[bytes];
        if (handles.empty()) {
            handles.push_back(handle);
            return;
        }
    }
    cuMemRelease(handle);
}

std::size_t CudaArena::align_up(
    std::size_t value,
    std::size_t alignment) {
    return value == 0 ? 0 : (value + alignment - 1) / alignment * alignment;
}

CudaArena::CudaArena(
    std::shared_ptr<CudaPhysicalPool> pool,
    std::size_t max_bytes,
    std::string label)
    : pool_(std::move(pool)),
      label_(std::move(label)),
      max_bytes_(max_bytes) {
    if (pool_ == nullptr || max_bytes_ == 0) {
        throw std::invalid_argument("CudaArena requires a pool and non-zero size");
    }
    map_unit_bytes_ = std::min(
        pool_->handle_bytes(),
        align_up(max_bytes_, pool_->allocation_granularity()));
    map_unit_bytes_ = std::max(
        map_unit_bytes_,
        pool_->allocation_granularity());
    virtual_bytes_ = align_up(
        std::max(max_bytes_, max_bytes_ <= std::numeric_limits<std::size_t>::max() / 2
                                 ? max_bytes_ * 2
                                 : max_bytes_),
        map_unit_bytes_);
    check_cu(
        cuMemAddressReserve(
            &base_,
            virtual_bytes_,
            pool_->allocation_granularity(),
            0,
            0),
        "cuMemAddressReserve");
}

CudaArena::~CudaArena() {
    release_tail(0);
    if (base_ != 0) {
        cuMemAddressFree(base_, virtual_bytes_);
        base_ = 0;
    }
}

std::uint64_t CudaArena::base() const noexcept {
    return static_cast<std::uint64_t>(base_);
}

std::size_t CudaArena::committed_bytes() const noexcept {
    return handles_.size() * map_unit_bytes_;
}

void CudaArena::ensure_committed(std::size_t bytes) {
    if (bytes > max_bytes_) {
        throw std::out_of_range(
            label_ + ": requested commit exceeds arena capacity");
    }
    const std::size_t target = align_up(bytes, map_unit_bytes_);
    if (target <= committed_bytes()) return;
    const std::size_t delta = target - committed_bytes();
    const std::size_t logical_pages =
        pie::elastic::pages_for_bytes(delta, pool_->page_bytes());
    if (!pool_->try_reserve(logical_pages)) {
        throw std::runtime_error(
            label_ + ": shared physical pool budget exhausted");
    }

    const std::size_t old_count = handles_.size();
    try {
        while (committed_bytes() < target) {
            CUmemAllocationProp prop{};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = pool_->device_ordinal();
            CUmemGenericAllocationHandle handle =
                pool_->acquire_handle(map_unit_bytes_);
            const CUdeviceptr address =
                base_ + handles_.size() * map_unit_bytes_;
            try {
                check_cu(
                    cuMemMap(address, map_unit_bytes_, 0, handle, 0),
                    "cuMemMap");
                CUmemAccessDesc access{};
                access.location = prop.location;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                check_cu(
                    cuMemSetAccess(address, map_unit_bytes_, &access, 1),
                    "cuMemSetAccess");
            } catch (...) {
                pool_->release_handle(handle, map_unit_bytes_, true);
                throw;
            }
            handles_.push_back(handle);
        }
        pool_->mark_committed(logical_pages);
    } catch (...) {
        while (handles_.size() > old_count) {
            const std::size_t index = handles_.size() - 1;
            cuMemUnmap(base_ + index * map_unit_bytes_, map_unit_bytes_);
            pool_->release_handle(
                handles_.back(), map_unit_bytes_, true);
            handles_.pop_back();
        }
        pool_->unreserve(logical_pages);
        throw;
    }
}

void CudaArena::release_tail(std::size_t target_bytes) noexcept {
    const std::size_t target = align_up(target_bytes, map_unit_bytes_);
    const std::size_t before = committed_bytes();
    while (committed_bytes() > target && !handles_.empty()) {
        const std::size_t index = handles_.size() - 1;
        cuMemUnmap(base_ + index * map_unit_bytes_, map_unit_bytes_);
        pool_->release_handle(
            handles_.back(), map_unit_bytes_, false);
        handles_.pop_back();
    }
    const std::size_t released = before - committed_bytes();
    const std::size_t pages =
        pie::elastic::pages_for_bytes(released, pool_->page_bytes());
    pool_->mark_uncommitted(pages);
    pool_->unreserve(pages);
}

void CudaArena::trim_committed(std::size_t bytes) {
    if (bytes > max_bytes_) {
        throw std::out_of_range(
            label_ + ": requested trim exceeds arena capacity");
    }
    release_tail(bytes);
}

CudaArenaAllocator::CudaArenaAllocator(
    std::shared_ptr<CudaPhysicalPool> pool,
    std::string label,
    bool commit_on_allocate)
    : pool_(std::move(pool)),
      label_(std::move(label)),
      commit_on_allocate_(commit_on_allocate) {}

void* CudaArenaAllocator::allocate(
    std::size_t bytes,
    std::size_t /*alignment*/) {
    if (bytes == 0) return nullptr;
    std::lock_guard lock(mutex_);
    auto arena = std::make_unique<CudaArena>(
        pool_,
        bytes,
        label_ + "-" + std::to_string(arenas_.size()));
    if (commit_on_allocate_) arena->ensure_committed(bytes);
    void* result = reinterpret_cast<void*>(arena->base());
    allocated_bytes_ += bytes;
    arenas_.push_back(std::move(arena));
    return result;
}

void* CudaArenaAllocator::allocate_callback(
    void* context,
    std::size_t bytes,
    std::size_t alignment) {
    return static_cast<CudaArenaAllocator*>(context)->allocate(bytes, alignment);
}

void CudaArenaAllocator::ensure_fraction(
    std::size_t used,
    std::size_t capacity) {
    if (capacity == 0) return;
    used = std::min(used, capacity);
    std::lock_guard lock(mutex_);
    for (auto& arena : arenas_) {
        const std::size_t target =
            (arena->max_bytes() * used + capacity - 1) / capacity;
        arena->ensure_committed(target);
    }
}

void CudaArenaAllocator::ensure_all() {
    std::lock_guard lock(mutex_);
    for (auto& arena : arenas_) {
        arena->ensure_committed(arena->max_bytes());
    }
}

void CudaArenaAllocator::ensure_bytes(std::size_t bytes) {
    const std::size_t total = allocated_bytes();
    ensure_fraction(std::min(bytes, total), std::max<std::size_t>(1, total));
}

void CudaArenaAllocator::trim_fraction(
    std::size_t used,
    std::size_t capacity) {
    if (capacity == 0) return;
    used = std::min(used, capacity);
    std::lock_guard lock(mutex_);
    for (auto& arena : arenas_) {
        const std::size_t target =
            (arena->max_bytes() * used + capacity - 1) / capacity;
        arena->trim_committed(target);
    }
}

void CudaArenaAllocator::trim_bytes(std::size_t bytes) {
    const std::size_t total = allocated_bytes();
    trim_fraction(std::min(bytes, total), std::max<std::size_t>(1, total));
}

std::size_t CudaArenaAllocator::committed_bytes() const noexcept {
    std::lock_guard lock(mutex_);
    std::size_t total = 0;
    for (const auto& arena : arenas_) total += arena->committed_bytes();
    return total;
}

std::size_t CudaArenaAllocator::allocated_bytes() const noexcept {
    std::lock_guard lock(mutex_);
    return allocated_bytes_;
}

ScopedCudaArenaAllocator::ScopedCudaArenaAllocator(
    CudaArenaAllocator& allocator)
    : previous_(set_device_memory_allocator(
          &CudaArenaAllocator::allocate_callback,
          &allocator)) {}

ScopedCudaArenaAllocator::~ScopedCudaArenaAllocator() {
    set_device_memory_allocator(previous_.allocate, previous_.context);
}

}  // namespace pie_cuda_driver
