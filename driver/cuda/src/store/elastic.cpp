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

std::size_t proportional_target(
    std::size_t bytes,
    std::size_t used,
    std::size_t capacity) {
    if (bytes == 0 || used == 0 || capacity == 0) return 0;
    used = std::min(used, capacity);
    const auto numerator =
        static_cast<unsigned __int128>(bytes) * used + capacity - 1;
    return static_cast<std::size_t>(numerator / capacity);
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
    budget_pages_ = budget_bytes / pie::elastic::kLogicalPageBytes;
    hard_budget_pages_ = budget_pages_;
}

CudaPhysicalPool::~CudaPhysicalPool() = default;

bool CudaPhysicalPool::try_reserve(std::size_t pages) {
    std::lock_guard lock(mutex_);
    const std::size_t charged = committed_pages_ + held_pages_;
    if (pages > budget_pages_ - std::min(budget_pages_, charged)) {
        return false;
    }
    held_pages_ += pages;
    return true;
}

void CudaPhysicalPool::unreserve(std::size_t pages) noexcept {
    std::lock_guard lock(mutex_);
    const std::size_t released = std::min(held_pages_, pages);
    held_pages_ -= released;
    if (released != 0) ++generation_;
}

std::size_t CudaPhysicalPool::page_bytes() const noexcept {
    return pie::elastic::kLogicalPageBytes;
}

std::size_t CudaPhysicalPool::budget_pages() const noexcept {
    std::lock_guard lock(mutex_);
    return budget_pages_;
}

std::size_t CudaPhysicalPool::committed_pages() const noexcept {
    std::lock_guard lock(mutex_);
    return committed_pages_;
}

void CudaPhysicalPool::mark_committed(std::size_t pages) {
    std::lock_guard lock(mutex_);
    if (pages > held_pages_) {
        throw std::logic_error("committed pages exceed held pages");
    }
    held_pages_ -= pages;
    committed_pages_ += pages;
}

void CudaPhysicalPool::mark_uncommitted(std::size_t pages) noexcept {
    std::lock_guard lock(mutex_);
    const std::size_t released = std::min(committed_pages_, pages);
    committed_pages_ -= released;
    if (released != 0) ++generation_;
}

std::size_t CudaPhysicalPool::held_pages() const noexcept {
    std::lock_guard lock(mutex_);
    return held_pages_;
}

std::size_t CudaPhysicalPool::charged_pages() const noexcept {
    std::lock_guard lock(mutex_);
    return committed_pages_ + held_pages_;
}

std::size_t CudaPhysicalPool::hard_budget_pages() const noexcept {
    std::lock_guard lock(mutex_);
    return hard_budget_pages_;
}

std::uint64_t CudaPhysicalPool::generation() const noexcept {
    std::lock_guard lock(mutex_);
    return generation_;
}

void CudaPhysicalPool::recalibrate_budget(
    std::size_t available_bytes,
    std::size_t safety_floor_bytes,
    bool reset_hard_ceiling) {
    std::lock_guard lock(mutex_);
    const std::size_t charged = committed_pages_ + held_pages_;
    const std::size_t available_after_floor =
        available_bytes > safety_floor_bytes
            ? available_bytes - safety_floor_bytes
            : 0;
    const std::size_t available_pages =
        available_after_floor / pie::elastic::kLogicalPageBytes;
    const std::size_t recalibrated =
        charged > std::numeric_limits<std::size_t>::max() - available_pages
            ? std::numeric_limits<std::size_t>::max()
            : charged + available_pages;
    const std::size_t next_budget = std::max(charged, recalibrated);
    const std::size_t next_hard = reset_hard_ceiling
        ? next_budget
        : std::max(hard_budget_pages_, next_budget);
    if (budget_pages_ != next_budget || hard_budget_pages_ != next_hard) {
        budget_pages_ = next_budget;
        hard_budget_pages_ = next_hard;
        ++generation_;
    }
}

CUmemGenericAllocationHandle CudaPhysicalPool::acquire_handle(
    std::size_t bytes) {
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
    static_cast<void>(bytes);
    static_cast<void>(cache);
    if (handle == 0) return;
    // An unmapped generic handle still owns physical memory. Releasing it
    // keeps held/committed accounting exact; an uncharged handle cache could
    // otherwise exceed the shared budget after repeated trim/grow cycles.
    cuMemRelease(handle);
}

bool CudaPhysicalPool::should_fail_mapping_for_test() {
    std::lock_guard lock(mutex_);
    if (fail_mapping_after_ == std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    if (fail_mapping_after_ == 0) {
        fail_mapping_after_ = std::numeric_limits<std::size_t>::max();
        return true;
    }
    --fail_mapping_after_;
    return false;
}

void CudaPhysicalPool::fail_mapping_after_for_test(
    std::size_t successful_mappings) {
    std::lock_guard lock(mutex_);
    fail_mapping_after_ = successful_mappings;
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

std::size_t CudaArena::target_committed_bytes(std::size_t bytes) const {
    if (bytes > max_bytes_) {
        throw std::out_of_range(
            label_ + ": requested commit exceeds arena capacity");
    }
    return align_up(bytes, map_unit_bytes_);
}

std::size_t CudaArena::target_pages(std::size_t bytes) const {
    return pie::elastic::pages_for_bytes(
        target_committed_bytes(bytes), pool_->page_bytes());
}

void CudaArena::grow_reserved(std::size_t bytes) {
    const std::size_t target = target_committed_bytes(bytes);
    if (target <= committed_bytes()) return;
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
            bool mapped = false;
            try {
                if (pool_->should_fail_mapping_for_test()) {
                    throw std::runtime_error(
                        label_ + ": injected mapping failure");
                }
                check_cu(
                    cuMemMap(address, map_unit_bytes_, 0, handle, 0),
                    "cuMemMap");
                mapped = true;
                CUmemAccessDesc access{};
                access.location = prop.location;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                check_cu(
                    cuMemSetAccess(address, map_unit_bytes_, &access, 1),
                    "cuMemSetAccess");
            } catch (...) {
                if (mapped) {
                    cuMemUnmap(address, map_unit_bytes_);
                }
                pool_->release_handle(handle, map_unit_bytes_, true);
                throw;
            }
            handles_.push_back(handle);
        }
    } catch (...) {
        rollback_reserved(old_count * map_unit_bytes_);
        throw;
    }
}

void CudaArena::rollback_reserved(std::size_t bytes) noexcept {
    const std::size_t target = align_up(bytes, map_unit_bytes_);
    while (committed_bytes() > target && !handles_.empty()) {
        const std::size_t index = handles_.size() - 1;
        cuMemUnmap(base_ + index * map_unit_bytes_, map_unit_bytes_);
        pool_->release_handle(handles_.back(), map_unit_bytes_, false);
        handles_.pop_back();
    }
}

void CudaArena::ensure_committed(std::size_t bytes) {
    const std::size_t before = committed_bytes();
    const std::size_t target = target_committed_bytes(bytes);
    if (target <= before) return;
    const std::size_t logical_pages =
        pie::elastic::pages_for_bytes(target - before, pool_->page_bytes());
    if (!pool_->try_reserve(logical_pages)) {
        throw std::runtime_error(
            label_ + ": shared physical pool budget exhausted");
    }
    try {
        grow_reserved(bytes);
        pool_->mark_committed(logical_pages);
    } catch (...) {
        rollback_reserved(before);
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
    const auto result = commit_cuda_arena_targets_atomically(
        pool_, {{this, used, capacity}});
    if (result.outcome != CudaCommitOutcome::Committed) {
        throw std::runtime_error(
            label_ + ": shared physical pool budget exhausted");
    }
}

void CudaArenaAllocator::ensure_all() {
    ensure_fraction(1, 1);
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

std::size_t CudaArenaAllocator::target_bytes(
    std::size_t used,
    std::size_t capacity) const noexcept {
    if (capacity == 0) return 0;
    used = std::min(used, capacity);
    const std::size_t total = allocated_bytes();
    return proportional_target(total, used, capacity);
}

CudaCommitResult commit_cuda_arena_targets_atomically(
    const std::shared_ptr<CudaPhysicalPool>& pool,
    const std::vector<CudaAllocatorTarget>& targets) {
    if (pool == nullptr) {
        throw std::invalid_argument("atomic CUDA arena commit requires a pool");
    }
    struct Growth {
        CudaArena* arena = nullptr;
        std::size_t before = 0;
        std::size_t target = 0;
        std::size_t delta_pages = 0;
    };
    std::vector<Growth> growth;
    std::size_t delta_pages = 0;
    std::size_t required_pages = 0;
    for (const auto& target : targets) {
        if (target.allocator == nullptr ||
            target.allocator->pool_.get() != pool.get()) {
            throw std::invalid_argument(
                "atomic CUDA arena target belongs to another pool");
        }
        const std::size_t capacity = std::max<std::size_t>(1, target.capacity);
        const std::size_t used = std::min(target.used, capacity);
        std::lock_guard lock(target.allocator->mutex_);
        for (const auto& arena : target.allocator->arenas_) {
            const std::size_t arena_target =
                proportional_target(arena->max_bytes(), used, capacity);
            const std::size_t before = arena->committed_bytes();
            const std::size_t committed_target =
                arena->target_committed_bytes(arena_target);
            const std::size_t arena_target_pages =
                pie::elastic::pages_for_bytes(
                    committed_target, pool->page_bytes());
            const std::size_t arena_delta_pages =
                committed_target > before
                    ? pie::elastic::pages_for_bytes(
                          committed_target - before, pool->page_bytes())
                    : 0;
            if (required_pages >
                    std::numeric_limits<std::size_t>::max() -
                        arena_target_pages ||
                delta_pages >
                    std::numeric_limits<std::size_t>::max() -
                        arena_delta_pages) {
                throw std::overflow_error("CUDA arena page accounting overflow");
            }
            required_pages += arena_target_pages;
            delta_pages += arena_delta_pages;
            growth.push_back(
                {arena.get(), before, arena_target, arena_delta_pages});
        }
    }

    CudaCommitResult result{
        CudaCommitOutcome::Committed,
        required_pages,
        pool->budget_pages(),
        pool->generation(),
    };
    if (required_pages > pool->hard_budget_pages()) {
        result.outcome = CudaCommitOutcome::Impossible;
        return result;
    }
    if (!pool->try_reserve(delta_pages)) {
        result.outcome = CudaCommitOutcome::Exhausted;
        result.budget_pages = pool->budget_pages();
        result.generation = pool->generation();
        return result;
    }
    try {
        for (auto& item : growth) {
            if (item.delta_pages != 0) {
                item.arena->grow_reserved(item.target);
            }
        }
        pool->mark_committed(delta_pages);
    } catch (...) {
        for (auto it = growth.rbegin(); it != growth.rend(); ++it) {
            it->arena->rollback_reserved(it->before);
        }
        pool->unreserve(delta_pages);
        throw;
    }
    result.budget_pages = pool->budget_pages();
    result.generation = pool->generation();
    return result;
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
