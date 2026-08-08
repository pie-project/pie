#include "tensor.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <limits>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

thread_local DeviceTensorMemoryCallback g_memory_callback = nullptr;
thread_local void* g_memory_callback_context = nullptr;
thread_local DeviceMemoryAllocatorBinding g_memory_allocator{};

void sample_memory_callback() noexcept {
    if (g_memory_callback != nullptr) {
        g_memory_callback(g_memory_callback_context);
    }
}

}  // namespace

void set_device_tensor_memory_callback(
    DeviceTensorMemoryCallback callback,
    void* context) noexcept
{
    g_memory_callback = callback;
    g_memory_callback_context = context;
}

DeviceMemoryAllocatorBinding set_device_memory_allocator(
    DeviceMemoryAllocateCallback allocate,
    void* context) noexcept {
    const DeviceMemoryAllocatorBinding previous = g_memory_allocator;
    g_memory_allocator = {allocate, context};
    return previous;
}

ScopedDeviceAllocationCounter::ScopedDeviceAllocationCounter() noexcept
    : previous_(set_device_memory_allocator(
          &ScopedDeviceAllocationCounter::allocate, this)) {}

ScopedDeviceAllocationCounter::~ScopedDeviceAllocationCounter() {
    set_device_memory_allocator(previous_.allocate, previous_.context);
}

void* ScopedDeviceAllocationCounter::allocate(
    void* context,
    std::size_t bytes,
    std::size_t /*alignment*/) {
    auto& counter = *static_cast<ScopedDeviceAllocationCounter*>(context);
    if (bytes > std::numeric_limits<std::size_t>::max() -
                    counter.allocated_bytes_) {
        throw std::overflow_error("device allocation counter overflow");
    }
    counter.allocated_bytes_ += bytes;
    return reinterpret_cast<void*>(std::uintptr_t{256});
}

// Hot-path allocator accounting, enabled by PIE_CUDA_LOG_ALLOC=1. The
// mixtral/gpt-oss MoE forward allocates its scratch per call, and cudaFree
// synchronizes the device, so the question "what does that churn cost" has to
// be answered with a number before anyone hoists the buffers.
std::atomic<long long> g_alloc_calls{0};
std::atomic<long long> g_free_calls{0};
std::atomic<long long> g_alloc_us{0};
std::atomic<long long> g_free_us{0};

bool alloc_logging_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_CUDA_LOG_ALLOC");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

struct AllocStatsDump {
    ~AllocStatsDump() {
        if (!alloc_logging_enabled()) return;
        std::fprintf(stderr,
            "[pie-alloc] allocs=%lld (%lld us total)  frees=%lld (%lld us total)\n",
            g_alloc_calls.load(), g_alloc_us.load(),
            g_free_calls.load(), g_free_us.load());
    }
};
AllocStatsDump g_alloc_stats_dump;

DeviceMemoryBlock allocate_device_memory(
    std::size_t bytes,
    std::size_t alignment) {
    if (bytes == 0) return {};
    DeviceMemoryBlock block;
    if (g_memory_allocator.allocate != nullptr) {
        block.ptr = g_memory_allocator.allocate(
            g_memory_allocator.context,
            bytes,
            alignment);
        block.arena_owned = true;
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMalloc(&block.ptr, bytes));
        if (alloc_logging_enabled()) {
            g_alloc_us.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count(),
                std::memory_order_relaxed);
        }
    }
    sample_memory_callback();
    // PIE_CUDA_LOG_ALLOC=1: report the first allocations so we can tell
    // whether a forward runs with an ARENA installed (frees become no-ops and
    // addresses are recycled) or with the plain cudaMalloc/cudaFree pair
    // (cudaFree synchronizes the device, so recycling is safe but costly).
    // g_memory_allocator is thread_local, so this can only be answered on the
    // thread that actually runs the forward.
    {
        static const bool log_alloc = [] {
            const char* v = std::getenv("PIE_CUDA_LOG_ALLOC");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        if (log_alloc) {
            g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return block;
}

void free_device_memory(DeviceMemoryBlock block) noexcept {
    if (block.ptr == nullptr || block.arena_owned) return;
    sample_memory_callback();
    const auto t0 = std::chrono::steady_clock::now();
    cudaFree(block.ptr);
    if (alloc_logging_enabled()) {
        g_free_calls.fetch_add(1, std::memory_order_relaxed);
        g_free_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count(),
            std::memory_order_relaxed);
    }
    sample_memory_callback();
}

DeviceTensor DeviceTensor::allocate(DType dtype, std::vector<std::int64_t> shape) {
    DeviceTensor t;
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.numel_ = 1;
    for (auto d : t.shape_) {
        if (d < 0) throw std::runtime_error("DeviceTensor: negative shape");
        t.numel_ *= static_cast<std::size_t>(d);
    }
    t.nbytes_ = t.numel_ * dtype_bytes(dtype);
    if (t.nbytes_ > 0) {
        const DeviceMemoryBlock block =
            allocate_device_memory(t.nbytes_, 256);
        t.ptr_ = block.ptr;
        t.arena_owned_ = block.arena_owned;
    }
    t.owns_memory_ = true;
    return t;
}

DeviceTensor DeviceTensor::view(void* ptr, DType dtype,
                                std::vector<std::int64_t> shape) {
    DeviceTensor t;
    t.ptr_ = ptr;
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.numel_ = 1;
    for (auto d : t.shape_) {
        if (d < 0) throw std::runtime_error("DeviceTensor::view: negative shape");
        t.numel_ *= static_cast<std::size_t>(d);
    }
    t.nbytes_ = t.numel_ * dtype_bytes(dtype);
    t.owns_memory_ = false;
    return t;
}

void DeviceTensor::free_() noexcept {
    if (ptr_ && owns_memory_) {
        free_device_memory({ptr_, arena_owned_});
    }
    ptr_ = nullptr;
    arena_owned_ = false;
}

}  // namespace pie_cuda_driver
