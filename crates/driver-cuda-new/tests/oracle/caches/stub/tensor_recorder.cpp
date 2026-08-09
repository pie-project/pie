// The two things this oracle replaces: `DeviceTensor::allocate` and the CUDA
// runtime entry points the three cache builders call.
//
// `kernels-cuda/csrc/src/tensor.cpp` ends in a `cudaMalloc`, so every shape
// the shipping code computes is consumed by the driver and leaves only a
// pointer behind. The class is declared in the REAL `tensor.hpp`, which is
// copied verbatim; only this implementation differs.
//
// Unlike the kv_cache oracle's recorder, allocations here are REAL host
// memory rather than fabricated addresses, because `dsv4_compress_cache.cpp`
// hands those pointers to `cudaMemset` and the recorder has to be able to say
// which tensor a memset landed in.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "tensor.hpp"

namespace oracle_cuda {
namespace {

std::vector<std::string> g_log;

struct Region {
    std::size_t bytes;
    std::string name;
};

// Ordered so `where` finds the region containing an interior pointer with one
// `upper_bound` -- the "offset into a buffer" question.
std::map<const void*, Region> g_regions;
int g_host_ordinal = 0;
int g_memset_seen = 0;
int g_memset_fail_at = -1;

}  // namespace

const std::vector<std::string>& log() { return g_log; }
void note(const std::string& line) { g_log.push_back(line); }
int next_host_ordinal() { return g_host_ordinal++; }
void fail_memset_at(int n) { g_memset_fail_at = n; }

void name_region(const void* ptr, std::size_t bytes, const std::string& name) {
    if (ptr == nullptr) return;
    g_regions[ptr] = Region{bytes, name};
}

void forget_region(const void* ptr) {
    if (ptr == nullptr) return;
    g_regions.erase(ptr);
}

std::string where(const void* ptr) {
    if (ptr == nullptr) return "null";
    auto it = g_regions.upper_bound(ptr);
    if (it != g_regions.begin()) {
        --it;
        const auto* base = static_cast<const unsigned char*>(it->first);
        const auto* p = static_cast<const unsigned char*>(ptr);
        const std::size_t off = static_cast<std::size_t>(p - base);
        if (off <= it->second.bytes) {
            return it->second.name + "+" + std::to_string(off);
        }
    }
    return "unknown";
}

bool memset_should_fail() {
    const int n = g_memset_seen++;
    return g_memset_fail_at >= 0 && n == g_memset_fail_at;
}

void reset_stream_ordinal();

void reset_case() {
    g_log.clear();
    g_regions.clear();
    g_host_ordinal = 0;
    g_memset_seen = 0;
    g_memset_fail_at = -1;
    reset_stream_ordinal();
}

}  // namespace oracle_cuda

namespace oracle_cuda {
bool memset_should_fail();
}

namespace {
std::string sz(std::size_t v) { return std::to_string(v); }
cudaError_t g_last_error = cudaSuccess;
}  // namespace

const char* cudaGetErrorString(cudaError_t) { return "oracle: injected"; }
const char* cudaGetErrorName(cudaError_t) { return "cudaErrorInvalidValue"; }

cudaError_t cudaGetLastError() {
    const cudaError_t e = g_last_error;
    g_last_error = cudaSuccess;
    oracle_cuda::note("getlasterror -> " + std::to_string(e));
    return e;
}

cudaError_t cudaMalloc(void** ptr, std::size_t size) {
    *ptr = std::malloc(size == 0 ? 1 : size);
    return cudaSuccess;
}
cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

// Pinned host buffers are named by ALLOCATION ORDINAL, not by (layer, buffer),
// because the order is itself part of the contract: `allocate` walks layers
// outermost and always makes exactly two per layer, while `allocate_for_cache`
// makes as many as the device cache exposes. Naming them by role would hide a
// pool that allocated the right total in the wrong shape.
cudaError_t cudaMallocHost(void** ptr, std::size_t size) {
    void* p = std::malloc(size == 0 ? 1 : size);
    const std::string name = "host" + std::to_string(oracle_cuda::next_host_ordinal());
    oracle_cuda::name_region(p, size, name);
    oracle_cuda::note("mallochost " + name + " bytes=" + sz(size));
    *ptr = p;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    oracle_cuda::forget_region(ptr);
    std::free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMemset(void* dst, int value, std::size_t count) {
    const bool fail = oracle_cuda::memset_should_fail();
    oracle_cuda::note("memset " + oracle_cuda::where(dst) + " val=" +
                      std::to_string(value) + " len=" + sz(count) +
                      (fail ? " -> FAIL" : " -> ok"));
    if (fail) {
        g_last_error = cudaErrorInvalidValue;
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count,
                            cudaStream_t) {
    return cudaMemset(dst, value, count);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                            cudaMemcpyKind kind, cudaStream_t) {
    oracle_cuda::note("memcpy dst=" + oracle_cuda::where(dst) + " src=" +
                      oracle_cuda::where(src) + " len=" + sz(count) +
                      " kind=" + std::to_string(static_cast<int>(kind)));
    return cudaSuccess;
}

cudaError_t cudaMemcpyBatchAsync(void* const*, const void* const*, const std::size_t*,
                                 std::size_t count, cudaMemcpyAttributes*,
                                 std::size_t*, std::size_t, cudaStream_t) {
    oracle_cuda::note("memcpybatch n=" + sz(count));
    return cudaSuccess;
}

// Streams are ordinals too. The two a swap pool creates are distinguishable
// only by creation order, and which one carries restores is the point of
// having two.
namespace {
std::uintptr_t g_next_stream = 0x51;
}

// Per-case, so a transcript row does not depend on how many cases ran before
// it. Without this the stream names in section 4 shift whenever section 3
// gains a case, which would make every diff unreadable for no gain in
// coverage: what matters about the two streams is that both exist and that
// they are distinct, not what number they got.
namespace oracle_cuda {
void reset_stream_ordinal() { g_next_stream = 0x51; }
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    *stream = reinterpret_cast<void*>(g_next_stream);
    oracle_cuda::note("stream_create s" + std::to_string(g_next_stream - 0x51) +
                      " flags=" + std::to_string(flags));
    g_next_stream += 1;
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

namespace pie_cuda_driver {

std::vector<std::string> g_alloc_log;

namespace {
int g_tensor_ordinal = 0;
}

void reset_alloc_log() {
    g_alloc_log.clear();
    g_tensor_ordinal = 0;
}

const std::vector<std::string>& alloc_log() { return g_alloc_log; }

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

    std::string row = dtype_name(dtype);
    row += '[';
    for (std::size_t i = 0; i < t.shape_.size(); ++i) {
        if (i != 0) row += ',';
        row += std::to_string(t.shape_[i]);
    }
    row += "]=" + std::to_string(t.nbytes_);
    g_alloc_log.push_back(row);

    // Like the real allocator, a zero-byte request yields a null pointer, and
    // `dsv4_compress_cache.cpp` tests for exactly that before zeroing.
    if (t.nbytes_ > 0) {
        void* p = std::malloc(t.nbytes_);
        const std::string name = "dev" + std::to_string(g_tensor_ordinal++);
        oracle_cuda::name_region(p, t.nbytes_, name);
        t.ptr_ = p;
        t.arena_owned_ = false;
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
    // Deliberately leaks: the transcript renders addresses relative to their
    // region and a freed-then-reused address would silently rename a buffer
    // mid-case. The process is short-lived.
    ptr_ = nullptr;
    numel_ = 0;
    nbytes_ = 0;
    owns_memory_ = false;
    arena_owned_ = false;
}

namespace {
DeviceMemoryAllocatorBinding g_binding{};
}

DeviceMemoryAllocatorBinding set_device_memory_allocator(
    DeviceMemoryAllocateCallback allocate, void* context) noexcept {
    const DeviceMemoryAllocatorBinding prev = g_binding;
    g_binding.allocate = allocate;
    g_binding.context = context;
    return prev;
}

void set_device_tensor_memory_callback(DeviceTensorMemoryCallback,
                                       void*) noexcept {}

}  // namespace pie_cuda_driver
