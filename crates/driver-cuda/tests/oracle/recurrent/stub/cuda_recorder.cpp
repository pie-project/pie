// Implementation of the recording CUDA stand-in, plus the device-memory
// allocator `DeviceBuffer` calls into.
//
// Allocations are real host memory, so pointer arithmetic in the code under
// test is real pointer arithmetic and an off-by-one layer stride produces an
// address outside the region -- which renders as a raw number and shows up in
// the diff, rather than quietly aliasing.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

// Ordered so `where` can find the region containing an interior pointer with
// one `upper_bound`, which is exactly the "offset into a buffer" question.
std::map<const void*, Region> g_regions;
int g_next_ordinal = 0;

}  // namespace

int next_ordinal() { return g_next_ordinal++; }

void reset_log() { g_log.clear(); }

void reset_case() {
    g_log.clear();
    g_regions.clear();
    g_next_ordinal = 0;
}
const std::vector<std::string>& log() { return g_log; }
void note(const std::string& line) { g_log.push_back(line); }

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

}  // namespace oracle_cuda

namespace {

std::string sz(std::size_t v) { return std::to_string(v); }

}  // namespace

const char* cudaGetErrorString(cudaError_t) { return "oracle: no error"; }
const char* cudaGetErrorName(cudaError_t) { return "cudaSuccess"; }
cudaError_t cudaGetLastError() { return cudaSuccess; }

cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count,
                            cudaStream_t) {
    oracle_cuda::note("memset " + oracle_cuda::where(dst) + " val=" +
                      std::to_string(value) + " len=" + sz(count));
    return cudaSuccess;
}

cudaError_t cudaMemset2DAsync(void* dst, std::size_t pitch, int value,
                              std::size_t width, std::size_t height,
                              cudaStream_t) {
    oracle_cuda::note("memset2d " + oracle_cuda::where(dst) + " val=" +
                      std::to_string(value) + " pitch=" + sz(pitch) +
                      " width=" + sz(width) + " rows=" + sz(height));
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                            cudaMemcpyKind kind, cudaStream_t) {
    oracle_cuda::note("memcpy dst=" + oracle_cuda::where(dst) + " src=" +
                      oracle_cuda::where(src) + " len=" + sz(count) +
                      " kind=" + std::to_string(static_cast<int>(kind)));
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DAsync(void* dst, std::size_t dpitch, const void* src,
                              std::size_t spitch, std::size_t width,
                              std::size_t height, cudaMemcpyKind kind,
                              cudaStream_t) {
    oracle_cuda::note("memcpy2d dst=" + oracle_cuda::where(dst) + " src=" +
                      oracle_cuda::where(src) + " dpitch=" + sz(dpitch) +
                      " spitch=" + sz(spitch) + " width=" + sz(width) +
                      " rows=" + sz(height) + " kind=" +
                      std::to_string(static_cast<int>(kind)));
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count,
                       cudaMemcpyKind kind) {
    return cudaMemcpyAsync(dst, src, count, kind, nullptr);
}

cudaError_t cudaMallocHost(void** ptr, std::size_t size) {
    *ptr = std::malloc(size == 0 ? 1 : size);
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
    *event = std::malloc(1);
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
    std::free(event);
    return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }

// ---- the device-memory allocator DeviceBuffer calls ------------------------
//
// Named by the harness immediately after each allocation (see oracle.cpp):
// `DeviceBuffer` does not know what it is holding, so the harness supplies the
// name and this layer only supplies the address.
namespace pie_cuda_driver {

// Buffers are named by ALLOCATION ORDINAL rather than by role, because the
// order is itself part of the contract: a stack with no linear layers skips
// the conv and recurrent slabs entirely, so its MTP tier is `buf0`. Naming
// them by role would hide that.
DeviceMemoryBlock allocate_device_memory(std::size_t bytes, std::size_t) {
    if (bytes == 0) return DeviceMemoryBlock{nullptr, false};
    void* p = std::malloc(bytes);
    const std::string name = "buf" + std::to_string(oracle_cuda::next_ordinal());
    oracle_cuda::name_region(p, bytes, name);
    oracle_cuda::note("alloc " + name + " bytes=" + std::to_string(bytes));
    return DeviceMemoryBlock{p, false};
}


void free_device_memory(DeviceMemoryBlock block) noexcept {
    oracle_cuda::note("free " + oracle_cuda::where(block.ptr));
    oracle_cuda::forget_region(block.ptr);
    std::free(block.ptr);
}

}  // namespace pie_cuda_driver
