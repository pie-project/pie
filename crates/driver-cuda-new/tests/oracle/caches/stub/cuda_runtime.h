#pragma once

// A recording stand-in for <cuda_runtime.h>, for the three allocation-side
// cache builders.
//
// What has to be observable here is different from the recurrent cache's
// stream traffic: these files allocate, and the allocation *sizes* and their
// *order* are the whole contract. `cudaMallocHost` returning a pointer throws
// away the request size, and `cudaMemset` failing on an uncommitted arena
// range is a control-flow event the shipping code deliberately swallows -- so
// both are recorded, and the memset is made failable so the swallow itself can
// be exercised.
//
// Found ahead of the real header because it is copied into $WORK and $WORK is
// first on the include path; the real CUDA include directory is never added.
// Only the surface the three files under test actually use is declared.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using cudaError_t = int;
using cudaStream_t = void*;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaMemcpySrcAccessOrder {
    cudaMemcpySrcAccessOrderInvalid = 0,
    cudaMemcpySrcAccessOrderStream = 1,
};

struct cudaMemcpyAttributes {
    cudaMemcpySrcAccessOrder srcAccessOrder = cudaMemcpySrcAccessOrderInvalid;
};

inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr cudaError_t cudaErrorInvalidValue = 1;
inline constexpr unsigned int cudaStreamNonBlocking = 1;

// `swap_pool.cpp` forks on this: >= 13000 selects `cudaMemcpyBatchAsync`,
// below it the per-copy loop. Pinned at 13000 so the batch path is the one
// that compiles, matching the toolkit this crate is built against. The
// copy half is proved separately by tests/oracle/store; what matters here is
// only that the file still compiles as a whole.
#define CUDART_VERSION 13000

const char* cudaGetErrorString(cudaError_t err);
const char* cudaGetErrorName(cudaError_t err);
cudaError_t cudaGetLastError();

cudaError_t cudaMalloc(void** ptr, std::size_t size);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaMallocHost(void** ptr, std::size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMemset(void* dst, int value, std::size_t count);
cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count,
                            cudaStream_t stream);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = nullptr);
cudaError_t cudaMemcpyBatchAsync(void* const* dsts, const void* const* srcs,
                                 const std::size_t* sizes, std::size_t count,
                                 cudaMemcpyAttributes* attrs,
                                 std::size_t* attrsIdxs, std::size_t numAttrs,
                                 cudaStream_t stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

// ---- the transcript --------------------------------------------------------
namespace oracle_cuda {

// Clears the transcript, the region table, the ordinals and the injected
// memset failure. Called once per case.
void reset_case();

const std::vector<std::string>& log();
void note(const std::string& line);

// Register `bytes` at `ptr` under `name`; operations inside the range render
// as `<name>+<offset>`.
void name_region(const void* ptr, std::size_t bytes, const std::string& name);
void forget_region(const void* ptr);
std::string where(const void* ptr);

// Next pinned-host-buffer ordinal, consumed by `cudaMallocHost`.
int next_host_ordinal();

// Make the Nth (0-based) `cudaMemset` of the case fail. Negative disables.
//
// This exists because `dsv4_compress_cache.cpp`'s zeroing pass is explicitly
// best-effort -- the tensors sit in a reservation whose physical pages are
// committed on demand, so a failing memset is the EXPECTED case in production,
// not an error path. Its reaction (`cudaGetLastError` then `break`, abandoning
// the rest of that layer but not the cache) cannot be observed at all unless
// a failure can be provoked.
void fail_memset_at(int n);

}  // namespace oracle_cuda
