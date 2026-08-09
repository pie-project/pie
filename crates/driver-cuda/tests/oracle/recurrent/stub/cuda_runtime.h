#pragma once

// A recording stand-in for <cuda_runtime.h>.
//
// The recurrent state cache's entire observable behaviour is the sequence of
// stream operations it issues: which buffer, at what offset, with what pitch,
// how wide and how many rows. None of that survives the call -- a
// `cudaMemset2DAsync` with the wrong pitch zeroes another layer's state and
// returns success. So the API is replaced with recorders and the operations
// themselves become the transcript.
//
// Found ahead of the real header because it is copied into $WORK and $WORK is
// first on the include path; the real CUDA include directory is never added.
// Only the surface `device_buffer.hpp` and `recurrent_state_cache.cpp`
// actually use is declared -- anything else they might reach for should fail
// to compile rather than silently no-op.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr unsigned int cudaEventDisableTiming = 2;

const char* cudaGetErrorString(cudaError_t err);
const char* cudaGetErrorName(cudaError_t err);
cudaError_t cudaGetLastError();

cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count,
                            cudaStream_t stream);
cudaError_t cudaMemset2DAsync(void* dst, std::size_t pitch, int value,
                              std::size_t width, std::size_t height,
                              cudaStream_t stream);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = nullptr);
cudaError_t cudaMemcpy2DAsync(void* dst, std::size_t dpitch, const void* src,
                              std::size_t spitch, std::size_t width,
                              std::size_t height, cudaMemcpyKind kind,
                              cudaStream_t stream);
cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count,
                       cudaMemcpyKind kind);

cudaError_t cudaMallocHost(void** ptr, std::size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);

// ---- the transcript --------------------------------------------------------
//
// Every recorded pointer is rendered as `<buffer>+<offset>` by looking it up
// in the table of allocations the recorder itself handed out. An operation on
// an address that belongs to no live allocation renders as a raw number, which
// is what a pointer-arithmetic bug would produce.
namespace oracle_cuda {

// Clears the transcript only -- the region table must survive, or every
// operation after the first drain renders as "unknown".
void reset_log();

// Clears the transcript, the region table and the allocation ordinal. Called
// once per case.
void reset_case();

// Next buffer ordinal, consumed by the allocator.
int next_ordinal();
const std::vector<std::string>& log();
void note(const std::string& line);

// Register `bytes` at `ptr` under `name`; subsequent operations inside that
// range render relative to it.
void name_region(const void* ptr, std::size_t bytes, const std::string& name);
void forget_region(const void* ptr);
std::string where(const void* ptr);

}  // namespace oracle_cuda
