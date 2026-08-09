// Stub <cuda_runtime.h> for the score-capture oracle.
//
// The page-mask stub's four entry points plus the two the score capture
// adds: `cudaMemcpyAsync` (the CSR upload — a capture whose upload writes
// the wrong extent replays a stale channel view of the KV lengths) and the
// memcpy kind enum it is called with.
#pragma once
#include <cstddef>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;

using cudaStream_t = struct CUstream_st*;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

cudaError_t cudaMalloc(void** ptr, std::size_t bytes);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaMemsetAsync(
    void* ptr, int value, std::size_t bytes, cudaStream_t stream);
cudaError_t cudaMemcpyAsync(
    void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind,
    cudaStream_t stream);
