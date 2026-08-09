// Stub <cuda_runtime.h> for the page-mask oracle.
//
// Covers the four entry points `attn_page_mask.cu` and
// `hook_sideband_arena.cpp` touch between them. `cudaMemsetAsync` is here
// because `begin_layer` seeds the keep rows with it, and the oracle needs to
// see the *extent* of that seed — a memset short by one row leaves stale keep
// bits governing a request, which is a silent eviction bug.
#pragma once
#include <cstddef>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;

using cudaStream_t = struct CUstream_st*;

cudaError_t cudaMalloc(void** ptr, std::size_t bytes);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaMemsetAsync(
    void* ptr, int value, std::size_t bytes, cudaStream_t stream);
