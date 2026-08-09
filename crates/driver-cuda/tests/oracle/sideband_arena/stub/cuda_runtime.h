// Stub <cuda_runtime.h>: `hook_sideband_arena.cpp` touches exactly three entry
// points. They are DEFINED by the oracle rather than linked from the real
// runtime, for two reasons: the harness must run without a GPU, and — the
// point of this particular oracle — the allocator has to be MADE TO FAIL on
// command. The out-of-memory path is the one that frees the old block before
// discovering it cannot replace it, and no real allocator can be asked to
// reach that state on cue.
#pragma once
#include <cstddef>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;

using cudaStream_t = struct CUstream_st*;

cudaError_t cudaMalloc(void** ptr, std::size_t bytes);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
