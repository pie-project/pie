// Stub <cuda_runtime.h>: the planner touches exactly three entry points, and
// linking the real runtime would make the oracle depend on a GPU being present
// -- the opposite of what this harness is for. The three are DEFINED by the
// driver so the transcript can sweep device shapes no single machine has.
#pragma once
#include <cstddef>
#include <cstring>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    int multiProcessorCount;
    std::size_t totalGlobalMem;
};

cudaError_t cudaGetDevice(int* dev);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int dev);
cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes);
const char* cudaGetErrorString(cudaError_t e);

using cudaStream_t = struct CUstream_st*;

#include "cuda_check.hpp"

