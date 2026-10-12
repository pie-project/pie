#pragma once

// HIP graphs have no conditional node, so there is no device-side
// `cudaGraphSetConditional`. On ROCm a conditional's handle is the device
// address of one 32-bit word, and setting it stores the arm there: the engine
// reads the word to pick the body a CUDA graph would have branched to.
__device__ __forceinline__ void cudaGraphSetConditional(
    unsigned long long handle,
    unsigned int value)
{
    if (handle != 0) {
        __hip_atomic_store(reinterpret_cast<unsigned int*>(handle), value, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

namespace pie::graph {

__global__ void set_conditional(
    unsigned long long handle,
    const int* __restrict__ indptr,
    int lanes,
    unsigned int absent,
    int arm,
    const unsigned int* __restrict__ win)
{
    if (arm == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int at = (win != nullptr) ? static_cast<int>(win[2]) : lanes;

    unsigned int value = absent;
    if (indptr != nullptr && at >= 0) {
        value = (indptr[at] != 0) ? 1u : 0u;
    }
    cudaGraphSetConditional(handle, value);
}

__global__ void set_conditional_byte(
    unsigned long long handle,
    const unsigned char* __restrict__ live,
    unsigned int absent,
    int arm)
{
    if (arm == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const unsigned int value = (live == nullptr) ? absent : ((*live != 0) ? 1u : 0u);
    cudaGraphSetConditional(handle, value);
}

}

namespace pie::graph {

__global__ void set_switch(
    unsigned long long handle,
    unsigned int arm,
    const int* __restrict__ indptr,
    int lanes,
    int armed,
    const unsigned int* __restrict__ win)
{
    if (armed == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int at = (win != nullptr) ? static_cast<int>(win[2]) : lanes;
    if (indptr == nullptr || at < 0) return;
    if (indptr[at] == 0) return;

    cudaGraphSetConditional(handle, arm);
}

}
