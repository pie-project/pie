#ifndef PIE_NVRTC_CUDA_RUNTIME_H_
#define PIE_NVRTC_CUDA_RUNTIME_H_

#include <cstdint>

typedef unsigned short ushort;
typedef unsigned char uchar;

// Programmatic dependent launch has no HIP counterpart: a kernel starts
// only once the one before it on the stream has finished, which is what
// the wait waits for, so both are empty.
__device__ __forceinline__ void cudaGridDependencySynchronize() {}

__device__ __forceinline__ void cudaTriggerProgrammaticLaunchCompletion() {}

#endif
