#pragma once

// Row-wise RMSNorm over bf16. Launcher declaration; the kernel body lives
// in rmsnorm.cu, lifted verbatim from driver/cuda/src/kernels/rmsnorm.cu
// (the base, non-Gemma, non-fused variant). The fused variants are lifted
// later as the forward bodies that need them land.

#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * weight
// x / y: [num_rows, hidden] bf16 row-major; weight: [hidden] bf16.
void rmsnorm_bf16(const void* x, const void* weight, void* y,
                  int num_rows, int hidden, float eps, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
