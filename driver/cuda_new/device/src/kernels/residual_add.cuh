#pragma once

// In-place elementwise bf16 add: y[i] = round_bf16(y[i] + x[i]) over n
// elements. Launcher decl; body lifted verbatim from
// driver/cuda/src/kernels/residual_add.cu.

#include <cstddef>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void residual_add_bf16(void* y, const void* x, std::size_t n, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
