#pragma once

// SwiGLU activation: y = silu(gate) * up.
//
//     silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Element-wise. `gate`, `up`, `y` are bf16 row-major, all the same size.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_swiglu_bf16(
    const void* gate,
    const void* up,
    void* y,
    int num_elements,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
