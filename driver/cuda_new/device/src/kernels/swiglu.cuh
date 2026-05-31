#pragma once

// SwiGLU MLP activation: y = silu(gate) * up, where silu(x) = x*sigmoid(x).
// Elementwise; gate/up/y are bf16, all `num_elements` long. Launcher decl;
// body lifted verbatim from driver/cuda/src/kernels/swiglu.cu (base
// variant). GeGLU/GPT-OSS/chunked/MoE variants are lifted when their MLP
// paths land.

#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void swiglu_bf16(const void* gate, const void* up, void* y,
                 int num_elements, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
