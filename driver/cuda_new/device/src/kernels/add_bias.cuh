#pragma once

// In-place broadcast bias-add over a [num_tokens, dim] bf16 row-major tensor:
//   x[t*dim + i] = round_bf16(x[t*dim + i] + bias[i])
// for every token row t in [0,num_tokens) and channel i in [0,dim). The bias
// is a [dim] bf16 vector broadcast across rows. The add is performed in fp32
// (decode both operands, sum, then round to bf16). Used to add the q/k/v
// projection biases after the fused qkv GEMM in Qwen2-style attention.

#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void add_bias_bf16(void* x, const void* bias, int num_tokens, int dim, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
