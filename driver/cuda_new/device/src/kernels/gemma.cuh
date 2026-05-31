#pragma once

// Gemma-specific elementwise kernels. Launcher declarations; the kernel
// bodies live in gemma.cu, lifted verbatim from the legacy driver
// (driver/cuda/src/kernels/{rmsnorm,swiglu,softcap}.cu). The only changes
// are the namespace (pie_cuda_driver::kernels -> pie_cuda_device::kernels)
// and dropping the `launch_` prefix on each entry name.

#include <cstddef>

#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// Gemma RMSNorm: y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * (1 + weight).
// x / y: [num_rows, hidden] bf16 row-major; weight: [hidden] bf16. Uses the
// `(1 + w) * x_hat` convention (WEIGHT_PLUS_ONE), kept as a self-contained
// copy of the templated kernel rather than reusing the base rmsnorm.cu.
void rmsnorm_gemma_bf16(const void* x, const void* weight, void* y,
                        int num_rows, int hidden, float eps, cudaStream_t stream);

// Gemma MLP activation: y = gelu_tanh(gate) * up, elementwise over
// num_elements. gate / up / y: bf16.
void geglu_tanh_bf16(const void* gate, const void* up, void* y,
                     int num_elements, cudaStream_t stream);

// In-place logit soft-capping: x = cap * tanh(x / cap) over n bf16 elements.
void logit_softcap_bf16(void* x, float cap, std::size_t n, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
