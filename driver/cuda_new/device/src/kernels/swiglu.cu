#include "swiglu.cuh"

#include <cuda_bf16.h>

// Lifted verbatim from driver/cuda/src/kernels/swiglu.cu (base variant).

namespace pie_cuda_device::kernels {

namespace {

__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ y,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = __bfloat162float(gate[idx]);
    const float u = __bfloat162float(up[idx]);
    const float silu = g / (1.f + expf(-g));
    y[idx] = __float2bfloat16(silu * u);
}

}  // namespace

void swiglu_bf16(const void* gate, const void* up, void* y,
                 int num_elements, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    swiglu_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up),
        static_cast<__nv_bfloat16*>(y),
        num_elements);
}

}  // namespace pie_cuda_device::kernels
