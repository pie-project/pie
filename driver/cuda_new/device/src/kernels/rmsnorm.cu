#include "rmsnorm.cuh"

#include <cuda_bf16.h>

// Lifted from driver/cuda/src/kernels/rmsnorm.cu (base variant). The only
// change is dropping the WEIGHT_PLUS_ONE template parameter — the Gemma
// `(1 + w)` convention is a separate entry point lifted when Gemma lands.

namespace pie_cuda_device::kernels {

namespace {

// One block per row. `BLOCK` threads cooperate on the L2-norm reduction;
// each thread handles `hidden / BLOCK` elements.
template <int BLOCK>
__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* xr = x + row * hidden;
    __nv_bfloat16* yr = y + row * hidden;

    // L2 norm across the row, accumulated in fp32.
    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    // Block-wide reduction via shared memory.
    __shared__ float buf[BLOCK];
    buf[tid] = local;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = __bfloat162float(xr[i]);
        const float wv = __bfloat162float(weight[i]);
        yr[i] = __float2bfloat16(xv * inv_rms * wv);
    }
}

}  // namespace

void rmsnorm_bf16(const void* x, const void* weight, void* y,
                  int num_rows, int hidden, float eps, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    rmsnorm_bf16_kernel<BLOCK><<<dim3(num_rows), dim3(BLOCK), 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, eps);
}

}  // namespace pie_cuda_device::kernels
