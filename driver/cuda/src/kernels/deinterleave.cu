#include "kernels/deinterleave.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

__global__ void deinterleave_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ fused,
    __nv_bfloat16* __restrict__       gate_out,
    __nv_bfloat16* __restrict__       up_out,
    int I, int H)
{
    const int row = blockIdx.x;  // 0 .. I-1
    if (row >= I) return;
    const __nv_bfloat16* gate_src = fused + (2 * row    ) * H;
    const __nv_bfloat16* up_src   = fused + (2 * row + 1) * H;
    __nv_bfloat16* gate_dst = gate_out + row * H;
    __nv_bfloat16* up_dst   = up_out   + row * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        gate_dst[j] = gate_src[j];
        up_dst[j]   = up_src[j];
    }
}

__global__ void deinterleave_vec_bf16_kernel(
    const __nv_bfloat16* __restrict__ fused,
    __nv_bfloat16* __restrict__       gate_out,
    __nv_bfloat16* __restrict__       up_out,
    int I)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    gate_out[i] = fused[2 * i];
    up_out[i]   = fused[2 * i + 1];
}

}  // namespace

void launch_deinterleave_rows_bf16(
    const void* fused, void* gate_out, void* up_out,
    int I, int H, cudaStream_t stream)
{
    if (I <= 0 || H <= 0) return;
    const int block = (H < 128) ? 32 : (H > 256 ? 256 : 128);
    deinterleave_rows_bf16_kernel<<<I, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<__nv_bfloat16*>(gate_out),
        static_cast<__nv_bfloat16*>(up_out),
        I, H);
}

void launch_deinterleave_vec_bf16(
    const void* fused, void* gate_out, void* up_out,
    int I, cudaStream_t stream)
{
    if (I <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (I + BLOCK - 1) / BLOCK;
    deinterleave_vec_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<__nv_bfloat16*>(gate_out),
        static_cast<__nv_bfloat16*>(up_out),
        I);
}

}  // namespace pie_cuda_driver::kernels
