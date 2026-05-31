#include "dequant_wna16.cuh"

#include <cuda_bf16.h>

// Lifted from driver/cuda/src/kernels/dequant_wna16.cu (base load-time dequant
// variant). Verbatim apart from the namespace and the dropped `launch_` prefix;
// the `wna16_load_int4b8` device helper and the `_gate_up_decode` /
// `_down_decode` fused kernels/launchers are omitted (only the dequant kernel
// is reachable from this entry point).

namespace pie_cuda_device::kernels {

namespace {

__global__ void dequant_wna16_int4b8_kernel(
    const std::int32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int out_dim,
    int in_dim,
    int group_size)
{
    const int row = blockIdx.y;
    const int word_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int words_per_row = in_dim / 8;
    if (row >= out_dim || word_col >= words_per_row) return;

    const int word = packed[static_cast<long long>(row) * words_per_row + word_col];
    const int k_base = word_col * 8;
    __nv_bfloat16* row_out = out + static_cast<long long>(row) * in_dim;
    const __nv_bfloat16* row_scale =
        scale + static_cast<long long>(row) * (in_dim / group_size);

#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        const int k = k_base + lane;
        const int nibble = (word >> (lane * 4)) & 0xF;
        const float q = static_cast<float>(nibble - 8);
        const float s = __bfloat162float(row_scale[k / group_size]);
        row_out[k] = __float2bfloat16(q * s);
    }
}

}  // namespace

void dequant_wna16_int4b8_to_bf16(
    const std::int32_t* packed,
    const void* scale_bf16,
    void* out_bf16,
    int out_dim,
    int in_dim,
    int group_size,
    cudaStream_t stream)
{
    if (out_dim <= 0 || in_dim <= 0 || group_size <= 0) return;
    if (in_dim % 8 != 0 || in_dim % group_size != 0) return;
    constexpr int BLOCK = 128;
    const int words_per_row = in_dim / 8;
    dim3 grid((words_per_row + BLOCK - 1) / BLOCK, out_dim);
    dequant_wna16_int4b8_kernel<<<grid, BLOCK, 0, stream>>>(
        packed,
        static_cast<const __nv_bfloat16*>(scale_bf16),
        static_cast<__nv_bfloat16*>(out_bf16),
        out_dim,
        in_dim,
        group_size);
}

}  // namespace pie_cuda_device::kernels
