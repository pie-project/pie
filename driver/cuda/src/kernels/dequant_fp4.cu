#include "kernels/dequant_fp4.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// E2M1 codepoint → fp32 LUT. Index is the 4-bit code (high bit = sign,
// next two = exponent biased at 1, low bit = mantissa). Matches OCP's
// MX FP4 spec.
__device__ __constant__ float kFp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

// One block per output row; each thread strides by 32 elements (the
// block-scale granularity). Per-32-element block: read the E8M0 scale,
// dequantize 16 packed-byte pairs into 32 bf16 elements.
__global__ void dequant_mxfp4_kernel(
    const std::uint8_t* __restrict__ packed,
    const std::uint8_t* __restrict__ block_scale,
    __nv_bfloat16*      __restrict__ out,
    int                 in_dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = in_dim / 32;

    const std::uint8_t* row_packed = packed + static_cast<long long>(row) * (in_dim / 2);
    const std::uint8_t* row_scale  = block_scale + static_cast<long long>(row) * blocks_per_row;
    __nv_bfloat16*      row_out    = out + static_cast<long long>(row) * in_dim;

    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        const std::uint8_t e8m0 = row_scale[blk];
        // E8M0: byte b → scale = 2^(b - 127). 0xFF is reserved for NaN
        // in the MX spec; we let exp2f overflow → +inf and downstream
        // bf16 saturation handle it (matches reference impls).
        const float scale = exp2f(static_cast<float>(static_cast<int>(e8m0)) - 127.f);

        const int packed_base = blk * 16;       // 16 bytes hold 32 fp4 codes
        const int out_base    = blk * 32;
        for (int i = 0; i < 16; ++i) {
            const std::uint8_t b = row_packed[packed_base + i];
            const float v_lo = kFp4Lut[b & 0xF] * scale;
            const float v_hi = kFp4Lut[b >> 4]  * scale;
            row_out[out_base + 2 * i + 0] = __float2bfloat16(v_lo);
            row_out[out_base + 2 * i + 1] = __float2bfloat16(v_hi);
        }
    }
}

}  // namespace

void launch_dequant_mxfp4_to_bf16(
    const std::uint8_t* packed, const std::uint8_t* block_scale,
    void* out, int out_dim, int in_dim, cudaStream_t stream)
{
    if (out_dim <= 0 || in_dim <= 0) return;
    if (in_dim % 32 != 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(out_dim);
    dim3 block(BLOCK);
    dequant_mxfp4_kernel<<<grid, block, 0, stream>>>(
        packed, block_scale,
        static_cast<__nv_bfloat16*>(out), in_dim);
}

}  // namespace pie_cuda_driver::kernels
