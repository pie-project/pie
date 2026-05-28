#include "kernels/quant_bf16_to_mxfp4.hpp"

#include <cuda_bf16.h>
#include <math.h>

namespace pie_cuda_driver::kernels {

namespace {

// One CUDA block handles one row. Each thread strides over the row in
// 32-element groups; for each group it computes absmax via warp shuffles,
// derives the E8M0 byte scale, writes scale, then quantizes & packs the
// 32 nibbles into 16 bytes.
constexpr int BLOCK = 256;

// FP4 E2M1 absolute values in increasing order. Index in this table doubles
// as the magnitude index (3 bits). The sign bit goes in the MSB of the
// 4-bit codepoint.
__device__ __forceinline__ unsigned encode_fp4_e2m1(float x)
{
    // Magnitudes:                0    0.5  1.0  1.5  2.0  3.0  4.0  6.0
    // Midpoints (used as boundaries) between consecutive magnitudes:
    //   0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    // Anything ≥ 5.0 rounds to 6.0 (codepoint 7).
    const float a = fabsf(x);
    unsigned mag;
    if      (a < 0.25f) mag = 0;  // 0
    else if (a < 0.75f) mag = 1;  // 0.5
    else if (a < 1.25f) mag = 2;  // 1.0
    else if (a < 1.75f) mag = 3;  // 1.5
    else if (a < 2.5f)  mag = 4;  // 2.0
    else if (a < 3.5f)  mag = 5;  // 3.0
    else if (a < 5.0f)  mag = 6;  // 4.0
    else                mag = 7;  // 6.0
    const unsigned sign = (x < 0.0f) ? 0x8u : 0x0u;
    // Signed zero rounds to +0.
    return (mag == 0) ? 0u : (sign | mag);
}

__device__ __forceinline__ unsigned char encode_e8m0(float absmax)
{
    // E8M0 byte b encodes 2^(b - 127); we pick the smallest b such that
    // 6 * 2^(b - 127) >= absmax  ⇒  b = ceil(log2(absmax / 6)) + 127.
    // Clamp to [0, 254]; 255 is reserved (NaN) per the OCP spec.
    if (!(absmax > 0.0f)) return 0;  // covers absmax==0 and NaN
    const float l = log2f(absmax / 6.0f);
    int b = static_cast<int>(ceilf(l)) + 127;
    if (b < 0)   b = 0;
    if (b > 254) b = 254;
    return static_cast<unsigned char>(b);
}

__global__ void quant_bf16_to_mxfp4_row_kernel(
    const __nv_bfloat16* __restrict__ src,    // [rows, cols]
    std::uint8_t*       __restrict__ packed,  // [rows, cols/2]
    std::uint8_t*       __restrict__ scales,  // [rows, cols/32]
    int                              cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups = cols / 32;
    const std::size_t row_src    = static_cast<std::size_t>(row) * cols;
    const std::size_t row_packed = static_cast<std::size_t>(row) * (cols / 2);
    const std::size_t row_scale  = static_cast<std::size_t>(row) * groups;

    for (int g = tid; g < groups; g += BLOCK) {
        const int base = g * 32;
        // 1) absmax over the 32 elements (single thread does it — simple).
        float absmax = 0.0f;
        float vals[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float v = __bfloat162float(src[row_src + base + k]);
            vals[k] = v;
            const float a = fabsf(v);
            if (a > absmax) absmax = a;
        }
        // 2) E8M0 byte scale.
        const unsigned char sb = encode_e8m0(absmax);
        scales[row_scale + g] = sb;
        // Reconstruct the actual scale factor; clamp the divisor to 1 ulp
        // to avoid NaN when absmax is 0 (sb=0 ⇒ s=2^-127, ~5.88e-39).
        const float s = ldexpf(1.0f, static_cast<int>(sb) - 127);
        const float inv_s = (s == 0.0f) ? 0.0f : (1.0f / s);
        // 3) Encode 32 nibbles and pack into 16 bytes.
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            const unsigned lo = encode_fp4_e2m1(vals[2 * k]     * inv_s);
            const unsigned hi = encode_fp4_e2m1(vals[2 * k + 1] * inv_s);
            packed[row_packed + g * 16 + k] =
                static_cast<std::uint8_t>((hi << 4) | (lo & 0xFu));
        }
    }
}

}  // namespace

void quantize_bf16_to_mxfp4_e2m1_per_block(
    const void* W_bf16, std::uint8_t* W_packed, std::uint8_t* W_scale_e8m0,
    int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    quant_bf16_to_mxfp4_row_kernel<<<rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16),
        W_packed, W_scale_e8m0, cols);
}

}  // namespace pie_cuda_driver::kernels
