#pragma once

// Composable, register-fused transcode framework (WEIGHT_LOADER_TODO.md A2.2).
//
// The loader's quant->quant transcode (e.g. FP8 checkpoint -> MXFP4 runtime
// quant) is a single IR `Transcode` node, but the executor lowers it to two
// device kernels through a BF16 scratch buffer in HBM: dequant (source->BF16)
// then encode (BF16->target). That BF16 round-trip triples HBM traffic.
//
// Rather than hand-write one fused kernel per (source, target) pair — an a x B
// explosion — this composes `a` Decode functors with `B` Encode functors in a
// single kernel template. The compiler emits the a x B specializations; the
// intermediate value lives in REGISTERS, never HBM. The "narrow waist" moves
// from "BF16 in global memory" to "float[32] in registers" — same composition
// contract the two-step relied on, minus the traffic.
//
// Numerics: Decode functors round the dequantized value through BF16 so a fused
// transcode is BIT-IDENTICAL to the existing two-step (which materialized a BF16
// intermediate). That both preserves the validated model numerics and lets a
// standalone parity test assert exact equality (tests/test_transcode_fused.cu).

#include <cstdint>
#include <math.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace pie_cuda_driver::kernels::transcode {

// E8M0 stores a biased power-of-two exponent: value = 2^(byte - kE8M0Bias).
inline constexpr int kE8M0Bias = 127;

// ---- MXFP4 encode primitives (identical to quant_bf16_to_mxfp4.cu) ---------
// FP4 E2M1: magnitudes {0,0.5,1,1.5,2,3,4,6}; sign in the MSB of the nibble.
__device__ __forceinline__ unsigned encode_fp4_e2m1(float x)
{
    const float a = fabsf(x);
    unsigned mag;
    if      (a < 0.25f) mag = 0;
    else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2;
    else if (a < 1.75f) mag = 3;
    else if (a < 2.5f)  mag = 4;
    else if (a < 3.5f)  mag = 5;
    else if (a < 5.0f)  mag = 6;
    else                mag = 7;
    const unsigned sign = (x < 0.0f) ? 0x8u : 0x0u;
    return (mag == 0) ? 0u : (sign | mag);
}

__device__ __forceinline__ unsigned char encode_e8m0(float absmax)
{
    if (!(absmax > 0.0f)) return 0;  // absmax == 0 or NaN
    const float l = log2f(absmax / 6.0f);
    int b = static_cast<int>(ceilf(l)) + kE8M0Bias;
    if (b < 0)   b = 0;
    if (b > 254) b = 254;
    return static_cast<unsigned char>(b);
}

// ---- Decode functors: source element -> float (rounded through BF16) -------

// Raw BF16 source (no source scale).
struct DecodeBf16 {
    const __nv_bfloat16* __restrict__ src;
    int cols;
    __device__ __forceinline__ float load(int row, int col) const
    {
        return __bfloat162float(src[static_cast<std::size_t>(row) * cols + col]);
    }
};

// FP8 E4M3 with a per-group (block) FP32 scale, matching
// dequant_fp8_e4m3_per_group: scale index = [row/gs][col/gs]. The value is
// rounded through BF16 so the result matches the FP8->BF16->MXFP4 two-step.
struct DecodeFp8E4m3PerGroup {
    const __nv_fp8_storage_t* __restrict__ src;
    const float* __restrict__ scales;
    int cols;
    int scale_cols;
    int group_size;
    __device__ __forceinline__ float load(int row, int col) const
    {
        const int sr = row / group_size;
        const int sc = col / group_size;
        const float s = scales[static_cast<std::size_t>(sr) * scale_cols + sc];
        const __half h = __nv_cvt_fp8_to_halfraw(
            src[static_cast<std::size_t>(row) * cols + col], __NV_E4M3);
        return __bfloat162float(__float2bfloat16(__half2float(h) * s));
    }
};

// ---- Encode functor: 32 floats -> E8M0 scale byte + 16 packed E2M1 bytes ---
// Each Encode advertises its group width as `kGroup`; the dispatch instantiates
// the kernel template with it, so encode_group's array size always matches.
struct EncodeMxfp4 {
    static constexpr int kGroup = 32;        // values per E8M0 scale block
    static constexpr int kPackedPerByte = 2;  // E2M1 nibbles packed per output byte
    static constexpr int kBytesPerGroup = kGroup / kPackedPerByte;  // 16
    std::uint8_t* __restrict__ packed;  // [rows, cols/kPackedPerByte]
    std::uint8_t* __restrict__ scales;  // [rows, cols/kGroup]
    int cols;
    __device__ __forceinline__ void encode_group(
        const float (&vals)[kGroup], int row, int g) const
    {
        float absmax = 0.0f;
        #pragma unroll
        for (int k = 0; k < kGroup; ++k) {
            const float a = fabsf(vals[k]);
            if (a > absmax) absmax = a;
        }
        const unsigned char sb = encode_e8m0(absmax);
        scales[static_cast<std::size_t>(row) * (cols / kGroup) + g] = sb;
        const float s = ldexpf(1.0f, static_cast<int>(sb) - kE8M0Bias);
        const float inv_s = (s == 0.0f) ? 0.0f : (1.0f / s);
        const std::size_t po = static_cast<std::size_t>(row) * (cols / kPackedPerByte)
            + static_cast<std::size_t>(g) * kBytesPerGroup;
        #pragma unroll
        for (int k = 0; k < kBytesPerGroup; ++k) {
            const unsigned lo = encode_fp4_e2m1(vals[kPackedPerByte * k]     * inv_s);
            const unsigned hi = encode_fp4_e2m1(vals[kPackedPerByte * k + 1] * inv_s);
            packed[po + k] = static_cast<std::uint8_t>((hi << 4) | (lo & 0xFu));
        }
    }
};

// ---- Composable kernel: Decode -> register group of GROUP -> Encode --------
// One block per row; threads stride over the GROUP-wide blocks. The intermediate
// float[GROUP] never leaves registers. GROUP is the target's group width
// (Encode::kGroup) — 32 for MXFP4, 16 for NVFP4, etc.
template <int GROUP, typename Decode, typename Encode>
__global__ void transcode_rowmajor_kernel(Decode dec, Encode enc, int cols)
{
    const int row = blockIdx.x;
    const int groups = cols / GROUP;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const int base = g * GROUP;
        float vals[GROUP];
        #pragma unroll
        for (int k = 0; k < GROUP; ++k) {
            vals[k] = dec.load(row, base + k);
        }
        enc.encode_group(vals, row, g);
    }
}

}  // namespace pie_cuda_driver::kernels::transcode
