//===-- quant_bf16_to_mxfp4.cuh - the MXFP4 packer, as device text -------===//
//
// One `__global__` template and the two encoders it calls.
// `quant_bf16_to_mxfp4.cu` is the single launcher `model-loader` calls by
// name; it includes this file rather than holding a second copy of the
// packer, because the E2M1 boundary table and the E8M0 exponent rule are
// exactly the kind of constant that drifts silently when it exists twice.
//
// # What the launcher was doing, and where it went
//
//     if (rows == 0 || cols == 0) return;
//     kernel<<<rows, 256, 0, stream>>>(src, packed, scales, cols);
//
// which is `LaunchRule::RouteRows` -- one block per row, block width picked
// from the row width. The launcher is NOT deleted: `model-loader` calls
// `quantize_bf16_to_mxfp4_e2m1_per_block` directly from Rust while the
// checkpoint path is still ahead-of-time, and this migration splits device
// text out rather than removing entry points.
//
// `RouteRows` picks the block width rather than fixing it at 256, so the
// group loop strides by `blockDim.x`. That is the transformation
// `altup_aux.cuh` blesses and it is arithmetically inert: the ahead-of-time
// launcher still passes 256, so the stride is the stride it always was.
//
// # Why `<math.h>` is gone
//
// It was there for `fabsf`, `log2f`, `ceilf` and `ldexpf`. NVRTC ships no C
// standard library -- 0 of 31 standard headers answered when it was measured
// -- but it does not need to: all four are device builtins the compiler knows
// without a declaration, under nvcc as well. The include was buying nothing
// and would have made this file uncompilable at run time.
//
// # Why it is a template when the original was not
//
// The original wrote `device::bf16` because an ahead-of-time build has to
// choose. Nothing below is bf16-specific past the widening, so the kernel is
// written over `T` and an fp16 checkpoint packer costs a row rather than a
// translation unit.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::quant::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u8;
using ::pie_cuda_driver::kernels::device::usize;

/// FP4 E2M1, as a nibble.
///
/// The comparison ladder IS the rounding rule and not an implementation of
/// one: E2M1's eight magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6, and the
/// boundaries below are their midpoints, so a value lands on the nearer
/// magnitude and a tie lands on the larger. Anything at or above 5 saturates
/// to 6 -- E2M1 has no infinity to overflow to.
///
/// Signed zero encodes as +0. A negative zero nibble is a legal E2M1 pattern
/// that no dequantiser distinguishes, and emitting it would make two
/// bit-different packings of the same tensor.
__device__ __forceinline__ unsigned encode_fp4_e2m1(float x) {
    const float a = fabsf(x);
    unsigned mag;
    if (a < 0.25f) {
        mag = 0;
    } else if (a < 0.75f) {
        mag = 1;
    } else if (a < 1.25f) {
        mag = 2;
    } else if (a < 1.75f) {
        mag = 3;
    } else if (a < 2.5f) {
        mag = 4;
    } else if (a < 3.5f) {
        mag = 5;
    } else if (a < 5.0f) {
        mag = 6;
    } else {
        mag = 7;
    }
    const unsigned sign = (x < 0.0f) ? 0x8u : 0x0u;
    return (mag == 0) ? 0u : (sign | mag);
}

/// The E8M0 block scale: byte `b` denotes `2^(b - 127)`.
///
/// The smallest `b` whose block maximum 6 * 2^(b-127) covers `absmax`, which
/// is `ceil(log2(absmax / 6)) + 127`. Clamped to [0, 254] because 255 is
/// reserved for NaN by the OCP microscaling specification, and a scale byte
/// of 255 is a tensor no conforming reader will load.
///
/// The `!(absmax > 0)` test rather than `absmax == 0` catches NaN as well,
/// which `log2f` would otherwise turn into an out-of-range exponent.
__device__ __forceinline__ u8 encode_e8m0(float absmax) {
    if (!(absmax > 0.0f)) return 0;
    const float l = log2f(absmax / 6.0f);
    int b = static_cast<int>(ceilf(l)) + 127;
    if (b < 0) b = 0;
    if (b > 254) b = 254;
    return static_cast<u8>(b);
}

/// One row per block; 32 elements per group, packed two nibbles to a byte.
///
/// Each thread owns whole groups, so the absmax is a serial fold over 32
/// registers rather than a warp reduction -- the group is smaller than a
/// warp and a shuffle would cost more than it saves. The fold order is the
/// original's, element 0 upward, because a max over a row with a NaN in it
/// depends on that order.
template <class T>
__global__ void quant_bf16_to_mxfp4_row(
    const T* __restrict__ src,
    u8* __restrict__ packed,
    u8* __restrict__ scales,
    i32 cols) {
    const i32 row = blockIdx.x;
    const i32 groups = cols / 32;
    const usize row_src = static_cast<usize>(row) * cols;
    const usize row_packed = static_cast<usize>(row) * (cols / 2);
    const usize row_scale = static_cast<usize>(row) * groups;

    for (i32 g = threadIdx.x; g < groups; g += blockDim.x) {
        const i32 base = g * 32;
        float absmax = 0.0f;
        float vals[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const float v = Elem<T>::to_f32(src[row_src + base + k]);
            vals[k] = v;
            const float a = fabsf(v);
            if (a > absmax) absmax = a;
        }
        const u8 sb = encode_e8m0(absmax);
        scales[row_scale + g] = sb;
        // `sb == 0` is 2^-127, which is finite and normal in fp32, so the
        // reciprocal below cannot divide by zero -- the guard is kept
        // because the original had it and removing it would change nothing
        // except the reader's confidence.
        const float s = ldexpf(1.0f, static_cast<int>(sb) - 127);
        const float inv_s = (s == 0.0f) ? 0.0f : (1.0f / s);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const unsigned lo = encode_fp4_e2m1(vals[2 * k] * inv_s);
            const unsigned hi = encode_fp4_e2m1(vals[2 * k + 1] * inv_s);
            packed[row_packed + g * 16 + k] = static_cast<u8>((hi << 4) | (lo & 0xFu));
        }
    }
}

}  // namespace pie_cuda_driver::kernels::quant::device
