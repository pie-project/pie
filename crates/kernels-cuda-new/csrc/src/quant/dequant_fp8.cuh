//===-- dequant_fp8.cuh - E4M3 weight expansion, as device text ----------===//
//
// Three `__global__` templates and nothing else. `dequant_fp8.cu` is now the
// four host launchers that fire them ahead of time; the rows in
// `kernels_cuda_new::families::quant` fire the same templates at run time.
// There is exactly one definition of each -- the `.cu` includes this file
// rather than holding a second copy, because two copies that agree today
// drift tomorrow and `norm/altup_aux` shipped a release proving it.
//
// # Why this file was blocked, and what unblocked it
//
// `__nv_cvt_fp8_to_halfraw` is a hardware conversion, not arithmetic, so
// `new-horizon.md` §10.5 could not rewrite it in terms of the prelude -- the
// rounding is E4M3's and belongs to the format. Until the fp8 shims existed
// NVRTC had no `cuda_fp8.h` to resolve, and this file was listed as blocked.
// §15 closed that: `pie_fp8.cuh` forwards to `cuda_fp16.h` then `cuda_fp8.h`,
// which is the order `cuda_fp8.h` requires and deliberately does not perform
// itself, and every conversion in it was measured bit-identical to NVIDIA's
// over all 256 E4M3 byte patterns. So the intrinsic stays spelled the way it
// always was and the include is chosen per compiler.
//
// The guard below is not a stopgap. `kernels-cuda/csrc/CMakeLists.txt` puts
// this directory on the ahead-of-time build's path with `-iquote`, which
// answers `#include "..."` and is never searched for `#include <...>` --
// precisely so that a shim wearing NVIDIA's filename cannot shadow NVIDIA's
// real `cuda_fp16.h` for every translation unit in the tree. So nvcc reads
// the toolkit's headers, which is what the shim was measured against, and
// NVRTC -- which has no include path at all -- resolves `pie_fp8.cuh` out of
// the carried set.
//
// # What the launchers were doing, and where it went
//
//     dequant_fp8_e4m3_kernel        <<<(n + 255) / 256, 256>>>, `n == 0` guard
//     dequant_fp8_e4m3_per_channel   <<<rows, 256>>>, one row per block
//     dequant_fp8_e4m3_blocked       <<<rows, 256>>>, one row per block
//
// which are `LaunchRule::Elementwise` and `LaunchRule::RouteRows`. The
// `n == 0` guard is `Ungeometric::Empty`, which `Elementwise` already returns
// and the binder already refuses on.
//
// `RouteRows` picks the block width from the row width rather than fixing it
// at 256, so the two row kernels stride by `blockDim.x`. That is the one
// transformation `altup_aux.cuh` blesses: with the ahead-of-time launcher's
// `<<<rows, 256>>>` the stride is 256 and the loop is the loop it always was,
// so the ahead-of-time arithmetic is unchanged to the bit.
//
// # Why they are templates when the originals were not
//
// The originals wrote `__nv_bfloat16` because an ahead-of-time build has to
// choose its instantiations. Widen-compute-narrow is the same code at fp16,
// so the kernel is written over `T` and the fp16 dequantiser costs a row.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

#ifdef __CUDACC_RTC__
#include "pie_fp8.cuh"
#else
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace pie_cuda_driver::kernels::quant::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u8;
using ::pie_cuda_driver::kernels::device::usize;

/// One E4M3 byte, widened through fp16 exactly as the hardware does it.
///
/// `__half` and not a direct byte-to-float table: E4M3 is a strict subset of
/// fp16, so the vendor path is a bit shuffle with no rounding anywhere, and
/// the float that comes out is the only float the pattern denotes. A
/// hand-written expansion would have to reproduce the subnormal case, which
/// is the half of E4M3 that is easy to get wrong and impossible to notice.
__device__ __forceinline__ float fp8_e4m3_to_f32(u8 byte) {
    const __half h = __nv_cvt_fp8_to_halfraw(byte, __NV_E4M3);
    return __half2float(h);
}

/// `dst[i] = fp8(src[i]) * scale`, one scale for the whole tensor.
template <class T>
__global__ void dequant_fp8_e4m3(
    const u8* __restrict__ src, T* __restrict__ dst, float scale, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Elem<T>::from_f32(fp8_e4m3_to_f32(src[i]) * scale);
}

/// One scale per row, read once per block and reused down the row.
template <class T>
__global__ void dequant_fp8_e4m3_per_channel(
    const u8* __restrict__ src,
    T* __restrict__ dst,
    const float* __restrict__ scale_inv,
    i32 cols) {
    const i32 row = blockIdx.x;
    const float s = scale_inv[row];
    const usize off = static_cast<usize>(row) * cols;
    for (i32 j = threadIdx.x; j < cols; j += blockDim.x) {
        dst[off + j] = Elem<T>::from_f32(fp8_e4m3_to_f32(src[off + j]) * s);
    }
}

/// A scale per `row_block` x `col_block` tile, which is DeepSeek's layout.
///
/// `scale_cols` stays an operand rather than being recovered from `cols` and
/// `col_block` in the kernel: the ceiling division is the caller's statement
/// about the scale tensor's shape, and a kernel that recomputed it would be a
/// second place for the two to disagree.
///
/// The per-group launcher is this kernel with `row_block == col_block`, which
/// is why there are four kernels here and four entry points in the `.cu` --
/// but `per_group` is its own `__global__` rather than an alias, because a
/// row is a contract and the contract `quant::dequant_fp8_e4m3_to_bf16_per_group`
/// states takes `group_size`, not three tile extents two of which repeat it.
/// The BODY is shared, so there is still one implementation.
template <class T>
__device__ __forceinline__ void dequant_fp8_e4m3_tile(
    const u8* __restrict__ src,
    T* __restrict__ dst,
    const float* __restrict__ scales,
    i32 cols,
    i32 row_block,
    i32 col_block,
    i32 scale_cols) {
    const i32 row = blockIdx.x;
    const i32 scale_row = row / row_block;
    const usize off = static_cast<usize>(row) * cols;
    for (i32 j = threadIdx.x; j < cols; j += blockDim.x) {
        const i32 scale_col = j / col_block;
        const float s = scales[scale_row * scale_cols + scale_col];
        dst[off + j] = Elem<T>::from_f32(fp8_e4m3_to_f32(src[off + j]) * s);
    }
}

template <class T>
__global__ void dequant_fp8_e4m3_blocked(
    const u8* __restrict__ src,
    T* __restrict__ dst,
    const float* __restrict__ scales,
    i32 cols,
    i32 row_block,
    i32 col_block,
    i32 scale_cols) {
    dequant_fp8_e4m3_tile<T>(src, dst, scales, cols, row_block, col_block,
                             scale_cols);
}

/// Square tiles, so `scale_cols` is `ceil(cols / group_size)` and the kernel
/// recovers it rather than being told. That is the whole difference from
/// [`dequant_fp8_e4m3_blocked`], and it is what lets the row's operand list
/// be the ahead-of-time table's minus the stream and the rule-recovered row
/// count -- an operand a caller could get wrong is an operand this form does
/// not have.
template <class T>
__global__ void dequant_fp8_e4m3_per_group(
    const u8* __restrict__ src,
    T* __restrict__ dst,
    const float* __restrict__ scales,
    i32 cols,
    i32 group_size) {
    const i32 scale_cols = (cols + group_size - 1) / group_size;
    dequant_fp8_e4m3_tile<T>(src, dst, scales, cols, group_size, group_size,
                             scale_cols);
}

}  // namespace pie_cuda_driver::kernels::quant::device
