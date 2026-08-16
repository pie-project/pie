//===-- dtype_cast.cuh - the loader's conversions, as device text --------===//
//
// Eleven `__global__`s -- six of them now templates -- and nothing else.
// `dtype_cast.cu` is the eleven host launchers that fire them ahead of time;
// the rows in `kernels_cuda::families::quant` fire the templates at run
// time. There is exactly one definition of each, because two copies that
// agree today drift tomorrow and `norm/altup_aux` shipped a release proving
// it.
//
// This is the file `model-loader` reaches into most directly:
// `pie_k_quant_cast_fp32_to_bf16` and `scale_rows_bf16` are called by name
// from Rust while the checkpoint path is still ahead-of-time. Nothing here
// deletes an entry point.
//
// # What the launchers were doing, and where it went
//
// Seven of them were the same four lines:
//
//     if (n == 0) return;
//     const auto blocks = (n + BLOCK - 1) / BLOCK;
//     kernel<<<blocks, BLOCK, 0, stream>>>(...);
//
// which is `LaunchRule::Elementwise`, with the `n == 0` guard as
// `Ungeometric::Empty` -- the rule already returns it and the binder already
// refuses on it.
//
// The other four have no rule and therefore no row:
//
//   * `marlin_permute_scales_per_group` is `<<<total64, 64>>>` -- a 64-wide
//     block, because the permutation it applies is an 8x8 transpose and 64
//     threads is the tile. No rule states a 64-wide block, and `RouteRows`
//     would hand it `ceil_warp(width)`, which is not the same number.
//   * `awq_dequant_to_bf16` and `gptq_dequant_to_bf16` launch a 2-D block
//     `(32, 8)` over a 2-D grid, one thread per output element of a
//     transposing dequantiser. No rule states a 2-D block.
//
// They are here anyway, because the recipe is that a `.cuh` holds the
// family's device text and a `.cu` holds launchers -- a kernel left behind in
// the `.cu` is the second copy this split exists to prevent. Inventing a
// launch rule to give them rows is the thing `new-horizon.md` §10 forbids
// outright, so they compile under NVRTC and wait for one.
//
// # Why six of them are templates
//
// The originals were `_bf16`, `_fp32` and `_fp16` as separate `__global__`s
// because an ahead-of-time build has to choose its instantiations and each
// one costs a translation unit's worth of `cicc`. Widen-compute-narrow is the
// same code at every width, so `scale` is one template with three rows where
// it was three kernels, and the fp16 casts the loader never had cost a line.
//
// [`Cast`] is why that works at `float` too. `Elem` is specialised on
// `bf16` and `f16` only -- deliberately, since those are the two formats the
// prelude's wrapper structs make distinguishable -- and `float` needs no
// conversion at all. Extending `Elem` itself would mean reopening the
// prelude's namespace from a family header, which is a second place the
// prelude is defined; a family-local trait that DELEGATES to `Elem` is not.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::quant {


/// `float` under a name a row can spell.
///
/// `DeviceKernel::elem` is a path under `::pie`, so the
/// keyword `float` cannot appear in one. The alias costs nothing and keeps
/// the row's element type looking like every other row's.
using f32 = float;

/// `Elem`, extended to `f32`.
///
/// The generic case delegates rather than reimplements: a row at `bf16` goes
/// through exactly the prelude conversion every other family's rows go
/// through, so there is no second rounding rule in this tree. Only the
/// identity case is stated here, and it is stated because it is the one the
/// prelude cannot hold -- `Elem<float>` in `pie_device.cuh` would be a
/// specialisation for a type the prelude has no opinion about.
template <class T>
struct Cast {
    static __device__ __forceinline__ float to_f32(T v) { return Elem<T>::to_f32(v); }
    static __device__ __forceinline__ T from_f32(float v) { return Elem<T>::from_f32(v); }
};

template <>
struct Cast<f32> {
    static __device__ __forceinline__ float to_f32(f32 v) { return v; }
    static __device__ __forceinline__ f32 from_f32(float v) { return v; }
};

/// fp32 in, `T` out.
template <class T>
__global__ void cast_f32_to(const float* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(src[i]);
}

/// `T` in, fp32 out.
template <class T>
__global__ void cast_to_f32(const T* __restrict__ src, float* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::to_f32(src[i]);
}

/// fp16 in, `T` out -- through fp32, which is exact in both directions.
template <class T>
__global__ void cast_f16_to(const f16* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(Cast<f16>::to_f32(src[i]));
}

/// E8M0 in, `T` out.
///
/// E8M0 stores an exponent and nothing else: byte `b` denotes `2^(b - 127)`,
/// with `0xFF` reserved for NaN. That is the fp32 exponent field verbatim, so
/// the decode is a shift rather than any arithmetic -- `b << 23` *is* the
/// answer, and `exp2f` would be a slower way to write it.
template <class T>
__global__ void cast_e8m0_to(const u8* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u32 bits = static_cast<u32>(src[i]);
    const float v = bits == 0xFFu ? __int_as_float(0x7FFFFFFF) : __int_as_float(bits << 23);
    dst[i] = Cast<T>::from_f32(v);
}

/// One multiply per element, in fp32 whatever the storage dtype.
///
/// The narrow dtypes round once, on the store -- accumulating in bf16 would
/// round the operand as well, and the loader's host executor (which
/// multiplies in fp32 and is compared against this) would disagree. At
/// `T = f32` both conversions are the identity, so this is the fp32 kernel's
/// single multiply and not a round trip through anything.
template <class T>
__global__ void scale(
    const T* __restrict__ src, T* __restrict__ dst, usize n, float factor) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(Cast<T>::to_f32(src[i]) * factor);
}

// ---------------------------------------------------------------------------
// The four with no rule, and therefore no row.
// ---------------------------------------------------------------------------

/// Marlin's per-group scale permutation: an 8x8 transpose of each 64 scalars.
///
/// One block per row of 64, 64 threads. Shared memory rather than a shuffle
/// because the permutation crosses the warp boundary at `tid == 32` and a
/// `__shfl` cannot.
__global__ void marlin_permute_scales_per_group(bf16* __restrict__ s, int total64_rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= total64_rows || tid >= 64) return;
    bf16* base = s + static_cast<usize>(row) * 64;
    __shared__ bf16 buf[64];
    buf[tid] = base[tid];
    __syncthreads();
    const int i = tid / 8;
    const int j = tid % 8;
    const int src_idx = j * 8 + i;
    base[tid] = buf[src_idx];
}

/// AWQ dequant straight to bf16, bypassing Marlin.
///
/// One thread per `(n, k)` output element; writes the `[N, K]` transposed
/// layout HF `Linear` weights use.
///
///   bf16[n, k] = (w[k, n] - zp[g(k), n]) * scales[g(k), n]
///
/// where `w[k, n] = (qweight[k, n/8] >> (4 * REV[n%8])) & 0xF`, likewise for
/// the zero points, and `REV = [0, 4, 1, 5, 2, 6, 3, 7]` is AWQ's "gemm"
/// reverse-pack order.
__global__ void awq_dequant_to_bf16(
    const u32* __restrict__ qweight,
    const u32* __restrict__ qzeros,
    const bf16* __restrict__ scales,
    bf16* __restrict__ out,
    int size_k,
    int size_n,
    int group_size) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= size_n || k >= size_k) return;

    constexpr int REV[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const int n8 = size_n / 8;
    const int n_packed = n / 8;
    const int n_in_8 = n % 8;
    const int shift = 4 * REV[n_in_8];

    const int g = k / group_size;
    const u32 w_word = qweight[k * n8 + n_packed];
    const u32 zp_word = qzeros[g * n8 + n_packed];
    const int w_int4 = static_cast<int>((w_word >> shift) & 0xFu);
    const int zp_int4 = static_cast<int>((zp_word >> shift) & 0xFu);

    const float sc = bf16_to_f32(scales[g * size_n + n]);
    const float val = static_cast<float>(w_int4 - zp_int4) * sc;
    out[n * size_k + k] = f32_to_bf16(val);
}

/// GPTQ dequant: `qweight` packed along K with no interleave, `qzeros` packed
/// along N with no interleave, optional `g_idx` for `desc_act = true`.
///
///   bf16[n, k] = (w[k, n] - (zp[g(k), n] + 1)) * scales[g(k), n]
///
/// The `+1` on the zero point is autogptq's storage convention -- `qzeros`
/// holds `zp - 1` -- so the dequantiser must add it back. For symmetric GPTQ
/// (`kU4B8` in Marlin) `qzeros` is filled with 7, `+1` gives 8, and
/// `nibble - 8` is the signed [-8, 7] range the scales apply on top of.
__global__ void gptq_dequant_to_bf16(
    const u32* __restrict__ qweight,
    const u32* __restrict__ qzeros,
    const bf16* __restrict__ scales,
    const i32* __restrict__ g_idx,
    bf16* __restrict__ out,
    int size_k,
    int size_n,
    int group_size) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= size_n || k >= size_k) return;

    const int n8 = size_n / 8;
    const int g = (g_idx != nullptr) ? g_idx[k] : (k / group_size);

    const u32 w_word = qweight[(k / 8) * size_n + n];
    const u32 z_word = qzeros[g * n8 + (n / 8)];
    const int w_int4 = static_cast<int>((w_word >> ((k % 8) * 4)) & 0xFu);
    const int zp_int4 = static_cast<int>((z_word >> ((n % 8) * 4)) & 0xFu) + 1;

    const float sc = bf16_to_f32(scales[g * size_n + n]);
    const float val = static_cast<float>(w_int4 - zp_int4) * sc;
    out[n * size_k + k] = f32_to_bf16(val);
}

/// `buf[r, c] *= l[c]`, in place: one block per row, columns strided by
/// `blockDim.x`.
///
/// Was a 2-D grid over a `(128, 2)` block, which no ported rule states. The
/// stride loop is the same transformation `dequant_fp8_e4m3_per_channel`
/// took -- a block width the kernel never names is a block width
/// `LaunchRule::RouteRows` may choose, and `ceil_warp(width)` capped at 1024
/// is what it chooses. The arithmetic per element is byte-identical; only
/// which thread does it moved.
template <class T>
__global__ void scale_rows(T* buf, const T* l, int width) {
    const int row = blockIdx.x;
    T* row_buf = buf + static_cast<usize>(row) * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        row_buf[c] =
            Cast<T>::from_f32(Cast<T>::to_f32(row_buf[c]) * Cast<T>::to_f32(l[c]));
    }
}

}  // namespace pie::quant
