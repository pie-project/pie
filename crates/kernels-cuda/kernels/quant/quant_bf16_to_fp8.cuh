//===-- quant_bf16_to_fp8.cuh - the narrowing quantisers, as device text -===//
//
// Nine `__global__` templates where the ahead-of-time file had twelve
// `__global__`s. `quant_bf16_to_fp8.cu` is the fourteen host entry points
// that fire them; the rows in `kernels_cuda::families::quant` fire the
// templates at run time. There is exactly one definition of each, because two
// copies that agree today drift tomorrow and `norm/altup_aux` shipped a
// release proving it.
//
// `model-loader` calls `quantize_bf16_to_fp8_e4m3_per_channel` directly from
// Rust, through `tile.rs::CUDA_QUANTIZE_BF16_TO_FP8`. That is the ONE entry
// point it names; §43 measured the rest and deleted the eight the sentence
// above used to cover -- `_per_tensor` among them. The device text is
// untouched.
//
// # Twelve kernels became nine, and the four that merged are the point
//
// The ahead-of-time file held four PAIRS that differed in two constants and
// nothing else: `quant_per_channel` / `quant_int8_per_channel`,
// `cast_per_channel` / `cast_per_channel_int8`, `absmax_to_scale_inv` /
// `absmax_to_scale_inv_int8`. Each pair was a copy-paste with `448.f` swapped
// for `127.f` and `__nv_cvt_float_to_fp8` swapped for a `rintf` and a clamp
// -- which is to say each pair was one kernel the ahead-of-time build could
// not spell, because choosing the narrow format at compile time meant a
// second instantiation and a second translation unit's worth of `cicc`.
//
// [`fp8_e4m3`] and [`int8_sym`] are that choice as a template parameter. The
// saturation point and the narrowing are the format's, stated once each, and
// a third format -- E5M2, say -- is a struct and a row rather than three more
// copies of a reduction. The arithmetic per instantiation is the original's
// to the bit: `Fmt::max_abs()` inlines to the same literal the pair held.
//
// # What the launchers were doing, and where it went
//
//   * `quant_bf16_to_fp8`, `absmax_to_scale_inv`, `dequant_int8_per_channel`
//     are `(n + 255) / 256` blocks of 256 with an `n == 0` guard, which is
//     `LaunchRule::Elementwise` -- the guard is `Ungeometric::Empty`, which
//     the rule already returns.
//   * `absmax_per_row` and `quant_per_channel` are
//     `<<<rows, 256, (256 / 32) * sizeof(float)>>>`, which is
//     `LaunchRule::Rms` exactly: one block per row, 256 wide, 32 bytes of
//     shared memory for the eight per-warp partials. That equality is not a
//     coincidence -- it is the same row-wise reduction shape RMSNorm has, and
//     the rule was written for it.
//   * `cast_per_channel` is `<<<rows, 256, 0>>>`, which is
//     `LaunchRule::RouteRows`.
//
// Three have no rule and therefore no row:
//
//   * `absmax_bf16` launches `min(ceil(n / 256), 1024)` blocks and folds with
//     an `atomicMax`. The cap is the whole design -- it bounds atomic
//     contention -- and no rule states a capped grid. A rule that ignored the
//     cap would launch 100,000 blocks contending on one word.
//   * `quant_act_fp8_per_group` launches a 2-D grid `(n_groups, m)` with
//     128-wide blocks. No rule states a 2-D grid over a non-row axis.
//   * `w8a8_dequant` launches a 2-D block `(32, 8)`. No rule states a 2-D
//     block.
//
// They are here anyway: the recipe is that a `.cuh` holds the family's device
// text: a kernel left behind in the `.cu` is the second copy this split
// exists to prevent. Inventing a launch rule for them is what
// `new-horizon.md` §10 forbids outright.
//
// # Why the reductions still say 256 and the casts say `blockDim.x`
//
// `Rms` fixes the block at 256 and the shared-memory size at `256 / 32`
// floats, so a reduction under it can and must read [`kBlock`] -- the
// `tid < kBlock / 32` test that decides which lanes hold a valid partial is
// only correct for the width the shared array was sized at. `RouteRows`
// picks the block width from the row width, so a kernel under it strides by
// `blockDim.x`. That is the one transformation `altup_aux.cuh` blesses, and
// it is inert ahead of time: the launcher still passes 256.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

#ifdef __CUDACC_RTC__
#include "prelude/fp8.cuh"
#else
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace pie::quant {


/// The block width `LaunchRule::Rms` fires these reductions at.
///
/// Stated rather than read from `blockDim.x` because the shared array is
/// sized `kBlock / 32` by the launch and the final fold reads
/// `tid < kBlock / 32` lanes out of it. A kernel that took the width from
/// `blockDim.x` and the array size from the launch would be correct only
/// while the two agreed, and nothing would say so.
constexpr int kBlock = 256;

/// E4M3, as a compile-time choice of narrow format.
///
/// `__nv_cvt_float_to_fp8` and not a hand-written rounding rule: E4M3's
/// round-to-nearest-even with saturation is the format's, the shims
/// reproduce it bit-identically over all 256 byte patterns (`new-horizon.md`
/// §15.2), and a second implementation of it in this tree is exactly the
/// drift the shims exist to prevent.
struct fp8_e4m3 {
    using store = u8;
    /// OCP MX's largest finite E4M3 magnitude.
    static __host__ __device__ __forceinline__ float max_abs() { return 448.f; }
    static __device__ __forceinline__ store narrow(float v) {
        return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
};

/// Symmetric INT8, as the same kind of choice.
///
/// `rintf` is round-to-nearest-even, which is the sane default and the one
/// the host-side reference quantiser uses. The clamp is asymmetric because
/// int8 is -- [-128, 127] -- even though the SCALE is symmetric, so a row
/// whose minimum is exactly `-absmax` lands on -127 and not -128.
struct int8_sym {
    using store = i8;
    static __host__ __device__ __forceinline__ float max_abs() { return 127.f; }
    static __device__ __forceinline__ store narrow(float v) {
        int q = static_cast<int>(rintf(v));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        return static_cast<store>(q);
    }
};

/// The per-row absmax fold, shared by every reduction below.
///
/// `smem` must hold `kBlock / 32` floats, which is what `LaunchRule::Rms`
/// requests. The fold ORDER is the original's -- warp shuffles down, then
/// warp 0 over the partials -- because a max over a row containing a NaN
/// depends on it, and `driver-pipeline`'s tolerance contract holds nothing
/// about which NaN wins.
__device__ __forceinline__ float row_absmax(float local, float* smem, int tid) {
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    const int lane = tid & 31;
    const int warp = tid / 32;
    if (lane == 0) smem[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = (tid < kBlock / 32) ? smem[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) smem[0] = local;
    }
    __syncthreads();
    return smem[0];
}

/// `out[i] = Fmt(W[i] * scale_inv)`, one scale for the whole tensor.
template <class Fmt>
__global__ void quant_flat(
    const bf16* __restrict__ W,
    typename Fmt::store* __restrict__ out,
    float scale_inv,
    usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = Fmt::narrow(bf16_to_f32(W[i]) * scale_inv);
}

/// Per-row absmax in place becomes `weight_scale_inv = absmax / Fmt::max`.
///
/// A degenerate row -- all zeros -- gets 1 rather than 0, so the dequantiser
/// multiplies by one and the GEMM dispatcher never divides by zero.
template <class Fmt>
__global__ void absmax_to_scale_inv(float* x, i32 n) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = (v > 0.f) ? (v / Fmt::max_abs()) : 1.f;
}

/// Stage 1 of the tensor-parallel path: one absmax per row, no narrowing.
///
/// Split from the cast so the host can all-reduce the per-row absmax across
/// ranks before the scales are decided -- a rank that picked its own scale
/// would produce a shard the others cannot be concatenated with.
template <class T>
__global__ void absmax_per_row(
    const T* __restrict__ W, float* __restrict__ absmax_out, i32 cols) {
    const int tid = threadIdx.x;
    extern __shared__ float warp_max[];

    const usize row_off = static_cast<usize>(blockIdx.x) * cols;
    float local = 0.f;
    for (i32 j = tid; j < cols; j += kBlock) {
        const float v = fabsf(Elem<T>::to_f32(W[row_off + j]));
        if (v > local) local = v;
    }
    const float row_max = row_absmax(local, warp_max, tid);
    if (tid == 0) absmax_out[blockIdx.x] = row_max;
}

/// Stage 2: narrow a row with a scale someone else already decided.
template <class Fmt>
__global__ void cast_per_channel(
    const bf16* __restrict__ W,
    typename Fmt::store* __restrict__ out,
    const float* __restrict__ scale_inv,
    i32 cols) {
    const float s = scale_inv[blockIdx.x];
    const float s_recip = (s > 0.f) ? (1.f / s) : 0.f;
    const usize row_off = static_cast<usize>(blockIdx.x) * cols;
    for (i32 j = threadIdx.x; j < cols; j += blockDim.x) {
        out[row_off + j] = Fmt::narrow(bf16_to_f32(W[row_off + j]) * s_recip);
    }
}

/// Both stages in one block: absmax the row, then narrow it.
///
/// Emits BOTH the narrow row and `weight_scale_inv = absmax / Fmt::max` --
/// the MULTIPLICATIVE factor the GEMM dispatcher hands cuBLASLt -- so the
/// dispatcher never computes a reciprocal at fire time.
template <class Fmt>
__global__ void quant_per_channel(
    const bf16* __restrict__ W,
    typename Fmt::store* __restrict__ out,
    float* __restrict__ scale_inv,
    i32 cols) {
    const int tid = threadIdx.x;
    extern __shared__ float warp_max[];

    const usize row_off = static_cast<usize>(blockIdx.x) * cols;
    float local = 0.f;
    for (i32 j = tid; j < cols; j += kBlock) {
        const float v = fabsf(bf16_to_f32(W[row_off + j]));
        if (v > local) local = v;
    }
    const float row_max = row_absmax(local, warp_max, tid);

    // Degenerate row: scale_inv = 1 and an all-zero narrow row.
    const float quant = (row_max > 0.f) ? (Fmt::max_abs() / row_max) : 1.f;
    const float weight_scale_inv = (row_max > 0.f) ? (row_max / Fmt::max_abs()) : 1.f;
    if (tid == 0) scale_inv[blockIdx.x] = weight_scale_inv;

    for (i32 j = tid; j < cols; j += kBlock) {
        out[row_off + j] = Fmt::narrow(bf16_to_f32(W[row_off + j]) * quant);
    }
}

/// INT8 back to `T`, flat, with the row recovered from the linear index.
///
/// Templated on the WIDE side rather than the narrow one: there was no fp8
/// twin of this kernel to unify with. `quant/dequant_fp8.cuh` held the fp8
/// dequantisers, they were row-shaped rather than flat, and the file is gone
/// -- its only Rust namers were the seven launchers `gemm/quant.rs` alone
/// read.
template <class T>
__global__ void dequant_int8_per_channel(
    const i8* __restrict__ W,
    T* __restrict__ out,
    const float* __restrict__ scale_inv,
    i32 cols,
    usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const i32 row = static_cast<i32>(i / static_cast<usize>(cols));
    out[i] = Elem<T>::from_f32(static_cast<float>(W[i]) * scale_inv[row]);
}

// ---------------------------------------------------------------------------
// The three with no rule, and therefore no row.
// ---------------------------------------------------------------------------

/// Whole-tensor absmax into one device scalar.
///
/// The grid is capped by the launcher and the fold ends in a single
/// `atomicMax` per block, so the atomic traffic is bounded by the cap rather
/// than by `n`. `atomicMax` on the bit pattern works because these are
/// magnitudes: a non-negative float compares the same as its bits do as a
/// signed integer.
__global__ void absmax_bf16(
    const bf16* __restrict__ W, float* __restrict__ out, usize n) {
    __shared__ float warp_max[kBlock / 32];
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid / 32;
    const unsigned lane = tid & 31;
    usize i = static_cast<usize>(blockIdx.x) * kBlock + tid;
    const usize stride = static_cast<usize>(gridDim.x) * kBlock;

    float local = 0.f;
    for (; i < n; i += stride) {
        const float v = fabsf(bf16_to_f32(W[i]));
        if (v > local) local = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = (tid < kBlock / 32) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) atomicMax(reinterpret_cast<int*>(out), __float_as_int(local));
    }
}

/// DeepSeek's blockwise W8A8 activation quantiser: one block per
/// `(row, k-group)`.
///
/// `gs` is 128 in every real deployment, so a 128-thread block gives each
/// thread exactly one element and the reduction is four warp shuffles.
/// Larger groups stride.
///
/// The scale is MULTIPLICATIVE -- `value = fp8 * scale` -- which is
/// cuBLASLt's contract and the opposite of the per-channel weight path's
/// `scale_inv`. Both names are the caller's; the arithmetic below is what
/// decides.
__global__ void quant_act_fp8_per_group(
    const bf16* __restrict__ act,
    u8* __restrict__ out,
    float* __restrict__ scale_out,
    i32 m,
    i32 k,
    i32 gs,
    i32 n_groups) {
    const i32 row = blockIdx.y;
    const i32 g = blockIdx.x;
    if (row >= m || g >= n_groups) return;

    const i32 base = g * gs;
    const i32 remaining = k - base;
    const i32 count = (gs < remaining) ? gs : remaining;
    const usize off = static_cast<usize>(row) * k + base;

    float amax = 0.f;
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(bf16_to_f32(act[off + i])));
    }
    __shared__ float warp_max[128 / 32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    }
    if (lane == 0) warp_max[warp] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = warp_max[0];
        for (unsigned w = 1; w < blockDim.x / 32; ++w) v = fmaxf(v, warp_max[w]);
        warp_max[0] = v;
    }
    __syncthreads();
    amax = warp_max[0];

    const float scale = (amax > 0.f) ? (amax / fp8_e4m3::max_abs()) : 1.f;
    const float scale_rcp = (amax > 0.f) ? (fp8_e4m3::max_abs() / amax) : 0.f;
    if (threadIdx.x == 0) {
        scale_out[static_cast<usize>(row) * n_groups + g] = scale;
    }
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        out[off + i] = fp8_e4m3::narrow(bf16_to_f32(act[off + i]) * scale_rcp);
    }
}

/// W8A8 post-GEMM dequant: `bf16[m, n] = int32[m, n] * act_inv[m] * w_inv[n]`.
///
/// One thread per output element, and deliberately not fused with the GEMM:
/// cuBLAS writes the int32 accumulator and this scales it row by column
/// afterwards, which is bandwidth-bound either way.
__global__ void w8a8_dequant(
    const i32* __restrict__ acc,
    const float* __restrict__ act_inv,
    const float* __restrict__ w_inv,
    bf16* __restrict__ out,
    i32 M,
    i32 N) {
    const i32 n = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;
    const float v = static_cast<float>(acc[m * N + n]) * act_inv[m] * w_inv[n];
    out[m * N + n] = f32_to_bf16(v);
}

}  // namespace pie::quant
