#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

// **THE DECODE, WRITTEN OUT** — OCP E4M3, one byte: sign, four exponent bits
// bias 7, three mantissa bits. The host's `checkpoint::codec::fp8` states the
// same numbers in Rust and the two are one claim made twice.
//
// Deliberately arithmetic rather than `__nv_cvt_fp8_to_halfraw`: the NVRTC
// prelude carries `cuda_fp8.h` only where the toolchain ships it as a header,
// and a weight decode that compiles on one card and not the next is a worse
// bargain than four instructions. E4M3 has NO infinity — the all-ones
// exponent carries ordinary values up to 448, and `S.1111.111` alone is NaN.
__device__ __forceinline__ float e4m3_to_f32(u8 byte) {
    const int exp = (byte >> 3) & 0xF;
    const int mant = byte & 0x7;
    float mag;
    if (exp == 0) {
        // Subnormal: units of 2^-9, which is what makes this branch differ
        // from the normal one by more than the implicit bit.
        mag = static_cast<float>(mant) * 0.001953125f;
    } else if (exp == 0xF && mant == 0x7) {
        mag = __int_as_float(0x7fc00000);
    } else {
        // `2^(exp - 7)` composed straight into the exponent field: exact, and
        // `exp - 7` spans `[-6, 8]`, so the assembled float is always finite.
        mag = (1.f + static_cast<float>(mant) * 0.125f)
            * __int_as_float((exp - 7 + 127) << 23);
    }
    return (byte & 0x80) ? -mag : mag;
}

// **THE PER-OUTPUT-ROW FP8 PROJECTION** (`gr_e4m3_f32_n`, QNF §J2 priority 2):
// `linear.matmul` and `linear.lm_head` over a weight the store seats as e4m3
// bytes with ONE f32 scale per output row. `y[t, r] = s[r] · Σ_k w[r,k]·x[t,k]`.
//
// This is the lane cuBLAS cannot serve: its fp8 arms want e4m3 on BOTH
// operands, and the serving contract here is weight-only against a bf16/f16
// activation. So the decode happens inside the dot, one byte at a time, and
// the scale — being constant over the whole contraction — lands ONCE, after
// the warp reduce, rather than per group. That is not an optimisation but the
// exactness argument: a factor pulled out of the sum is applied to the sum.
//
// The geometry is `matmul_affine`'s, for the same reason: one block column
// per ACTIVATION ROW, `kRowsT` weight rows per warp, a lane per contraction
// step striding by thirty-two so consecutive lanes read consecutive code
// bytes. A decode step's row count is small and the weight is read once per
// row; the tiled point that amortises a long prefill arrives with a caller
// that measures it.
template <class T, int kRowsT>
__global__ void matmul_fp8_row(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    const int token = blockIdx.x;
    // The staged-geometry seat (`quant.cuh`'s idiom): a replay whose grid was
    // carved at a bucket retires its padded rows here, off a word the fire
    // staged, not a parameter the recording baked.
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const float* __restrict__ sf = reinterpret_cast<const float*>(scales);
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int j = lane_id; j < k; j += 32) {
        const float xv = Elem<T>::to_f32(x[j]);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const u8 code =
                codes[static_cast<long long>(row_of[r]) * k + j];
            acc[r] = fmaf(e4m3_to_f32(code), xv, acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r] * sf[row]);
        }
    }
}

// **THE 128x128 BLOCK-SCALED FP8 PROJECTION** (`g128x128_e4m3_f32_n`, the
// DeepSeek-class stored form): the same e4m3 codes, but the scale plane is a
// `[ceil(n/128), ceil(k/128)]` f32 rectangle, row-major — one factor per
// 128-row band per 128-wide contraction tile.
//
// A tile scale does NOT factor out of the whole dot, so the fold is per
// k-tile: accumulate the tile's partial UNSCALED, multiply by that tile's
// factor, and add. That ordering is the contract — a kernel that scaled each
// term as it landed would be arithmetically the same claim and numerically a
// different one, and the goldens hold this order.
//
// The band is read per ROW rather than per warp tile, because `row_of` clamps
// a trailing warp's rows onto `n - 1` and a clamped row can sit in a band its
// tile does not own; the clamped lanes' answers are discarded, but they must
// not read out of the scale plane to produce them.
template <class T, int kRowsT>
__global__ void matmul_fp8_tile(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    constexpr int kTile = 128;
    const int token = blockIdx.x;
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int ktiles = (k + kTile - 1) / kTile;
    const float* __restrict__ sf = reinterpret_cast<const float*>(scales);
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
    const float* band_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        row_of[r] = min(row0 + r, n - 1);
        band_of[r] = sf + static_cast<long long>(row_of[r] / kTile) * ktiles;
    }

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int kt = 0; kt < ktiles; ++kt) {
        const int base = kt * kTile;
        const int lim = min(kTile, k - base);

        float part[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) part[r] = 0.f;

        for (int j = lane_id; j < lim; j += 32) {
            const float xv = Elem<T>::to_f32(x[base + j]);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const u8 code =
                    codes[static_cast<long long>(row_of[r]) * k + base + j];
                part[r] = fmaf(e4m3_to_f32(code), xv, part[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] = fmaf(part[r], band_of[r][kt], acc[r]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

}  // namespace pie::linear
