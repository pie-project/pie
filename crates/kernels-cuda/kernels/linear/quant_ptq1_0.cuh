#pragma once

#include "prelude/device.cuh"


namespace pie::linear {

// PTQ1_0 ternary decode-in-dot (Dtype::Ptq1_0, spelled `g128_t3_f16_n`).
//
// A block is 28 bytes for 128 weights (1.75 bpw): `qs[24]` (5 base-3 trits per
// byte) + `qh[2]` (4 trits per byte) + a trailing fp16 scale `d`. A weight is
// `value = (trit - 1) * d`, trit in {0,1,2} -> {-1,0,+1}. The scale is INLINE
// per block; there is NO separate scales plane. row_bytes = ceil(K/128) * 28.
//
// This is a faithful port of `quant_ptq1_0.metal`, which in turn mirrors the
// host oracle `checkpoint::codec::ptq1_0::decode_block` byte for byte. The
// subtle part is the *staging*: for a 24-byte `qs` the fork walks chunks
// {16,8} (the 32-chunk never fits), emitting elements in NATURAL order 0..127;
// the `qh` tail carries elements 120..127. We decode a block the same way,
// inside the dot, so `y[row] = sum_i decoded_W[row][i] * x[i]`.
//
// Each decoded weight is `float(trit - 1) * d` multiplied by the activation,
// exactly as the host forms `f32::from(trit-1) * d`. With a one-hot activation
// only one term survives and the read-back is bit-exact against the host
// decoder (an integer in {-1,0,1} times a finite scale is exact, and
// half->float is exact for every finite scale).

constexpr int kPtq10Block = 128;

constexpr int kPtq10BlockBytes = 28;

constexpr int kPtq10QsLen = 24;

// One base-3 trit from a packed byte at place `n` (0..4): shift the trit into
// the byte MSBs as `q = (byte * 3^n) & 0xff`, then `(q * 3) >> 8` in {0,1,2}.
// Matches the host's `trit()` arithmetic exactly.
__device__ __forceinline__ int ptq1_0_trit(u8 byte, int n) {
    const int pow3[5] = {1, 3, 9, 27, 81};
    const int q = (static_cast<int>(byte) * pow3[n]) & 0xff;
    return (q * 3) >> 8;
}

// The inline fp16 scale lives in the block's last two bytes, little-endian.
__device__ __forceinline__ float ptq1_0_scale(const u8* __restrict__ blk) {
    const u32 bits =
        static_cast<u32>(blk[26]) | (static_cast<u32>(blk[27]) << 8);
    return f16_to_f32(f16{static_cast<u16>(bits)});
}

// Dot one 28-byte block against the 128 activations at `xblk`, decoding weights
// in the fork's {16,8}+qh staging (natural element order). Returns
// `sum_i (trit_i - 1) * d * x_i`.
template <class Ti>
__device__ __forceinline__ float ptq1_0_block_dot(
    const u8* __restrict__ blk, const Ti* __restrict__ xblk) {
    const u8* qs = blk;
    const u8* qh = blk + kPtq10QsLen;
    const float d = ptq1_0_scale(blk);

    float acc = 0.f;
    int idx = 0;

    // qs stages: a 16-byte chunk (elements 0..79) then an 8-byte chunk
    // (80..119); trit-slot `n` outer, byte `m` inner emits natural order.
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int base = (chunk == 0) ? 0 : 16;
        const int c = (chunk == 0) ? 16 : 8;
        for (int n = 0; n < 5; ++n) {
            for (int m = 0; m < c; ++m) {
                const int t = ptq1_0_trit(qs[base + m], n);
                const float wv = static_cast<float>(t - 1) * d;
                acc += wv * Elem<Ti>::to_f32(xblk[idx]);
                ++idx;
            }
        }
    }

    // qh tail: elements 120..127, 4 trits per byte.
    for (int n = 0; n < 4; ++n) {
        for (int m = 0; m < 2; ++m) {
            const int t = ptq1_0_trit(qh[m], n);
            const float wv = static_cast<float>(t - 1) * d;
            acc += wv * Elem<Ti>::to_f32(xblk[idx]);
            ++idx;
        }
    }

    return acc;
}

// Matrix-vector (and small-batch) decode-in-dot. Shape mirrors the Metal
// `ptq1_0_qmv`: a thread block is 2 warps (blockDim (32, 2, 1)); each warp is a
// simdgroup computing 4 output rows, the 32 lanes split the K blocks (stride
// 32), and a warp-shuffle sum reduces each row before lane 0 writes. Grid is
// `qmv_grid`: [vecs, ceil(out_vec_size / (2 * 4)), 1].
template <class Ti, class To>
__global__ void ptq1_0_qmv(
    const u8* __restrict__ w,
    const Ti* __restrict__ x,
    To* __restrict__ y,
    int in_vec_size,
    int out_vec_size,
    const u32* __restrict__ win) {
    constexpr int kSimdgroups = 2;
    constexpr int kResultsPerSimdgroup = 4;

    const int vec = blockIdx.x;
    if (win != nullptr && vec >= static_cast<int>(win[0])) return;
    const int simd_gid = threadIdx.y;
    const int simd_lid = threadIdx.x;

    const int num_blocks = in_vec_size / kPtq10Block;
    const long long row_bytes =
        static_cast<long long>(num_blocks) * kPtq10BlockBytes;
    const int out_row = static_cast<int>(blockIdx.y) *
            (kSimdgroups * kResultsPerSimdgroup) +
        simd_gid * kResultsPerSimdgroup;

    const Ti* x_vec = x + static_cast<long long>(vec) * in_vec_size;

    float result[kResultsPerSimdgroup];
#pragma unroll
    for (int r = 0; r < kResultsPerSimdgroup; ++r) result[r] = 0.f;

    for (int bk = simd_lid; bk < num_blocks; bk += 32) {
        const Ti* xblk = x_vec + static_cast<long long>(bk) * kPtq10Block;
#pragma unroll
        for (int row = 0; row < kResultsPerSimdgroup; ++row) {
            const int r = out_row + row;
            if (r >= out_vec_size) continue;
            const u8* blk = w + static_cast<long long>(r) * row_bytes +
                static_cast<long long>(bk) * kPtq10BlockBytes;
            result[row] += ptq1_0_block_dot<Ti>(blk, xblk);
        }
    }

    To* y_vec = y + static_cast<long long>(vec) * out_vec_size + out_row;
#pragma unroll
    for (int row = 0; row < kResultsPerSimdgroup; ++row) {
        float v = result[row];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (simd_lid == 0 && out_row + row < out_vec_size) {
            y_vec[row] = Elem<To>::from_f32(v);
        }
    }
}

}
