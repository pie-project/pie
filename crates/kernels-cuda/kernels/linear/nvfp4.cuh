#pragma once

#include "prelude/device.cuh"
// The e4m3 decode is `linear/fp8.cuh`'s, not a second copy of it: NVFP4's
// GROUP scale is an ordinary OCP e4m3 byte, and a weight format whose scale
// decoded differently from the fp8 point's would be two claims about one
// spec. Spelled root-relative, the way every sibling include in this tree is.
#include "linear/fp8.cuh"

namespace pie::linear {

// **THE E2M1 CODE, WRITTEN OUT** — one nibble: bit 3 sign, bits 2-1 exponent
// bias 1, bit 0 mantissa. The exponent field zero is the subnormal branch, so
// `0x1` is 0.5 and not 1.0, and the eight magnitudes the scheme can spell are
// exactly `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` — small enough that the arithmetic
// `fp8.cuh` writes out for e4m3 would be longer than the table it computes.
// So: a table, sixteen floats, indexed by the nibble itself.
//
// The negative half is the positive half with the sign bit set, `-0.f`
// included: a code plane may carry `0x8` and the fold must read it as a zero
// that does not change a sum's sign.
//
// The twin of `quant.cuh`'s `kFp4Lut`, which serves MXFP4's identical code
// alphabet under a different scale scheme. Two names for one table because
// the two units compile apart and a shared `__constant__` would have to move
// to the prelude to be shared at all.
__device__ __constant__ float kNvfp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

// **THE NVFP4 WEIGHT-ONLY PROJECTION** (`g16_e2m1_gt_e4m3_f32_n_n`, QNF §J2 —
// the row NVIDIA is now shipping official checkpoints in): `linear.matmul`
// and `linear.lm_head` over a weight stored as e2m1 nibbles in groups of
// sixteen, one e4m3 scale per group, and ONE f32 for the whole tensor.
//
//   `y[t, r] = S · Σ_g ( s[r,g] · Σ_{j∈g} lut(w[r,j])·x[t,j] )`
//
// The depth-two scale is the whole point of the format and it is why this is
// not `matmul_fp8_row` with a different table. `s[r,g]` does NOT factor out
// of the contraction, so it lands on the GROUP's partial — the same ordering
// `matmul_fp8_tile` keeps for its 128-tiles, and the goldens hold it. `S`
// does factor out of everything, so it lands ONCE, after the warp reduce,
// for the exactness argument `matmul_fp8_row` states: a factor pulled out of
// the sum is applied to the sum. It arrives as a kernel ARGUMENT rather than
// as a third plane because it is one number and a load per block to fetch it
// would buy nothing.
//
// **NATIVE MMA IS SM120-ONLY.** `mma.sync...e2m1` decodes these codes in
// hardware on Blackwell and nowhere else; the cards this engine mostly runs
// on cannot spell the instruction. So the decode happens inside the dot, on
// any card with a warp — the same bargain `fp8.cuh` names, for the same
// reason.
//
// **THE PACKING**: a row is `k/2` bytes, and byte `i` carries element `2i` in
// its LOW nibble and `2i + 1` in its high one — `quant.cuh`'s convention,
// where the packer writes `(hi << 4) | (lo & 0xF)` with `lo` the earlier
// element. A group is sixteen codes, so it is eight bytes, so it is two
// aligned `u32` words; `k % 16 == 0` makes `k/2` a multiple of eight and
// every group-word pair aligned. Little-endian byte `b` of a word is element
// pair `4q + b` of the group's two words, which is what the shift schedule
// below spells.
//
// The geometry is `matmul_fp8_row`'s, unchanged: one block column per
// ACTIVATION ROW, `kRowsT` weight rows per warp, a lane per GROUP striding by
// thirty-two, the row clamp, the shuffle reduce, and the staged-rows seat.
template <class T, int kRowsT>
__global__ void matmul_nvfp4(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    T* __restrict__ out,
    float tensor_scale,
    int n,
    int k,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    constexpr int kGroup = 16;
    const int token = blockIdx.x;
    // The staged-geometry seat (`quant.cuh`'s idiom): a replay whose grid was
    // carved at a bucket retires its padded rows here, off a word the fire
    // staged, not a parameter the recording baked.
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int groups_per_row = k / kGroup;
    // Two `u32` per group, so the word index is the group index doubled.
    const int words_per_row = k / 8;
    const unsigned* __restrict__ w32 =
        reinterpret_cast<const unsigned*>(codes);
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < groups_per_row; g += 32) {
        float xv[kGroup];
#pragma unroll
        for (int j = 0; j < kGroup; ++j)
            xv[j] = Elem<T>::to_f32(x[g * kGroup + j]);

#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            float part = 0.f;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const unsigned word =
                    w32[static_cast<long long>(row_of[r]) * words_per_row
                        + g * 2 + q];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned byte = (word >> (8 * b)) & 0xFFu;
                    const int at = q * 8 + b * 2;
                    part = fmaf(kNvfp4Lut[byte & 0xFu], xv[at], part);
                    part = fmaf(kNvfp4Lut[byte >> 4], xv[at + 1], part);
                }
            }
            // The group's own e4m3 factor, on the group's partial — it does
            // not factor out of the contraction and must not be hoisted.
            const u8 sb = scales[static_cast<long long>(row_of[r])
                                 * groups_per_row + g];
            acc[r] = fmaf(part, e4m3_to_f32(sb), acc[r]);
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
                    Elem<T>::from_f32(acc[r] * tensor_scale);
        }
    }
}

}  // namespace pie::linear
