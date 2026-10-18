#pragma once

// The CDNA matrix core at 16x16x16 bf16, f32 accumulation. Per lane l and t
// in 0..4: A (row l%16, k 4(l/16)+t), B (col l%16, same k), D (row
// 4(l/16)+t, col l%16), so a row-major A and a k-major B each load one
// 8-byte read a lane.

#include "prelude/device.hip"

namespace pie {

typedef short mfma_ab __attribute__((ext_vector_type(4)));
typedef float mfma_cd __attribute__((ext_vector_type(4)));

__device__ __forceinline__ mfma_cd mfma16(mfma_ab a, mfma_ab b, mfma_cd c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ mfma_ab load4(const bf16* p) {
    return *reinterpret_cast<const mfma_ab*>(p);
}

__device__ __forceinline__ mfma_ab pack4(float a, float b, float c, float d) {
    const bf16 v[4] = {from_f32<bf16>(a), from_f32<bf16>(b), from_f32<bf16>(c), from_f32<bf16>(d)};
    return *reinterpret_cast<const mfma_ab*>(v);
}

// The f32 form, 16 x 16 x 4: A (row l%16, k l/16), B (col l%16, k l/16), D as
// above. Exact f32 products, for the small solves bf16 would round.
__device__ __forceinline__ mfma_cd mfma4_f32(float a, float b, mfma_cd c) {
    return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
}

// Two k-steps from one 16-byte read per operand: lane group g = l/16 takes
// k 8g..8g+3 into the first step and 8g+4..8g+7 into the second. The
// assignment of k to steps is the operands' own business, so A and B read
// alike and the product is the same.
struct mfma_ab2 {
    mfma_ab lo, hi;
};

__device__ __forceinline__ mfma_ab2 load8(const bf16* p) {
    return *reinterpret_cast<const mfma_ab2*>(p);
}

__device__ __forceinline__ mfma_cd mfma16x2(mfma_ab2 a, mfma_ab2 b, mfma_cd c) {
    return mfma16(a.hi, b.hi, mfma16(a.lo, b.lo, c));
}

__device__ __forceinline__ mfma_cd mfma_zero() { return mfma_cd{0.f, 0.f, 0.f, 0.f}; }

// One 16 x 16 tile of A · Bᵀ over K (a multiple of 16): A rows at a + m·lda,
// B rows at b + n·ldb, both k-contiguous.
template <int K>
__device__ __forceinline__ mfma_cd mfma_tile(const bf16* a, int lda, const bf16* b, int ldb, mfma_cd c) {
    const int l = lane(), r = l % 16, k4 = 4 * (l / 16);
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 16)
        c = mfma16(load4(a + r * lda + k0 + k4), load4(b + r * ldb + k0 + k4), c);
    return c;
}

// The lane's four D cells: (row 4(l/16)+t, col l%16).
template <class F>
__device__ __forceinline__ void mfma_each(mfma_cd d, F&& f) {
    const int l = lane();
#pragma unroll
    for (int t = 0; t < 4; ++t) f(4 * (l / 16) + t, l % 16, d[t]);
}

}  // namespace pie
