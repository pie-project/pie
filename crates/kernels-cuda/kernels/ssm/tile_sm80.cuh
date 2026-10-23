#pragma once

#include "prim/mma_sm80.cuh"
#include "prim/mma.cuh"
#include "ssm/chunk.cuh"
#include "ssm/copy.cuh"

namespace pie::ssm {

// The chunked scan's tensor-core pieces: a bf16 `m16n8k16` with f32
// accumulators and `ldmatrix` fragment loads, plain and transposed.

// Lane `l`'s row address for each fragment an x4 `ldmatrix` lands, over a
// row-major tile with leading dimension `ld`:
// the A operand of rows `m0..m0+16`, columns `k0..k0+16`;
__device__ __forceinline__ const bf16* gdn_at_a(const bf16* s, int ld, int m0, int k0, int l) {
    return s + (m0 + (l & 15)) * ld + k0 + ((l >> 4) << 3);
}

// the B operand of two eight-wide column tiles `n0`, `n0 + 8` stored as
// `[n][k]` rows (no transpose);
__device__ __forceinline__ const bf16* gdn_at_bt(const bf16* s, int ld, int n0, int k0, int l) {
    return s + (n0 + (l & 7) + ((l >> 4) << 3)) * ld + k0 + (((l >> 3) & 1) << 3);
}

// the B operand of the same two tiles stored as `[k][n]` rows (transposed);
__device__ __forceinline__ const bf16* gdn_at_b(const bf16* s, int ld, int k0, int n0, int l) {
    return s + (k0 + (l & 7) + (((l >> 3) & 1) << 3)) * ld + n0 + ((l >> 4) << 3);
}

// the A operand of rows `m0..m0+16` stored as `[k][m]` rows (transposed).
__device__ __forceinline__ const bf16* gdn_at_at(const bf16* s, int ld, int k0, int m0, int l) {
    return s + (k0 + (l & 7) + ((l >> 4) << 3)) * ld + m0 + (((l >> 3) & 1) << 3);
}

// A 128-wide bf16 row padded so the eight rows one `ldmatrix` reads land in
// distinct banks, and a 64-wide row likewise.
constexpr int GDN_LD = 136;
constexpr int GDN_LDC = 72;
constexpr int GDN_LDA = 68;

// The chunked KDA rule in sixteen-token chunks, a decay a key channel
// (`S = S diag(exp(g_t))` before each update). Over sixteen steps a chunk's
// decay stays within sixteen steps' decay of one, so its products need no
// split: `exp(G_i - G_j) = exp(G_i - G_0) exp(G_0 - G_j)`, both factors in
// bf16.
//
// The chunk-parallel half: two warps a (chunk, head), eight rows each, lane
// `l` the key channels `4l..4l+3`. They form the gates, their running sum
// `G` down the chunk, the unit-length rows (the queries scaled by
// `1 / sqrt(D)`), and land
//   q_g = q exp(G),  k_g = k exp(G_L - G),  k_e = k exp(G),  exp(G_L),
//   A = tril(beta_i k_i . k_j exp(G_i - G_j), -1),  T diag(beta) = (I + A)^-1 diag(beta),
//   P = tril(q_i . k_j exp(G_i - G_j)).
constexpr int KDA_CH = 16;
// The prep's exp2, log2 and tanh: the approximate unit ops, subnormals
// flushed.
__device__ __forceinline__ float kda_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float kda_lg2(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float kda_tanh(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// Two products an instruction where the card pairs them (sm_100 on), the
// same rounding as apart.
__device__ __forceinline__ float2 kda_mul2(float2 a, float2 b) {
#if __CUDA_ARCH__ >= 1000
    unsigned long long x, y, r;
    asm("mov.b64 %0, {%1, %2};" : "=l"(x) : "f"(a.x), "f"(a.y));
    asm("mov.b64 %0, {%1, %2};" : "=l"(y) : "f"(b.x), "f"(b.y));
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(x), "l"(y));
    float2 o;
    asm("mov.b64 {%0, %1}, %2;" : "=f"(o.x), "=f"(o.y) : "l"(r));
    return o;
#else
    return make_float2(a.x * b.x, a.y * b.y);
#endif
}
constexpr int KDA16_TASKS = 2;
// A (chunk, head)'s shared memory: three bf16 chunk tiles (the scaled keys,
// their column twins, the scaled queries), `A` in f32, and the two warps'
// running sums.
constexpr int KDA16_LDT = 24;
constexpr int KDA16_TASK_SMEM = 3 * KDA_CH * GDN_LD * 2 + KDA_CH * (KDA_CH + 1) * 4 + 3 * 128 * 4;

}
