// metal_rand_mv.metal
// ─────────────────────────────────────────────────────────────
// Metal kernels for Triton-exact Philox-4x32 PRNG + Box-Muller.
//
// Kernels:
//   randn_generate_f32          — W[b,i,o] = S[i,o] * N(0,1; seed=seeds[b])
//   randn_matmul_{f32,f16,bf16} — y[b,o] = Σ_k x[b,k] * S[k,o] * N(0,1)
//                                  Fused: random weights stay in registers.
//
// Philox constants and Box-Muller match triton.language.random exactly,
// producing bit-identical float32 for the same (seed, offset) pair.
// ─────────────────────────────────────────────────────────────

#include <metal_stdlib>
using namespace metal;

// ── Philox-4x32 (10 rounds, fully unrolled) ─────────────────────────────

constant uint PHILOX_KEY_A   = 0x9E3779B9u;
constant uint PHILOX_KEY_B   = 0xBB67AE85u;
constant uint PHILOX_ROUND_A = 0xD2511F53u;
constant uint PHILOX_ROUND_B = 0xCD9E8D57u;

inline void philox_4x32_10(
    thread uint &c0, thread uint &c1, thread uint &c2, thread uint &c3,
    uint k0, uint k1
) {
    uint _c0, _c2;

    #define PHILOX_ROUND(rk0, rk1) \
        _c0 = c0; _c2 = c2; \
        c0 = mulhi(_c2, PHILOX_ROUND_B) ^ c1 ^ (rk0); \
        c2 = mulhi(_c0, PHILOX_ROUND_A) ^ c3 ^ (rk1); \
        c1 = PHILOX_ROUND_B * _c2; \
        c3 = PHILOX_ROUND_A * _c0;

    PHILOX_ROUND(k0,                  k1)
    PHILOX_ROUND(k0 +   PHILOX_KEY_A, k1 +   PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 2*PHILOX_KEY_A, k1 + 2*PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 3*PHILOX_KEY_A, k1 + 3*PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 4*PHILOX_KEY_A, k1 + 4*PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 5*PHILOX_KEY_A, k1 + 5*PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 6*PHILOX_KEY_A, k1 + 6*PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 7*PHILOX_KEY_A, k1 + 7*PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 8*PHILOX_KEY_A, k1 + 8*PHILOX_KEY_B)
    PHILOX_ROUND(k0 + 9*PHILOX_KEY_A, k1 + 9*PHILOX_KEY_B)

    #undef PHILOX_ROUND
}

// ── Box-Muller: uint32 → N(0,1), matching tl.randn exactly ─────────────

inline float uint_to_uniform(uint x) {
    const float SCALE = 4.6566127342e-10f;  // ≈ 1/2^31
    int sx = as_type<int>(x);
    sx = (sx < 0) ? (-sx - 1) : sx;
    return float(sx) * SCALE;
}

inline float philox_randn(uint seed, uint offset) {
    uint c0 = offset, c1 = 0u, c2 = 0u, c3 = 0u;
    philox_4x32_10(c0, c1, c2, c3, seed, 0u);
    float u1 = max(1.0e-7f, uint_to_uniform(c0));
    float u2 = uint_to_uniform(c1);
    return sqrt(-2.0f * log(u1)) * cos(6.283185307179586f * u2);
}

// ═════════════════════════════════════════════════════════════════════════
//  randn_generate_f32
//  Grid: (ceil(I*O / 256), B, 1)   Group: (256, 1, 1)
//  One thread per (b, i, o) element.
// ═════════════════════════════════════════════════════════════════════════

kernel void randn_generate_f32(
    device const int*   seeds   [[buffer(0)]],   // [B]
    device const float* S       [[buffer(1)]],   // [I, O]
    device float*       output  [[buffer(2)]],   // [B, I, O]
    device const float* params  [[buffer(3)]],   // [B, I, O, col_offset, global_cols]
    uint2 gid [[thread_position_in_grid]]
) {
    const int B  = (int)params[0], I = (int)params[1], O = (int)params[2];
    const int col_offset = (int)params[3], global_cols = (int)params[4];

    const int flat_io = (int)gid.x, b = (int)gid.y;
    if (b >= B || flat_io >= I * O) return;

    const int i = flat_io / O, o = flat_io % O;
    const int seed_val = seeds[b];
    const int out_idx  = b * I * O + i * O + o;

    if (seed_val == 0) { output[out_idx] = 0.0f; return; }

    const uint offset = uint(i) * uint(global_cols) + uint(o) + uint(col_offset);
    output[out_idx] = philox_randn(as_type<uint>(seed_val), offset) * S[i * O + o];
}

// ═════════════════════════════════════════════════════════════════════════
//  randn_matmul  (fused — never materializes the weight matrix)
//  Grid: (O * 32, B, 1)   Group: (32, 1, 1)
//  One simdgroup per (b, o). 32 lanes split K in strides of 32,
//  then simd_sum reduces the partial sums.
// ═════════════════════════════════════════════════════════════════════════

template<typename T>
kernel void randn_matmul_kernel(
    device const T*     x       [[buffer(0)]],   // [B, I]
    device const int*   seeds   [[buffer(1)]],   // [B]
    device const float* S       [[buffer(2)]],   // [I, O]
    device float*       output  [[buffer(3)]],   // [B, O]
    device const float* params  [[buffer(4)]],   // [B, I, O, col_offset, global_cols]
    uint2 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const int B  = (int)params[0], I = (int)params[1], O = (int)params[2];
    const int col_offset = (int)params[3], global_cols = (int)params[4];

    const int o = (int)gid.x / 32, b = (int)gid.y;
    if (b >= B || o >= O) return;

    const int seed_val = seeds[b];
    if (seed_val == 0) {
        if (lane == 0) output[b * O + o] = 0.0f;
        return;
    }

    const uint seed_u = as_type<uint>(seed_val);
    const uint o_off  = uint(o) + uint(col_offset);
    const uint gcols  = uint(global_cols);

    float acc = 0.0f;
    for (int k = (int)lane; k < I; k += 32) {
        float w = philox_randn(seed_u, uint(k) * gcols + o_off) * S[k * O + o];
        acc += float(x[b * I + k]) * w;
    }

    acc = simd_sum(acc);
    if (lane == 0) output[b * O + o] = acc;
}

// Explicit template instantiations
template [[host_name("randn_matmul_f32")]]
kernel void randn_matmul_kernel<float>(
    device const float*, device const int*, device const float*,
    device float*, device const float*, uint2, uint);

template [[host_name("randn_matmul_f16")]]
kernel void randn_matmul_kernel<half>(
    device const half*, device const int*, device const float*,
    device float*, device const float*, uint2, uint);

template [[host_name("randn_matmul_bf16")]]
kernel void randn_matmul_kernel<bfloat>(
    device const bfloat*, device const int*, device const float*,
    device float*, device const float*, uint2, uint);
