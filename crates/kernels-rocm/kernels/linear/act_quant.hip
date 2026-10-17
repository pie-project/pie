#pragma once

// Activations quantized for a W8A8 / W4A8 product, and the FP8 projections
// (DeepSeek's `fp8b128`: e4m3 codes under one ue8m0 exponent a 128x128
// block) they multiply.
//
//   E4M3   e4m3 (OCP e4m3fn) under a ue8m0 scale a `GRAN` (DeepSeek's
//          `act_quant`): the amax floored at 1e-4, `s = 2^ceil(log2(amax /
//          448))`, `q = e4m3(clamp(x / s, ±448))`, rounded to nearest even.
//
// The quantized activations land as their values, `q · s`, in bf16: every
// such value is exact in bf16 (three mantissa bits under a power of two), so
// a bf16 product of them against the weight's exact values sums the same
// products the W8A8 GEMM does, in f32. An fp8b128 weight's values, `code ·
// 2^(e − 127)`, are exact in bf16 too.
//
// gfx942's fp8 conversions are the FNUZ encoding, not the OCP one the
// checkpoints and DeepSeek's recipe use, so the codes are read and written
// here by their bits.

#include "prelude/device.hip"

namespace pie::linear {

// The value of OCP e4m3fn code `b` (0x7f / 0xff, its NaNs, are not weights).
__device__ __forceinline__ float e4m3fn_value(u32 b) {
    const u32 e = (b >> 3) & 0xFu, m = b & 0x7u;
    const float mag = e == 0u ? static_cast<float>(m) * 0.001953125f  // m · 2^-9
                              : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) != 0u ? -mag : mag;
}

// `v` rounded to the nearest e4m3fn value, ties to even; |v| ≤ 448.
__device__ __forceinline__ float e4m3fn_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a < 0.015625f) {
        // Subnormal: steps of 2^-9.
        r = rintf(a * 512.f) * 0.001953125f;
    } else {
        // Three mantissa bits under the binade's exponent.
        const int e = static_cast<int>((__float_as_uint(a) >> 23) & 0xFFu) - 127;
        const float step = __uint_as_float(static_cast<u32>(e - 3 + 127) << 23);
        r = rintf(a / step) * step;
    }
    r = fminf(r, 448.f);
    return v < 0.f ? -r : r;
}

// x's rows quantized as E4M3 in GRAN-wide blocks, landed as their bf16
// values. One wavefront a block, GRAN / 64 elements a lane (GRAN ≥ 64), or
// 64 / GRAN blocks a wavefront.
template <int GRAN>
__global__ __launch_bounds__(256) void act_quant_e4m3_values(const bf16* __restrict__ x, bf16* __restrict__ y,
                                                             i32 rows, i32 k, const u32* __restrict__ win) {
    static_assert(GRAN % 8 == 0 && (GRAN % kWave == 0 || kWave % GRAN == 0), "a block a lane run");
    constexpr int EACH = GRAN >= kWave ? GRAN / kWave : 1;
    constexpr int SPAN = GRAN >= kWave ? kWave : GRAN;  // lanes a block spans
    const Window window{win};
    const int blocks = k / GRAN;
    const long long w = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / kWave;
    const int l = lane();
    const long long slot = w * (kWave / SPAN) + l / SPAN;
    if (slot >= static_cast<long long>(rows) * blocks) return;
    const long long r = slot / blocks;
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    const int b = static_cast<int>(slot % blocks);
    const int sub = l % SPAN;
    const long long at = row * k + static_cast<long long>(b) * GRAN + sub * EACH;
    float v[EACH];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < EACH; ++i) {
        v[i] = to_f32(x[at + i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int off = SPAN / 2; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, kWave));
    amax = fmaxf(amax, 1e-4f);
    // 2^ceil(log2(amax / 448)), from the bits of the quotient.
    const u32 bits = __float_as_uint(amax * (1.f / 448.f));
    const int e = static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    const float s = __uint_as_float(static_cast<u32>(e + 127) << 23);
#pragma unroll
    for (int i = 0; i < EACH; ++i) y[at + i] = from_f32<bf16>(e4m3fn_round(fminf(fmaxf(v[i] / s, -448.f), 448.f)) * s);
}

// The scale of fp8b128 block (n / 128, kk / 128): 2^(e − 127), 2^-127 (an
// f32 denormal) at e = 0.
__device__ __forceinline__ float e8m0_value(u32 e) { return __uint_as_float(e == 0u ? 0x400000u : e << 23); }

// An fp8b128 projection's values in bf16, `[n, k]`: eight codes a thread.
// B: the block side, 128 (`fp8b128`) or 32 (`fp8b32`).
template <int B = 128>
__global__ __launch_bounds__(256) void fp8b128_decode(const u8* __restrict__ codes, const u8* __restrict__ scales,
                                                      bf16* __restrict__ out, i32 n, i32 k) {
    const long long i8 = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i8 >= static_cast<long long>(n) * k) return;
    const int r = static_cast<int>(i8 / k), c = static_cast<int>(i8 % k);
    const int kb = (k + B - 1) / B;
    const float s = e8m0_value(scales[static_cast<long long>(r / B) * kb + c / B]);
    const unsigned long long word = *reinterpret_cast<const unsigned long long*>(codes + i8);
    Pack8<bf16> o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.v[j] = from_f32<bf16>(e4m3fn_value(static_cast<u32>(word >> (8 * j)) & 0xFFu) * s);
    *reinterpret_cast<Pack8<bf16>*>(out + i8) = o;
}

// An fp8b32 projection in the FP8 matrix core's order (`fp8_unit_at` in
// linear/fp8.hip: each 16-row strip K / 64 units of 1 KB, lane l's 16 bytes
// row l % 16's codes 8·(l / 16) .. +8 of the unit's first 32-code step, then
// of its second) back to row-major bf16 values under its exponents, `[n, k]`.
// A thread a lane's 16 bytes.
__global__ __launch_bounds__(256) void fp8b32_lanes_decode(const u8* __restrict__ codes, const u8* __restrict__ scales,
                                                           bf16* __restrict__ out, i32 n, i32 k) {
    const long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= static_cast<long long>(n) * k / 16) return;
    const long long strip = q / k;
    const int within = static_cast<int>(q % k), unit = within / 64, l = within % 64;
    const long long row = strip * 16 + l % 16;
    const int col = unit * 64 + 8 * (l / 16);
    const int kb = k / 32;
    const uint4 v = *reinterpret_cast<const uint4*>(codes + q * 16);
    const u32 w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int c = col + 32 * half;
        const float s = e8m0_value(scales[(row / 32) * kb + c / 32]);
        Pack8<bf16> o;
#pragma unroll
        for (int j = 0; j < 8; ++j) o.v[j] = from_f32<bf16>(e4m3fn_value((w[2 * half + j / 4] >> (8 * (j % 4))) & 0xFFu) * s);
        *reinterpret_cast<Pack8<bf16>*>(out + row * k + c) = o;
    }
}

// y = x · wᵀ for an fp8b128 w and few rows of x (bf16, the quantized values),
// the codes decoded in registers: a wavefront an output column, sixteen
// codes a lane a step (one 128-block scale each), the rows' sums kept per
// lane and reduced at the end. ROWS rows of x a pass.
template <int ROWS, int B = 128>
__global__ __launch_bounds__(256) void fp8b128_gemv(const bf16* __restrict__ x, const u8* __restrict__ codes,
                                                    const u8* __restrict__ scales, bf16* __restrict__ y, i32 m,
                                                    i32 n, i32 k, const u32* __restrict__ win) {
    const Window window{win};
    if (win != nullptr) m = min(m, static_cast<int>(win[0]));
    const int col = static_cast<int>((static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / kWave);
    if (col >= n) return;
    const int l = lane();
    const int kb = (k + B - 1) / B;
    const u8* wrow = codes + static_cast<long long>(col) * k;
    const u8* srow = scales + static_cast<long long>(col / B) * kb;
    const long long first = window.plane(0);
    for (int r0 = 0; r0 < m; r0 += ROWS) {
        float acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
        for (int c = 16 * l; c < k; c += 16 * kWave) {
            const uint4 word = *reinterpret_cast<const uint4*>(wrow + c);
            const float s = e8m0_value(srow[c / B]);
            float w[16];
            const u32 parts[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
            for (int j = 0; j < 16; ++j) w[j] = e4m3fn_value((parts[j / 4] >> (8 * (j % 4))) & 0xFFu) * s;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r0 + r >= m) break;
                const bf16* xr = x + (first + r0 + r) * k + c;
                const Pack8<bf16> a0 = *reinterpret_cast<const Pack8<bf16>*>(xr);
                const Pack8<bf16> a1 = *reinterpret_cast<const Pack8<bf16>*>(xr + 8);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[r] += to_f32(a0.v[j]) * w[j] + to_f32(a1.v[j]) * w[8 + j];
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float v = acc[r];
#pragma unroll
            for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
            if (l == 0 && r0 + r < m) y[(first + r0 + r) * n + col] = from_f32<bf16>(v);
        }
    }
}

}  // namespace pie::linear
