#pragma once

// Activations quantized for W8A8 / W4A8 products (DeepSeek's `act_quant`):
// e4m3 under a ue8m0 scale a `GRAN`, s = 2^ceil(log2(max(amax, 1e-4) /
// 448)), rounded to nearest even. They land as their values q·s in bf16,
// exact there, as an fp8b128 weight's values are. gfx942 converts fnuz, so
// the OCP codes are read and written by their bits.

#include "prim/device.hip"

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
        r = rintf(a * __uint_as_float(static_cast<u32>(127 + 3 - e) << 23)) * step;
    }
    r = fminf(r, 448.f);
    return v < 0.f ? -r : r;
}

// x's rows quantized as E4M3 in GRAN-wide blocks, landed as their bf16
// values. Eight elements a thread, GRAN / 8 threads a block.
template <int GRAN>
__global__ __launch_bounds__(256) void act_quant_e4m3_values(const bf16* __restrict__ x, bf16* __restrict__ y,
                                                             i32 rows, i32 k, i32 fan, const u32* __restrict__ win) {
    constexpr int LANES = GRAN / 8;  // a thread eight elements
    static_assert(GRAN % 8 == 0 && LANES <= kWave && (LANES & (LANES - 1)) == 0, "a block a lane run");
    const Window window{win};
    // The launch's threads fit 32 bits (the launcher's check).
    const u32 per = static_cast<u32>(k) / 8;
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= static_cast<u32>(rows) * per) return;
    const u32 r = t / per;
    if (!window.live(r, fan)) return;
    const long long at = window.plane(r, fan) * k + (t - r * per) * 8;
    const Pack8<bf16> in = *reinterpret_cast<const Pack8<bf16>*>(x + at);
    float v[8];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = to_f32(in.v[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int off = LANES / 2; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, kWave));
    amax = fmaxf(amax, 1e-4f);
    // 2^ceil(log2(amax / 448)), from the bits of the quotient.
    const u32 bits = __float_as_uint(amax * (1.f / 448.f));
    const int e = static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    const float s = __uint_as_float(static_cast<u32>(e + 127) << 23);
    const float inv = __uint_as_float(static_cast<u32>(127 - e) << 23);  // 1 / s, exact
    Pack8<bf16> out;
#pragma unroll
    for (int i = 0; i < 8; ++i) out.v[i] = from_f32<bf16>(e4m3fn_round(fminf(fmaxf(v[i] * inv, -448.f), 448.f)) * s);
    *reinterpret_cast<Pack8<bf16>*>(y + at) = out;
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
