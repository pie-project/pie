#pragma once

// Activations quantized for a low-precision product, in the two formats a
// deployment states:
//
//   NVFP4  e2m1 under an e4m3 scale a 16, the scales under a static global
//          `gs = 1 / input_scale` (vLLM's and TensorRT-LLM's recipe, their
//          approximate reciprocals included): `sf = e4m3(gs * amax / 6)`,
//          `q = e2m1(x / (sf / gs))`, the value `q * sf / gs`.
//   E4M3   e4m3 under a ue8m0 scale a `kGran` (DeepSeek's `act_quant`): the
//          amax floored at 1e-4, `s = 2^ceil(log2(amax / 448))`,
//          `q = e4m3(clamp(x / s, ±448))`.
//
// Each lands either as the tcgen05 GEMM reads it (codes, and scales in its
// 512-byte atoms of 128 rows x 4 scale columns) or read back to bf16 without
// the global, `q * sf`, every such product exact in bf16, for a bf16 product
// that multiplies by the global once.
//
// Out row `r` reads `x` row `r / div`; with `routes`, its global is its
// route's expert's, and a dropped route (`routes[r] < 0`) lands zeros. With
// `pos`, its codes land at row `pos[r]` of a grouped layout (a dropped route
// nowhere), its scales at that row's atom row: the same row for the
// contiguous layout, expert `pos / m_max`'s whole row blocks for the masked
// one (`m_max > 0`).

#include "linear/quant.cuh"

#include "cuda_fp8.h"

namespace pie::linear {

constexpr int kQuantCodes = 0;
constexpr int kQuantValues = 1;

// The byte of scale `(row, col)` in an atom-layout tensor of `cols` columns.
__device__ __forceinline__ long long atom_offset(int row, int col, int cols) {
    const int atoms_k = (cols + 3) / 4;
    return (static_cast<long long>(row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16
        + ((row / 32) % 4) * 4 + col % 4;
}

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

// Round to the nearest e2m1 code, ties to the even code, saturating at 6;
// the sign in bit 3.
__device__ __forceinline__ unsigned e2m1_code(float v) {
    const float a = fabsf(v);
    unsigned c;
    if (a <= 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.f) c = 6;
    else c = 7;
    return (v < 0.f ? 8u : 0u) | c;
}

__device__ __forceinline__ float e4m3_value(u8 b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

// DeepSeek's ue8m0 scale exponent for a block whose largest magnitude is
// `amax`: 2^ceil(log2(max(amax, 1e-4) / 448)), read off the quotient's bits.
__device__ __forceinline__ int e4m3_block_exp(float amax) {
    amax = fmaxf(amax, 1e-4f);
    const unsigned bits = __float_as_uint(amax * (1.f / 448.f));
    return static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
}

// The atom row of grouped row `o`.
__device__ __forceinline__ int atom_row(int o, int m_max) {
    if (m_max == 0) return o;
    const int pad = (m_max + 127) / 128 * 128;
    return o / m_max * pad + o % m_max;
}

__device__ __forceinline__ float global_of(const float* __restrict__ input_scales,
                                           const i32* __restrict__ routes, int r, bool* dropped) {
    int e = 0;
    if (routes != nullptr) {
        e = routes[r];
        if (e < 0) {
            *dropped = true;
            return 0.f;
        }
    }
    return __fdiv_rn(1.f, input_scales[e]);
}

// One thread a 16-wide block. Under `kFan` (codes only) row `r` is a token
// read once, its block's codes landed at each of its `div` routes' rows
// (`pos[r * div + j]`) under that route's expert's global, as each route's
// own row would land them.
template <int kMode, bool kFan = false>
__global__ void act_quant_nvfp4(
    const bf16* __restrict__ x,
    const i32* __restrict__ routes,
    const float* __restrict__ input_scales,
    const i32* __restrict__ pos,
    int m_max,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    const int blocks = k / 16;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(rows) * blocks) return;
    const int r = static_cast<int>(at / blocks);
    const int b = static_cast<int>(at % blocks);
    if constexpr (kFan) {
        uint4 raw[2];
        const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<long long>(r) * k + b * 16);
        raw[0] = src[0];
        raw[1] = src[1];
        const bf16* h = reinterpret_cast<const bf16*>(raw);
        float v[16];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = bf16_to_f32(h[i]);
            amax = fmaxf(amax, fabsf(v[i]));
        }
        // The routes' globals are one (load lays the experts' largest), so
        // the block's codes are quantized once and land at each route.
        float last_gs = -1.f;
        uint2 w = make_uint2(0u, 0u);
        u8 sf_code = 0;
        for (int j = 0; j < div; ++j) {
            const int route = r * div + j;
            const int o = pos[route];
            if (o < 0) continue;
            bool dropped = false;
            const float gs = global_of(input_scales, routes, route, &dropped);
            if (dropped) continue;
            if (gs != last_gs) {
                last_gs = gs;
                const float sf = gs * (amax * rcp_approx_ftz(6.f));
                sf_code = __nv_cvt_float_to_fp8(sf, __NV_SATFINITE, __NV_E4M3);
                const float sf_value = e4m3_value(sf_code);
                const float out_scale =
                    sf_value != 0.f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(gs)) : 0.f;
                u32 c[2] = {0u, 0u};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const u32 byte =
                        e2m1_code(v[2 * i] * out_scale) | (e2m1_code(v[2 * i + 1] * out_scale) << 4);
                    c[i / 4] |= byte << (8 * (i % 4));
                }
                w = make_uint2(c[0], c[1]);
            }
            *reinterpret_cast<uint2*>(codes + static_cast<long long>(o) * (k / 2) + b * 8) = w;
            atoms[atom_offset(atom_row(o, m_max), b, blocks)] = sf_code;
        }
        return;
    }
    // A route that lands nowhere -- dropped, or past a captured fire's live
    // window, whose route ids this fire never wrote -- is not read: its id
    // is no expert's, so it indexes no input scale.
    const int o = pos != nullptr ? pos[r] : r;
    if (kMode == kQuantCodes && o < 0) return;
    bool dropped = false;
    const float gs = global_of(input_scales, routes, r, &dropped);
    const bf16* src = x + static_cast<long long>(r / div) * k + b * 16;
    float v[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = dropped ? 0.f : bf16_to_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float sf = gs * (amax * rcp_approx_ftz(6.f));
    const u8 sf_code = __nv_cvt_float_to_fp8(sf, __NV_SATFINITE, __NV_E4M3);
    const float sf_value = e4m3_value(sf_code);
    const float out_scale = sf_value != 0.f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(gs)) : 0.f;
    if constexpr (kMode == kQuantCodes) {
        u8* dst = codes + static_cast<long long>(o) * (k / 2) + b * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[i] = static_cast<u8>(e2m1_code(v[2 * i] * out_scale)
                                     | (e2m1_code(v[2 * i + 1] * out_scale) << 4));
        }
        atoms[atom_offset(atom_row(o, m_max), b, blocks)] = sf_code;
    } else {
        bf16* dst = values + static_cast<long long>(r) * k + b * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            dst[i] = f32_to_bf16(kFp4Lut[e2m1_code(v[i] * out_scale)] * sf_value);
        }
    }
}

// One warp a `kGran`-wide block, each lane `kGran / 32` elements. Under
// `kFan` (codes only) row `r` is a token read once and landed at each of its
// `div` routes' rows (`pos[r * div + j]`, none where negative), as each
// route's own row would land it.
template <int kMode, int kGran, bool kFan = false>
__global__ void act_quant_e4m3(
    const bf16* __restrict__ x,
    const i32* __restrict__ pos,
    int m_max,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    constexpr int kEach = kGran / 32;
    const int blocks = k / kGran;
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= static_cast<long long>(rows) * blocks) return;
    const int r = static_cast<int>(warp / blocks);
    const int b = static_cast<int>(warp % blocks);
    const int src_row = kFan ? r : r / div;
    const bf16* src = x + static_cast<long long>(src_row) * k + b * kGran + lane * kEach;
    float v[kEach];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < kEach; ++i) {
        v[i] = bf16_to_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const int e = e4m3_block_exp(amax);
    const float s = __uint_as_float(static_cast<unsigned>(e + 127) << 23);
    u8 q[kEach];
#pragma unroll
    for (int i = 0; i < kEach; ++i) {
        q[i] = __nv_cvt_float_to_fp8(fminf(fmaxf(v[i] / s, -448.f), 448.f),
                                     __NV_SATFINITE, __NV_E4M3);
    }
    if constexpr (kMode == kQuantCodes) {
        for (int j = 0; j < (kFan ? div : 1); ++j) {
            const int o = kFan ? pos[static_cast<long long>(r) * div + j] : (pos != nullptr ? pos[r] : r);
            if (o < 0) continue;
            u8* dst = codes + static_cast<long long>(o) * k + b * kGran + lane * kEach;
#pragma unroll
            for (int i = 0; i < kEach; ++i) dst[i] = q[i];
            if (lane == 0) atoms[atom_offset(atom_row(o, m_max), b, blocks)] = static_cast<u8>(e + 127);
        }
    } else {
        bf16* dst = values + static_cast<long long>(r) * k + b * kGran + lane * kEach;
#pragma unroll
        for (int i = 0; i < kEach; ++i) dst[i] = f32_to_bf16(e4m3_value(q[i]) * s);
    }
}

// The e4m3 quantizers eight elements a lane: a `kGran`-wide block is
// kGran / 8 lanes (a warp 32 / that many blocks), read in one 16-byte load
// and landed in one 8-byte store a lane. The scale a power of two, the
// quotient `v / s` is `v * 2^-e` exactly, so the codes and values are the
// one-element-a-lane kernels' byte for byte.
template <int kGran>
struct E4m3Block {
    static constexpr int kLanes = kGran / 8;
    float v[8];
    int e;
    // Lane `lane`'s eight of the block at `src` (its kLanes lanes adjacent).
    __device__ __forceinline__ void load(const bf16* __restrict__ src, int lane) {
        const uint4 w = *reinterpret_cast<const uint4*>(src + (lane % kLanes) * 8);
        const unsigned words[4] = {w.x, w.y, w.z, w.w};
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = __uint_as_float(words[i] << 16);
            v[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000u);
            amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
        }
#pragma unroll
        for (int o = kLanes / 2; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        }
        e = e4m3_block_exp(amax);
    }
    // The eight codes, packed low element first.
    __device__ __forceinline__ unsigned long long codes() const {
        const float inv = __uint_as_float(static_cast<unsigned>(127 - e) << 23);
        // Two codes a convert, the low element in the low byte.
        unsigned half[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float lo = fminf(fmaxf(v[2 * i] * inv, -448.f), 448.f);
            const float hi = fminf(fmaxf(v[2 * i + 1] * inv, -448.f), 448.f);
            // The shim's convert: the instruction from sm_89, its portable
            // rounding below (sm_80 serves its W8A16 load through here).
            half[i] = static_cast<unsigned short>(
                __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
        }
        return static_cast<unsigned long long>(half[0] | (half[1] << 16)) |
               (static_cast<unsigned long long>(half[2] | (half[3] << 16)) << 32);
    }
    // A producer's eight of the block, held in registers: each rounded to
    // bf16 first, as a quantizer reading the producer's bf16 row back
    // would see it, so the codes and scale are `load`'s from that row.
    __device__ __forceinline__ void hold(const float (&x)[8]) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = bf16_to_f32(f32_to_bf16(x[i]));
            amax = fmaxf(amax, fabsf(v[i]));
        }
#pragma unroll
        for (int o = kLanes / 2; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        }
        e = e4m3_block_exp(amax);
    }
    // Lands the block as row `o`'s block `b` of a GEMM's operand: lane
    // `lane`'s eight codes, and the block's ue8m0 scale in the atom layout.
    __device__ __forceinline__ void store(u8* __restrict__ to, u8* __restrict__ atoms, int o,
                                          int b, int k, int m_max, int lane) const {
        const int sub = lane % kLanes;
        *reinterpret_cast<unsigned long long*>(to + static_cast<long long>(o) * k + b * kGran +
                                               sub * 8) = codes();
        if (sub == 0) {
            atoms[atom_offset(atom_row(o, m_max), b, k / kGran)] = static_cast<u8>(e + 127);
        }
    }
};

// DeepSeek's fp8 1x128 activation landed by its producer, in place of an
// `act_quant_e4m3` launch over the producer's bf16 row: lane `lane` holds
// elements [8 * (lane % 16), +8) of row `o`'s 128-wide block `b` (`k` the
// row's width), a block's 16 lanes adjacent and 16-aligned in the warp, all
// 32 lanes calling. The same bytes as act_quant_e4m3 (mode 0, `m_max` its
// masked rows a group or 0) lands from that row.
__device__ __forceinline__ void e4m3_block_store(const float (&x)[8], int lane, int o, int b,
                                                 int k, int m_max, u8* __restrict__ codes,
                                                 u8* __restrict__ atoms) {
    E4m3Block<128> q;
    q.hold(x);
    q.store(codes, atoms, o, b, k, m_max, lane);
}

// `act_quant_e4m3` eight elements a lane (`kFan` as there).
template <int kMode, int kGran, bool kFan = false>
__global__ void act_quant_e4m3_v8(
    const bf16* __restrict__ x,
    const i32* __restrict__ pos,
    int m_max,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    // A programmatic dependent (`kernel.pdl`) reads nothing until its
    // producer is done; its consumer (a tcgen05 GEMM) may launch from here.
    pdl_wait();
    pdl_launch();
    using Block = E4m3Block<kGran>;
    const int blocks = k / kGran;
    const long long at = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    const long long slot = at / Block::kLanes;
    const int lane = static_cast<int>(threadIdx.x % 32);
    // A slot past the last block keeps its lanes in the shuffle, landing
    // nothing.
    const bool live = slot < static_cast<long long>(rows) * blocks;
    const long long sl = live ? slot : 0;
    const int r = static_cast<int>(sl / blocks);
    const int b = static_cast<int>(sl % blocks);
    Block q;
    q.load(x + static_cast<long long>(kFan ? r : r / div) * k + b * kGran, lane);
    if (!live) return;
    if constexpr (kMode == kQuantCodes) {
        for (int j = 0; j < (kFan ? div : 1); ++j) {
            const int o = kFan ? pos[static_cast<long long>(r) * div + j] : (pos != nullptr ? pos[r] : r);
            if (o < 0) continue;
            q.store(codes, atoms, o, b, k, m_max, lane);
        }
    } else {
        const int sub = lane % Block::kLanes;
        const float s = __uint_as_float(static_cast<unsigned>(q.e + 127) << 23);
        const unsigned long long c = q.codes();
        unsigned w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float lo = e4m3_value(static_cast<u8>(c >> (16 * i))) * s;
            const float hi = e4m3_value(static_cast<u8>(c >> (16 * i + 8))) * s;
            w[i] = pack_bf16x2(lo, hi);
        }
        *reinterpret_cast<uint4*>(values + static_cast<long long>(r) * k + b * kGran + sub * 8) =
            make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// A 128x128-block FP8 plane under f32 block factors (DeepSeek-V3's and
// GLM-5.3's release: `w = code * factor`) requantized in place to ue8m0
// exponents, as vLLM's DeepGEMM path does on Blackwell: each block's factor
// rounded up to its power of two `2^e`, each code re-rounded to
// `e4m3(code * factor / 2^e)` (never past e4m3's range, the ratio at most
// one), the exponent `e + 127` landing in `exps` (`[rows / 128][cols / 128]`,
// the factors' shape). A block a CTA; a factor not above zero leaves its
// codes and lands exponent 0.
__global__ void fp8_tile_requant(
    u8* __restrict__ codes,
    const float* __restrict__ factors,
    u8* __restrict__ exps,
    int rows,
    int cols,
    int per)
{
    const int tiles_k = (cols + 127) / 128;
    const int rb = blockIdx.y;
    const int cb = blockIdx.x;
    const int r1 = min(rows, (rb + 1) * 128);
    const int c0 = cb * 128;
    const int c1 = min(cols, c0 + 128);
    // Each row its factor's (`per` rows a factor: 128 or 1).
    for (int r = rb * 128 + static_cast<int>(threadIdx.y); r < r1; r += blockDim.y) {
        const long long at = static_cast<long long>(r / per) * tiles_k + cb;
        const float f = factors[at];
        int e = 0;
        float ratio = 1.f;
        if (f > 0.f) {
            int ex;
            const float m = frexpf(f, &ex);  // f = m * 2^ex, m in [0.5, 1)
            e = m == 0.5f ? ex - 1 : ex;     // 2^e the least power of two >= f
            ratio = f / ldexpf(1.f, e);
        }
        if (threadIdx.x == 0 && r % per == 0) {
            exps[at] = f > 0.f ? static_cast<u8>(min(max(e + 127, 0), 254)) : static_cast<u8>(0);
        }
        if (!(f > 0.f) || ratio == 1.f) continue;
        u8* row = codes + static_cast<long long>(r) * cols;
        for (int c = c0 + static_cast<int>(threadIdx.x); c < c1; c += blockDim.x) {
            const float v = e4m3_value(row[c]) * ratio;
            row[c] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// A row-major plane of one-byte scales, `cols` a row, into the atom layout;
// atom row `r` reads plane row `r / div` (a 128-row block's one scale
// spread to each of its rows).
__global__ void scales_to_atoms(
    const u8* __restrict__ plane,
    u8* __restrict__ atoms,
    int rows,
    int cols,
    int div)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(rows) * cols) return;
    const int r = static_cast<int>(at / cols);
    const int c = static_cast<int>(at % cols);
    atoms[atom_offset(r, c, cols)] = plane[static_cast<long long>(r / div) * cols + c];
}

// Routes laid out by expert for the grouped GEMM: each routed expert's
// count, its rows' start (`align` rows apart, rounded up, for the contiguous
// layout, `m_max` apart for the masked one) and each route's row.
// Under a live window (`win`: `win[0]` tokens from token `win[1]`, `top`
// routes a token) only its routes are laid out; the rest are neither read
// into the GEMM nor written back.
__device__ __forceinline__ bool route_live(int r, int top, const u32* __restrict__ win)
{
    if (win == nullptr) return true;
    const int t = r / top - static_cast<int>(win[1]);
    return t >= 0 && t < static_cast<int>(win[0]);
}

// Routes counted and placed a block at a time: the block
// tallies its routes' experts in shared memory and adds each expert's tally
// to the global count (or takes its rows from the cursor) once, not once a
// route. A route's row within its expert is its rank in the block's tally
// past the block's take: the same rows, the same counts, in another order
// within an expert. `tally` is `2 * experts` ints of dynamic shared memory.
constexpr int kTallyRoutes = 2048;

__global__ void route_count_tally(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    int experts,
    i32* __restrict__ counts)
{
    extern __shared__ int tally[];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) tally[e] = 0;
    __syncthreads();
    const int r0 = blockIdx.x * kTallyRoutes;
    const int r1 = min(r0 + kTallyRoutes, count);
    for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
        const int e = routes[r];
        if (e >= 0 && e < experts && route_live(r, top, win)) atomicAdd(&tally[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        if (tally[e] != 0) atomicAdd(&counts[e], tally[e]);
}

__global__ void route_place_tally(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    int experts,
    i32* __restrict__ cursor,
    i32* __restrict__ pos)
{
    extern __shared__ int tally[];
    int* base = tally + experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) tally[e] = 0;
    __syncthreads();
    const int r0 = blockIdx.x * kTallyRoutes;
    const int r1 = min(r0 + kTallyRoutes, count);
    constexpr int kPer = kTallyRoutes / 256;
    int rank[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int r = r0 + threadIdx.x + i * 256;
        rank[i] = -1;
        if (r < r1) {
            const int e = routes[r];
            if (e >= 0 && e < experts && route_live(r, top, win)) rank[i] = atomicAdd(&tally[e], 1);
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        base[e] = tally[e] != 0 ? atomicAdd(&cursor[e], tally[e]) : 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int r = r0 + threadIdx.x + i * 256;
        if (r < r1) pos[r] = rank[i] >= 0 ? base[routes[r]] + rank[i] : -1;
    }
}

// The experts' starts across a block: each thread sums a run of consecutive
// experts' rows, the block scans the runs, and each thread writes its run's
// starts. The same integers, without one thread walking every expert.
constexpr int kOffsetThreads = 1024;

// `route_count_tally`, `route_offsets_scan` and `route_place_tally` in one
// block, for a fire of at most `kOffsetThreads * kSmallPer` routes: the
// counts tallied in shared memory, scanned into the experts' starts, each
// route placed at its start past its rank. `shared` is `2 * experts` ints.
constexpr int kSmallPer = 16;

__global__ void __launch_bounds__(kOffsetThreads) route_lay_out_small(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    int experts,
    int m_max,
    int align,
    i32* __restrict__ counts,
    i32* __restrict__ offsets,
    i32* __restrict__ pos,
    const float* __restrict__ input_scales,
    const float* __restrict__ tensor_scales,
    float* __restrict__ alpha,
    int alpha_rows,
    const float* __restrict__ input_scales2,
    const float* __restrict__ tensor_scales2,
    float* __restrict__ alpha2,
    int alpha_rows2)
{
    extern __shared__ int lay[];
    int* tally = lay;
    int* start = lay + experts;
    __shared__ int run_sum[kOffsetThreads];
    const int tid = static_cast<int>(threadIdx.x);
    for (int e = tid; e < experts; e += kOffsetThreads) tally[e] = 0;
    __syncthreads();
    int rank[kSmallPer];
#pragma unroll
    for (int i = 0; i < kSmallPer; ++i) {
        const int r = tid + i * kOffsetThreads;
        rank[i] = -1;
        if (r < count) {
            const int e = routes[r];
            if (e >= 0 && e < experts && route_live(r, top, win)) rank[i] = atomicAdd(&tally[e], 1);
        }
    }
    __syncthreads();
    const int per = (experts + kOffsetThreads - 1) / kOffsetThreads;
    const int lo = min(tid * per, experts), hi = min(lo + per, experts);
    auto rows = [&](int e) { return m_max > 0 ? m_max : (tally[e] + align - 1) / align * align; };
    int sum = 0;
    for (int e = lo; e < hi; ++e) sum += rows(e);
    // The runs scanned a warp at a time by shuffles, then the warps' totals
    // by the first warp: two barriers, not two a doubling step.
    const int lane = tid & 31, warp = tid >> 5;
    int incl = sum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += y;
    }
    if (lane == 31) run_sum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = run_sum[lane];
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += y;
        }
        run_sum[lane] = w;
    }
    __syncthreads();
    int at = (warp > 0 ? run_sum[warp - 1] : 0) + incl - sum;
    for (int e = lo; e < hi; ++e) {
        start[e] = at;
        offsets[e] = at;
        counts[e] = tally[e];
        at += rows(e);
    }
    if (tid == kOffsetThreads - 1) offsets[experts] = run_sum[31];
    // NVFP4's per-expert product scale (`expert_alpha`), of each leg asked.
    if (alpha != nullptr) {
        for (int e = tid; e < experts; e += kOffsetThreads)
            alpha[e] = input_scales[e] * tensor_scales[static_cast<long long>(e) * alpha_rows];
    }
    if (alpha2 != nullptr) {
        for (int e = tid; e < experts; e += kOffsetThreads)
            alpha2[e] = input_scales2[e] * tensor_scales2[static_cast<long long>(e) * alpha_rows2];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kSmallPer; ++i) {
        const int r = tid + i * kOffsetThreads;
        if (r < count) pos[r] = rank[i] >= 0 ? start[routes[r]] + rank[i] : -1;
    }
}

__global__ void __launch_bounds__(kOffsetThreads) route_offsets_scan(
    const i32* __restrict__ counts,
    int experts,
    int m_max,
    int align,
    i32* __restrict__ offsets,
    i32* __restrict__ cursor)
{
    __shared__ int run_sum[kOffsetThreads];
    const int tid = static_cast<int>(threadIdx.x);
    const int per = (experts + kOffsetThreads - 1) / kOffsetThreads;
    const int lo = min(tid * per, experts), hi = min(lo + per, experts);
    auto rows = [&](int e) {
        return m_max > 0 ? m_max : (counts[e] + align - 1) / align * align;
    };
    int sum = 0;
    for (int e = lo; e < hi; ++e) sum += rows(e);
    run_sum[tid] = sum;
    __syncthreads();
    // Inclusive scan over the runs.
    for (int step = 1; step < kOffsetThreads; step <<= 1) {
        const int add = tid >= step ? run_sum[tid - step] : 0;
        __syncthreads();
        run_sum[tid] += add;
        __syncthreads();
    }
    int at = run_sum[tid] - sum;
    for (int e = lo; e < hi; ++e) {
        offsets[e] = at;
        cursor[e] = at;
        at += rows(e);
    }
    if (tid == kOffsetThreads - 1) offsets[experts] = run_sum[tid];
}

// `y[r] = grouped[pos[r]]`, a dropped route zeros, a route past the live
// window untouched; a thread a pair.
__global__ void route_gather(
    const bf16* __restrict__ grouped,
    const i32* __restrict__ pos,
    bf16* __restrict__ y,
    int count,
    int n,
    int top,
    const u32* __restrict__ win)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(count) * (n / 2)) return;
    const int r = static_cast<int>(at / (n / 2));
    if (!route_live(r, top, win)) return;
    const int c = static_cast<int>(at % (n / 2)) * 2;
    const int o = pos[r];
    bf16x2* dst = reinterpret_cast<bf16x2*>(y + static_cast<long long>(r) * n + c);
    if (o < 0) {
        *dst = f32_to_bf16x2(0.f, 0.f);
        return;
    }
    *dst = *reinterpret_cast<const bf16x2*>(grouped + static_cast<long long>(o) * n + c);
}

// `route_gather`, 16 bytes a thread: the same rows, written whole.
__global__ void route_gather_vec8(
    const bf16* __restrict__ grouped,
    const i32* __restrict__ pos,
    bf16* __restrict__ y,
    int count,
    int n,
    int top,
    const u32* __restrict__ win)
{
    const int runs = n / 8;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(count) * runs) return;
    const int r = static_cast<int>(at / runs);
    if (!route_live(r, top, win)) return;
    const int c = static_cast<int>(at % runs) * 8;
    const int o = pos[r];
    uint4* dst = reinterpret_cast<uint4*>(y + static_cast<long long>(r) * n + c);
    if (o < 0) {
        *dst = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    *dst = *reinterpret_cast<const uint4*>(grouped + static_cast<long long>(o) * n + c);
}

// The block's sum of `v`, as norm_gate's block folds it (warps, then the
// warps' sums in the first); `part` is reusable on return.
__device__ __forceinline__ float route_row_sum(float v, float* part)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = lane < static_cast<int>(blockDim.x >> 5) ? part[lane] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) w += __shfl_xor_sync(0xffffffffu, w, off);
        if (lane == 0) part[32] = w;
    }
    __syncthreads();
    const float total = part[32];
    __syncthreads();
    return total;
}

// `moe_weighted_sum_vec8` of the grouped product read where it lies: token
// t's eight columns from `grouped[pos[t * top + j]]`, weighted, summed in f32
// in `j` order; a dropped route adds nothing. A token past the live window
// (`win`, in tokens) is untouched. With a `scale`, a block holds the row
// (`n / 8` threads) and each route is first normed as norm_gate norms it:
// `u = x * scale` rounded, `u / rms(u)` rounded.
__global__ void route_weighted_gather_vec8(
    const bf16* __restrict__ grouped,
    const i32* __restrict__ pos,
    const float* __restrict__ weights,
    const bf16* __restrict__ shared,
    const bf16* __restrict__ gate,
    int gate_pitch,
    bf16* __restrict__ y,
    int n,
    int top,
    const bf16* __restrict__ scale,
    float eps,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int plane = win != nullptr ? t + static_cast<int>(win[1]) : t;
    const int h0 = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (h0 >= n) return;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    __shared__ float part[33];
    uint4 sv = {};
    if (scale != nullptr) sv = *reinterpret_cast<const uint4*>(scale + h0);
    const bf16* s8 = reinterpret_cast<const bf16*>(&sv);
    const long long base = static_cast<long long>(plane) * top;
    for (int j = 0; j < top; ++j) {
        const int o = pos[base + j];
        if (o < 0) continue;
        const uint4 w = *reinterpret_cast<const uint4*>(grouped + static_cast<long long>(o) * n + h0);
        const bf16* v8 = reinterpret_cast<const bf16*>(&w);
        float u[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) u[i] = bf16_to_f32(v8[i]);
        if (scale != nullptr) {
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                u[i] = bf16_to_f32(f32_to_bf16(u[i] * bf16_to_f32(s8[i])));
                ss = fmaf(u[i], u[i], ss);
            }
            const float inv = rsqrtf(route_row_sum(ss, part) / static_cast<float>(n) + eps);
#pragma unroll
            for (int i = 0; i < 8; ++i) u[i] = bf16_to_f32(f32_to_bf16(u[i] * inv));
        }
        // No weights: the routes already carry theirs (DeepSeek's, before
        // the down leg), and moe_sum_add sums them as they are.
        const float wr = weights != nullptr ? weights[base + j] : 1.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += wr * u[i];
    }
    // Qwen's shared expert under its sigmoid gate, as moe_sigmoid_gate_add
    // adds it; with no gate, the shared row added as it is (moe_sum_add).
    if (shared != nullptr) {
        float s = 1.f;
        if (gate != nullptr) {
            const float gv = bf16_to_f32(gate[static_cast<long long>(plane) * gate_pitch]);
            s = 1.f / (1.f + __expf(-gv));
        }
        const uint4 sv = *reinterpret_cast<const uint4*>(shared + static_cast<long long>(plane) * n + h0);
        const bf16* s8 = reinterpret_cast<const bf16*>(&sv);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += bf16_to_f32(s8[i]) * s;
    }
    uint4 out;
    bf16* o8 = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int i = 0; i < 8; ++i) o8[i] = f32_to_bf16(acc[i]);
    *reinterpret_cast<uint4*>(y + static_cast<long long>(plane) * n + h0) = out;
}

// `out[pos[r]] = v[r]`: a route-order f32 a route landed at its grouped row
// (a dropped route nowhere).
__global__ void route_scatter_f32(
    const float* __restrict__ v,
    const i32* __restrict__ pos,
    float* __restrict__ out,
    int count)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    const int o = pos[r];
    if (o >= 0) out[o] = v[r];
}

// `grouped[pos[r]] = x[r / div]`, 16 bytes a thread; a dropped route lands
// nowhere.
__global__ void route_scatter_rows(
    const bf16* __restrict__ x,
    const i32* __restrict__ pos,
    bf16* __restrict__ grouped,
    int count,
    int k,
    int div)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int runs = k / 8;
    if (at >= static_cast<long long>(count) * runs) return;
    const int r = static_cast<int>(at / runs);
    const int c = static_cast<int>(at % runs) * 8;
    const int o = pos[r];
    if (o < 0) return;
    *reinterpret_cast<uint4*>(grouped + static_cast<long long>(o) * k + c) =
        *reinterpret_cast<const uint4*>(x + static_cast<long long>(r / div) * k + c);
}

// Each expert's product scale: its activations' static global times its
// weight's tensor scale, the first row's standing for the expert's (vLLM
// reads the gate leg's for both of a gated bank's legs).
__global__ void expert_alpha(
    const float* __restrict__ input_scales,
    const float* __restrict__ tensor_scales,
    float* __restrict__ alpha,
    int experts,
    int rows)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts) return;
    alpha[e] = input_scales[e] * tensor_scales[static_cast<long long>(e) * rows];
}

// `mlp_swiglu_clamp_routed` landing its rows as the e4m3 read-back
// (`act_quant_e4m3_v8`'s values mode) of the bf16 rows it would land: eight
// columns a lane, a 128-wide block's 16 lanes adjacent. A select reading
// them back again lands the same values (each block's values sit on its
// scale's grid, or on the grid one below where the rounding lowered its
// amax), so it may read them as they are. `width` whole blocks.
__global__ void mlp_swiglu_clamp_routed_e4m3(
    const bf16* __restrict__ packed,
    const float* __restrict__ weights,
    bf16* __restrict__ y,
    i32 lanes,
    i32 width,
    float limit,
    i32 fan,
    const u32* __restrict__ win)
{
    using Block = E4m3Block<128>;
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = static_cast<int>(threadIdx.x % 32);
    const long long per_row = width / 8;
    long long live_lanes = lanes;
    long long first = 0;
    if (win != nullptr) {
        live_lanes = static_cast<long long>(win[0]) * fan * per_row;
        first = static_cast<long long>(win[1]) * fan * per_row;
    }
    // A lane past the last block keeps its place in the shuffle.
    const bool live = idx < lanes && idx < live_lanes;
    const long long at = (live ? idx : 0) + first;
    const long long row = at / per_row;
    const int col = static_cast<int>(at - row * per_row) * 8;
    const bf16* src = packed + row * 2 * width + col;
    const uint4 gw = *reinterpret_cast<const uint4*>(src);
    const uint4 uw = *reinterpret_cast<const uint4*>(src + width);
    const unsigned gs[4] = {gw.x, gw.y, gw.z, gw.w};
    const unsigned us[4] = {uw.x, uw.y, uw.z, uw.w};
    const float wt = weights[row];
    float v[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned gb = h == 0 ? gs[i] << 16 : gs[i] & 0xFFFF0000u;
            const unsigned ub = h == 0 ? us[i] << 16 : us[i] & 0xFFFF0000u;
            const float g = fminf(__uint_as_float(gb), limit);
            const float u = fminf(fmaxf(__uint_as_float(ub), -limit), limit);
            v[2 * i + h] = (g / (1.f + expf(-g))) * u * wt;
        }
    }
    Block q;
    q.hold(v);
    if (!live) return;
    const float s = __uint_as_float(static_cast<unsigned>(q.e + 127) << 23);
    const unsigned long long c = q.codes();
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = e4m3_value(static_cast<u8>(c >> (16 * i))) * s;
        const float hi = e4m3_value(static_cast<u8>(c >> (16 * i + 8))) * s;
        w[i] = pack_bf16x2(lo, hi);
    }
    *reinterpret_cast<uint4*>(y + row * width + col) = make_uint4(w[0], w[1], w[2], w[3]);
}

}  // namespace pie::linear
