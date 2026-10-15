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

// One thread a 16-wide block.
template <int kMode>
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

// One warp a `kGran`-wide block, each lane `kGran / 32` elements.
template <int kMode, int kGran>
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
    const bf16* src = x + static_cast<long long>(r / div) * k + b * kGran + lane * kEach;
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
    amax = fmaxf(amax, 1e-4f);
    // 2^ceil(log2(amax / 448)), from the bits of the quotient.
    const unsigned bits = __float_as_uint(amax * (1.f / 448.f));
    const int e = static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    const float s = __uint_as_float(static_cast<unsigned>(e + 127) << 23);
    if constexpr (kMode == kQuantCodes) {
        const int o = pos != nullptr ? pos[r] : r;
        if (o < 0) return;
        u8* dst = codes + static_cast<long long>(o) * k + b * kGran + lane * kEach;
#pragma unroll
        for (int i = 0; i < kEach; ++i) {
            dst[i] = __nv_cvt_float_to_fp8(fminf(fmaxf(v[i] / s, -448.f), 448.f),
                                           __NV_SATFINITE, __NV_E4M3);
        }
        if (lane == 0) atoms[atom_offset(atom_row(o, m_max), b, blocks)] = static_cast<u8>(e + 127);
    } else {
        bf16* dst = values + static_cast<long long>(r) * k + b * kGran + lane * kEach;
#pragma unroll
        for (int i = 0; i < kEach; ++i) {
            const u8 q = __nv_cvt_float_to_fp8(fminf(fmaxf(v[i] / s, -448.f), 448.f),
                                               __NV_SATFINITE, __NV_E4M3);
            dst[i] = f32_to_bf16(e4m3_value(q) * s);
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

__global__ void route_count(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    i32* __restrict__ counts)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < count && routes[r] >= 0 && route_live(r, top, win)) atomicAdd(&counts[routes[r]], 1);
}

// `route_count` and `route_place` a block of routes at a time: the block
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

__global__ void route_offsets(
    const i32* __restrict__ counts,
    int experts,
    int m_max,
    int align,
    i32* __restrict__ offsets,
    i32* __restrict__ cursor)
{
    if (threadIdx.x != 0) return;
    int at = 0;
    for (int e = 0; e < experts; ++e) {
        offsets[e] = at;
        cursor[e] = at;
        at += m_max > 0 ? m_max : (counts[e] + align - 1) / align * align;
    }
    offsets[experts] = at;
}

// `route_offsets` across a block: each thread sums a run of consecutive
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
    int alpha_rows)
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
    // NVFP4's per-expert product scale (`expert_alpha`), where asked.
    if (alpha != nullptr) {
        for (int e = tid; e < experts; e += kOffsetThreads)
            alpha[e] = input_scales[e] * tensor_scales[static_cast<long long>(e) * alpha_rows];
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

__global__ void route_place(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    i32* __restrict__ cursor,
    i32* __restrict__ pos)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    pos[r] = routes[r] >= 0 && route_live(r, top, win) ? atomicAdd(&cursor[routes[r]], 1) : -1;
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

// `moe_weighted_sum_vec8` of the grouped product read where it lies: token
// t's eight columns from `grouped[pos[t * top + j]]`, weighted, summed in f32
// in `j` order; a dropped route adds nothing. A token past the live window
// (`win`, in tokens) is untouched.
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
    const long long base = static_cast<long long>(plane) * top;
    for (int j = 0; j < top; ++j) {
        const int o = pos[base + j];
        if (o < 0) continue;
        const uint4 w = *reinterpret_cast<const uint4*>(grouped + static_cast<long long>(o) * n + h0);
        const bf16* v8 = reinterpret_cast<const bf16*>(&w);
        const float wr = weights[base + j];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += wr * bf16_to_f32(v8[i]);
    }
    // Qwen's shared expert under its sigmoid gate, as moe_sigmoid_gate_add
    // adds it.
    if (shared != nullptr) {
        const float gv = bf16_to_f32(gate[static_cast<long long>(plane) * gate_pitch]);
        const float s = 1.f / (1.f + __expf(-gv));
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

}  // namespace pie::linear
