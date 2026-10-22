#pragma once

#include "prelude/device.cuh"

#ifdef __CUDACC_RTC__
#include "prelude/half2.cuh"
#include "prelude/fp8.cuh"
#include "prelude/mma.cuh"
#else
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#endif

namespace pie::linear {

constexpr int kBlock = 256;

struct fp8_e4m3 {
    using store = u8;

    static __host__ __device__ __forceinline__ float max_abs() { return 448.f; }
    static __device__ __forceinline__ store narrow(float v) {
        return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
};

struct int8_sym {
    using store = i8;
    static __host__ __device__ __forceinline__ float max_abs() { return 127.f; }
    static __device__ __forceinline__ store narrow(float v) {
        int q = static_cast<int>(rintf(v));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        return static_cast<store>(q);
    }
};

__device__ __forceinline__ float row_absmax(float local, float* smem, int tid) {
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    const int lane = tid & 31;
    const int warp = tid / 32;
    if (lane == 0) smem[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = (tid < kBlock / 32) ? smem[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) smem[0] = local;
    }
    __syncthreads();
    return smem[0];
}

__global__ void quant_act_fp8_per_group(
    const bf16* __restrict__ act,
    u8* __restrict__ out,
    float* __restrict__ scale_out,
    i32 m,
    i32 k,
    i32 gs,
    i32 n_groups) {
    const i32 row = blockIdx.y;
    const i32 g = blockIdx.x;
    if (row >= m || g >= n_groups) return;

    const i32 base = g * gs;
    const i32 remaining = k - base;
    const i32 count = (gs < remaining) ? gs : remaining;
    const usize off = static_cast<usize>(row) * k + base;

    float amax = 0.f;
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(bf16_to_f32(act[off + i])));
    }
    __shared__ float warp_max[128 / 32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    }
    if (lane == 0) warp_max[warp] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = warp_max[0];
        for (unsigned w = 1; w < blockDim.x / 32; ++w) v = fmaxf(v, warp_max[w]);
        warp_max[0] = v;
    }
    __syncthreads();
    amax = warp_max[0];

    const float scale = (amax > 0.f) ? (amax / fp8_e4m3::max_abs()) : 1.f;
    const float scale_rcp = (amax > 0.f) ? (fp8_e4m3::max_abs() / amax) : 0.f;
    if (threadIdx.x == 0) {
        scale_out[static_cast<usize>(row) * n_groups + g] = scale;
    }
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        out[off + i] = fp8_e4m3::narrow(bf16_to_f32(act[off + i]) * scale_rcp);
    }
}





__device__ __constant__ float kFp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

__device__ __forceinline__ void mxfp4_unpack8(unsigned word, __half2 out[4]) {
    constexpr unsigned kMagHi01234567 = 0x3E3C3800u;
    constexpr unsigned kMagHi4567     = 0x46444240u;
    constexpr unsigned kSignBytes     = 0x80808080u;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const unsigned sel = (word >> (half * 16)) & 0xFFFFu;
        const unsigned mag =
            __byte_perm(kMagHi01234567, kMagHi4567, sel & 0x7777u);
        const unsigned sgn =
            __byte_perm(0u, kSignBytes, (sel & 0x8888u) >> 1);
        const unsigned hi = mag | sgn;
        const unsigned a = __byte_perm(hi, 0u, 0x1404u);
        const unsigned b = __byte_perm(hi, 0u, 0x3424u);
        out[half * 2 + 0] = *reinterpret_cast<const __half2*>(&a);
        out[half * 2 + 1] = *reinterpret_cast<const __half2*>(&b);
    }
}

__device__ __forceinline__ float mxfp4_block_scale(u8 b) {
    return b == 0 ? exp2f(-127.f)
                  : __int_as_float(static_cast<int>(b) << 23);
}

struct alignas(16) MoeGroupBases {
    const u8* codes;
    const u8* scales;

    const u8* biases;
    const u8* pad;
};

__device__ __forceinline__ void moe_note_group(
    unsigned int* __restrict__ group_hits)
{
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) {
        atomicAdd(group_hits, 1u);
    }
}

// Eight e2m1 codes of a word (code `i` its nibble `i`) dotted with `xv`
// onto `acc`, the codes laid into bf16 by integer ops: each 2^-126 times its
// value, exact, so `xv` comes 2^100 up and the caller's scale takes back the
// 2^26 left. No f16 converts.
__device__ __forceinline__ float e2m1x8_dot_lean(unsigned wd, const float (&xv)[8], float acc) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        // Codes 3 - p and 7 - p at bits 12..15 and 28..31.
        const unsigned t = wd << (4 * p);
        const unsigned b2 = (t & 0x80008000u) | ((t >> 6) & 0x01C001C0u);
        acc = fmaf(__uint_as_float(b2 << 16), xv[3 - p], acc);
        acc = fmaf(__uint_as_float(b2 & 0xFFFF0000u), xv[7 - p], acc);
    }
    return acc;
}

// With kLean (A100's profile) one 32-code block a row in flight a lane, the
// codes laid into bf16 by integer ops (2^-126 times their value, exact) and
// the activations taken 2^100 up so the products stay normal, the block
// scale taking back the 2^26 left: no f16 converts.
template <class T, int kRowsT, bool kLean = false>
__global__ void moe_matmul_select_mxfp4(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int top_k,
    int act_div,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    const int route = blockIdx.x;

    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;

    const int plane_route = win != nullptr ? route + static_cast<int>(win[1]) * top_k : route;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes[plane_route];

    moe_note_group(group_hits);

    const int groups_per_row = k / 32;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
    }

    const u8* w = codes_at + static_cast<long long>(expert) * n * (k / 2);
    const u8* s = scales_at + static_cast<long long>(expert) * n * groups_per_row;

    const T* x = act + static_cast<long long>(plane_route / act_div) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    // One 16-byte load is a whole 32-code block of a row; a lane issues every
    // row's block for kU blocks before it computes, so the weights stream.
    constexpr int kU = kLean ? 1 : 2;
    const uint4* wq = reinterpret_cast<const uint4*>(w);
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g0 = lane_id; g0 < groups_per_row; g0 += 32 * kU) {
        uint4 ww[kU][kRows];
        u8 sc[kU][kRows];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int g = min(g0 + 32 * u, groups_per_row - 1);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const long long at = static_cast<long long>(row_of[r]) * groups_per_row + g;
                ww[u][r] = wq[at];
                sc[u][r] = s[at];
            }
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int g = g0 + 32 * u;
            if (g >= groups_per_row) break;
            float part[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[r] = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xq = x4[g * 4 + q];
                const T* xe = reinterpret_cast<const T*>(&xq);
                float xv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) xv[j] = Elem<T>::to_f32(xe[j]) * (kLean ? 0x1p100f : 1.f);
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if constexpr (kLean) {
                        part[r] = e2m1x8_dot_lean((&ww[u][r].x)[q], xv, part[r]);
                    } else {
                        __half2 qd[4];
                        mxfp4_unpack8((&ww[u][r].x)[q], qd);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f = __half22float2(qd[j]);
                            part[r] = fmaf(f.x, xv[2 * j], part[r]);
                            part[r] = fmaf(f.y, xv[2 * j + 1], part[r]);
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                acc[r] = fmaf(part[r],
                              kLean ? __uint_as_float((static_cast<unsigned>(sc[u][r]) + 26u) << 23)
                                    : mxfp4_block_scale(sc[u][r]),
                              acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
        const T* b = bias != nullptr
            ? bias + static_cast<long long>(expert) * n
            : nullptr;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row >= n) break;
            float v = acc[r];
            if (b != nullptr) v += Elem<T>::to_f32(b[row]);
            out[static_cast<long long>(plane_route) * n + row] = Elem<T>::from_f32(v);
        }
    }
}

__global__ void moe_route_order(
    const i32* __restrict__ routes,
    i32* __restrict__ order,
    i32* __restrict__ offsets,
    i32* __restrict__ work,
    int work_cap,
    int group_routes,
    int top_k,
    int route_count,
    int num_experts,
    const u32* __restrict__ win)
{
    extern __shared__ i32 order_smem[];
    i32* counts = order_smem;
    i32* fill = counts + num_experts + 1;
    const int live = win != nullptr
        ? min(static_cast<int>(win[0]) * top_k, route_count)
        : route_count;
    const int base = win != nullptr ? static_cast<int>(win[1]) * top_k : 0;
    for (int e = threadIdx.x; e <= num_experts; e += blockDim.x) {
        counts[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < live; r += blockDim.x) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        atomicAdd(counts + e, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int running = 0;
        for (int e = 0; e <= num_experts; ++e) {
            const int c = counts[e];
            counts[e] = running;
            running += c;
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e <= num_experts; e += blockDim.x) offsets[e] = counts[e];
    for (int r = threadIdx.x; r < live; r += blockDim.x) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        const int pos = counts[e] + atomicAdd(fill + e, 1);
        order[pos] = base + r;
    }

    if (threadIdx.x == 0) {
        int w = 0;
        for (int e = 0; e < num_experts && w < work_cap; ++e) {
            const int c = counts[e + 1] - counts[e];
            for (int g = 0; g * group_routes < c && w < work_cap; ++g) work[w++] = e * 65536 + g;
        }
        offsets[num_experts + 1] = w;
    }
}

template <class T, int kBits, int kGroup>
__global__ void moe_matmul_select_mlxu4_grouped(
    const T* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{

    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kBatch = 16;
    constexpr int kSlots = 4;
    constexpr int kRowsPerThread = 2;
    constexpr int kRoutesPerThread = kBatch / kSlots;
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;
    constexpr int kWordsPerTileK = kTileK / kPerWord;
    static_assert(kTileK % kGroup == 0 || kGroup % kTileK == 0, "tile and group align");
    static_assert(kTileN / kRowsPerThread * kSlots == 256, "one thread per (row pair, slot)");
    const int expert = blockIdx.x;
    if (expert >= num_experts) return;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    if (begin >= end) return;

    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int groups_per_row = k / kGroup;
    const int words_per_row = k / kPerWord;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }
    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * words_per_row * 4);
    const bf16* s16 = reinterpret_cast<const bf16*>(
        scales_at + static_cast<long long>(expert) * n * groups_per_row * 2);
    const bf16* b16 = reinterpret_cast<const bf16*>(
        biases_at + static_cast<long long>(expert) * n * groups_per_row * 2);

    __shared__ float w_tile[kTileN][kTileK + 1];
    __shared__ float x_tile[kBatch][kTileK];
    const int tid = threadIdx.x;
    const int pair = tid % (kTileN / kRowsPerThread);
    const int slot = tid / (kTileN / kRowsPerThread);
    const int row_a = pair * kRowsPerThread;
    const int row_b = row_a + 1;

    for (int b0 = begin; b0 < end; b0 += kBatch) {
        const int batch = min(kBatch, end - b0);
        float acc[kRowsPerThread][kRoutesPerThread];
#pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i)
#pragma unroll
            for (int j = 0; j < kRoutesPerThread; ++j) acc[i][j] = 0.f;
        for (int k0 = 0; k0 < k; k0 += kTileK) {

            for (int idx = tid; idx < kTileN * kWordsPerTileK; idx += blockDim.x) {
                const int r = idx / kWordsPerTileK;
                const int wq = idx % kWordsPerTileK;
                const int kk = k0 + wq * kPerWord;
                if (kk < k) {
                    const int rr = min(row0 + r, n - 1);
                    const int g = kk / kGroup;
                    const long long fx = static_cast<long long>(rr) * groups_per_row + g;
                    const float sv = Elem<bf16>::to_f32(s16[fx]);
                    const float bv = Elem<bf16>::to_f32(b16[fx]);
                    const unsigned word =
                        w32[static_cast<long long>(rr) * words_per_row + kk / kPerWord];
#pragma unroll
                    for (int j = 0; j < kPerWord; ++j) {
                        const float code = static_cast<float>((word >> (kBits * j)) & kMask);
                        w_tile[r][wq * kPerWord + j] = fmaf(code, sv, bv);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < kPerWord; ++j) w_tile[r][wq * kPerWord + j] = 0.f;
                }
            }

            for (int idx = tid; idx < batch * kTileK; idx += blockDim.x) {
                const int t = idx / kTileK;
                const int kk = idx % kTileK;
                const int plane_route = order[b0 + t];
                const T* x = act + static_cast<long long>(plane_route / act_div) * k;
                x_tile[t][kk] = k0 + kk < k ? Elem<T>::to_f32(x[k0 + kk]) : 0.f;
            }
            __syncthreads();

#pragma unroll 4
            for (int kk = 0; kk < kTileK; ++kk) {
                const float wa = w_tile[row_a][kk];
                const float wb = w_tile[row_b][kk];
#pragma unroll
                for (int j = 0; j < kRoutesPerThread; ++j) {
                    const float xv = x_tile[slot + kSlots * j][kk];
                    acc[0][j] = fmaf(wa, xv, acc[0][j]);
                    acc[1][j] = fmaf(wb, xv, acc[1][j]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < kRoutesPerThread; ++j) {
            const int t = slot + kSlots * j;
            if (t < batch) {
                const int plane_route = order[b0 + t];
                T* o = out + static_cast<long long>(plane_route) * n + row0;
                if (row0 + row_a < n) o[row_a] = Elem<T>::from_f32(acc[0][j]);
                if (row0 + row_b < n) o[row_b] = Elem<T>::from_f32(acc[1][j]);
            }
        }
    }
}

template <class T>
__global__ void moe_matmul_select_mxfp4_grouped(
    const T* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{
    constexpr int kTileN = 128;

    constexpr int kTileK = 64;
    constexpr int kBatch = 16;
    constexpr int kSlots = 4;
    constexpr int kRowsPerThread = 2;
    constexpr int kRoutesPerThread = kBatch / kSlots;

    constexpr int kPerWord = 8;
    constexpr int kBlock = 32;
    constexpr int kWordsPerTileK = kTileK / kPerWord;
    static_assert(kTileK % kBlock == 0, "a K tile is whole mxfp4 blocks");
    static_assert(kTileN / kRowsPerThread * kSlots == 256, "one thread per (row pair, slot)");

    const int expert = blockIdx.x;
    if (expert >= num_experts) return;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    if (begin >= end) return;

    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int groups_per_row = k / kBlock;
    const int words_per_row = k / kPerWord;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
    }
    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const u8* s8 = scales_at + static_cast<long long>(expert) * n * groups_per_row;

    __shared__ float w_tile[kTileN][kTileK + 1];
    __shared__ float x_tile[kBatch][kTileK];
    const int tid = threadIdx.x;
    const int pair = tid % (kTileN / kRowsPerThread);
    const int slot = tid / (kTileN / kRowsPerThread);
    const int row_a = pair * kRowsPerThread;
    const int row_b = row_a + 1;

    const T* b = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    float bias_a = 0.f;
    float bias_b = 0.f;
    if (b != nullptr) {
        if (row0 + row_a < n) bias_a = Elem<T>::to_f32(b[row0 + row_a]);
        if (row0 + row_b < n) bias_b = Elem<T>::to_f32(b[row0 + row_b]);
    }

    for (int b0 = begin; b0 < end; b0 += kBatch) {
        const int batch = min(kBatch, end - b0);
        float acc[kRowsPerThread][kRoutesPerThread];
#pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i)
#pragma unroll
            for (int j = 0; j < kRoutesPerThread; ++j) acc[i][j] = 0.f;
        for (int k0 = 0; k0 < k; k0 += kTileK) {

            for (int idx = tid; idx < kTileN * kWordsPerTileK; idx += blockDim.x) {
                const int r = idx / kWordsPerTileK;
                const int wq = idx % kWordsPerTileK;
                const int kk = k0 + wq * kPerWord;
                if (kk < k) {
                    const int rr = min(row0 + r, n - 1);
                    const float sc = mxfp4_block_scale(
                        s8[static_cast<long long>(rr) * groups_per_row + kk / kBlock]);
                    __half2 qd[4];
                    mxfp4_unpack8(
                        w32[static_cast<long long>(rr) * words_per_row + kk / kPerWord], qd);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(qd[j]);
                        w_tile[r][wq * kPerWord + 2 * j] = f.x * sc;
                        w_tile[r][wq * kPerWord + 2 * j + 1] = f.y * sc;
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < kPerWord; ++j) w_tile[r][wq * kPerWord + j] = 0.f;
                }
            }

            for (int idx = tid; idx < batch * kTileK; idx += blockDim.x) {
                const int t = idx / kTileK;
                const int kk = idx % kTileK;
                const int plane_route = order[b0 + t];
                const T* x = act + static_cast<long long>(plane_route / act_div) * k;
                x_tile[t][kk] = k0 + kk < k ? Elem<T>::to_f32(x[k0 + kk]) : 0.f;
            }
            __syncthreads();

#pragma unroll 4
            for (int kk = 0; kk < kTileK; ++kk) {
                const float wa = w_tile[row_a][kk];
                const float wb = w_tile[row_b][kk];
#pragma unroll
                for (int j = 0; j < kRoutesPerThread; ++j) {
                    const float xv = x_tile[slot + kSlots * j][kk];
                    acc[0][j] = fmaf(wa, xv, acc[0][j]);
                    acc[1][j] = fmaf(wb, xv, acc[1][j]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < kRoutesPerThread; ++j) {
            const int t = slot + kSlots * j;
            if (t < batch) {
                const int plane_route = order[b0 + t];
                T* o = out + static_cast<long long>(plane_route) * n + row0;
                if (row0 + row_a < n) o[row_a] = Elem<T>::from_f32(acc[0][j] + bias_a);
                if (row0 + row_b < n) o[row_b] = Elem<T>::from_f32(acc[1][j] + bias_b);
            }
        }
    }
}


__device__ __forceinline__ void pie_ldmatrix_x4(unsigned (&r)[4], const void* row) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}

__device__ __forceinline__ uint4 pie_ld_line(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned short pie_ld_line_u16(const unsigned short* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L2::128B.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ void pie_mma_bf16_16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void moe_matmul_select_mxfp4_wmma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{
    namespace wmma = ::nvcuda::wmma;
    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kBatch = 16;
    constexpr int kBatches = 2;
    constexpr int kRoutes = kBatch * kBatches;
    constexpr int kLd = kTileK + 8;
    constexpr int kWarps = kTileN / 16;

    constexpr int kPerWord = 8;
    constexpr int kGroup = 32;
    static_assert(kRoutes * kLd * 2 >= kWarps * 16 * 16 * 4, "the C tiles fit where the activations were");

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item / 65536;
    const int slice = item % 65536;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert] + slice * kRoutes;
    const int end = min(offsets[expert + 1], begin + kRoutes);
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int groups_per_row = k / kGroup;
    const int words_per_row = k / kPerWord;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
    }
    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const u8* s8 = scales_at + static_cast<long long>(expert) * n * groups_per_row;

    const bf16* b_row =
        bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;

    __shared__ __align__(32) bf16 w_tile[kTileN][kLd];
    __shared__ __align__(32) bf16 x_tile[kRoutes][kLd];
    float* c_tile = reinterpret_cast<float*>(&x_tile[0][0]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const bf16 zero = f32_to_bf16(0.f);

    constexpr int kQuadWords = 4;
    constexpr int kQuadCodes = kQuadWords * kPerWord;
    constexpr int kQuadsPerTileK = kTileK / kQuadCodes;
    constexpr int kQuadsPerThread = kTileN * kQuadsPerTileK / 256;
    constexpr int kVec = 8;
    constexpr int kVecsPerTileK = kTileK / kVec;
    constexpr int kVecsPerThread = kRoutes * kVecsPerTileK / 256;
    static_assert(kGroup % kQuadCodes == 0, "a quad lies within one group");
    static_assert(kTileN * kQuadsPerTileK % 256 == 0 && kRoutes * kVecsPerTileK % 256 == 0, "even split");

    struct Staged {
        uint4 quad[kQuadsPerThread];
        unsigned char scale_byte[kQuadsPerThread];
        bool quad_live[kQuadsPerThread];
        uint4 vec[kVecsPerThread];
    };

    {
        const int g0 = begin;
        const int group = end - g0;
        const int batches = (group + kBatch - 1) / kBatch;

        float acc[kBatches][2][4];
#pragma unroll
        for (int b = 0; b < kBatches; ++b)
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[b][t][e] = 0.f;

        const bf16* vec_src[kVecsPerThread];
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int t = (tid + i * 256) / kVecsPerTileK;
            vec_src[i] = nullptr;
            if (t < group) {
                const int plane_route = order[g0 + t];
                vec_src[i] = act + static_cast<long long>(plane_route / act_div) * k;
            }
        }

        auto fetch = [&](int k0, Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                const int kk = k0 + q * kQuadCodes;
                const int rr = row0 + r;
                st.quad_live[i] = kk + kQuadCodes <= k && rr < n;
                if (st.quad_live[i]) {
                    const long long fx = static_cast<long long>(rr) * groups_per_row + kk / kGroup;
                    st.scale_byte[i] = s8[fx];
                    st.quad[i] = pie_ld_line(reinterpret_cast<const uint4*>(
                        w32 + static_cast<long long>(rr) * words_per_row + kk / kPerWord));
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int v = (tid + i * 256) % kVecsPerTileK;
                const int kk = k0 + v * kVec;
                st.vec[i] = make_uint4(0u, 0u, 0u, 0u);
                if (vec_src[i] != nullptr && kk + kVec <= k) {
                    st.vec[i] = *reinterpret_cast<const uint4*>(vec_src[i] + kk);
                }
            }
        };

        auto store = [&](int k0, const Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                unsigned* dst = reinterpret_cast<unsigned*>(&w_tile[r][q * kQuadCodes]);
                if (st.quad_live[i]) {
                    const float sv = mxfp4_block_scale(st.scale_byte[i]);
                    const unsigned words[kQuadWords] = {st.quad[i].x, st.quad[i].y, st.quad[i].z, st.quad[i].w};
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
                        __half2 qd[4];
                        mxfp4_unpack8(words[w], qd);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f = __half22float2(qd[j]);
                            dst[w * 4 + j] = pack_bf16x2(f.x * sv, f.y * sv);
                        }
                    }
                } else {

                    const int kk = k0 + q * kQuadCodes;
                    const int rr = row0 + r;
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
                        const int kw = kk + w * kPerWord;
                        if (kw < k && rr < n) {
                            const long long fx = static_cast<long long>(rr) * groups_per_row + kw / kGroup;
                            const float sv = mxfp4_block_scale(s8[fx]);
                            __half2 qd[4];
                            mxfp4_unpack8(
                                w32[static_cast<long long>(rr) * words_per_row + kw / kPerWord], qd);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float2 f = __half22float2(qd[j]);
                                w_tile[r][q * kQuadCodes + w * kPerWord + 2 * j] = f32_to_bf16(f.x * sv);
                                w_tile[r][q * kQuadCodes + w * kPerWord + 2 * j + 1] = f32_to_bf16(f.y * sv);
                            }
                        } else {
#pragma unroll
                            for (int j = 0; j < kPerWord; ++j)
                                w_tile[r][q * kQuadCodes + w * kPerWord + j] = zero;
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int t = idx / kVecsPerTileK;
                const int v = idx % kVecsPerTileK;
                *reinterpret_cast<uint4*>(&x_tile[t][v * kVec]) = st.vec[i];
            }

            const int tail = k - k0;
            if (tail < kTileK && (tail % kVec) != 0) {
                const int from = (tail / kVec) * kVec;
                for (int idx = tid; idx < kRoutes * (kTileK - from); idx += blockDim.x) {
                    const int t = idx / (kTileK - from);
                    const int kk = from + idx % (kTileK - from);
                    bf16 vv = zero;
                    if (t < group && k0 + kk < k) {
                        const int plane_route = order[g0 + t];
                        vv = act[static_cast<long long>(plane_route / act_div) * k + k0 + kk];
                    }
                    x_tile[t][kk] = vv;
                }
            }
        };

        Staged staged;
        fetch(0, staged);
        for (int k0 = 0; k0 < k; k0 += kTileK) {
            store(k0, staged);
            __syncthreads();
            if (k0 + kTileK < k) fetch(k0 + kTileK, staged);
#pragma unroll
            for (int kk = 0; kk < kTileK; kk += 16) {

                unsigned bfrag[4];
                pie_ldmatrix_x4(bfrag, &w_tile[warp * 16 + (lane & 7) + ((lane >> 4) << 3)][kk + (((lane >> 3) & 1) << 3)]);

#pragma unroll
                for (int bt = 0; bt < kBatches; ++bt) {

                    unsigned afrag[4];
                    pie_ldmatrix_x4(afrag, &x_tile[bt * kBatch + (lane & 15)][kk + ((lane >> 4) << 3)]);
                    pie_mma_bf16_16816(acc[bt][0], afrag, bfrag[0], bfrag[1]);
                    pie_mma_bf16_16816(acc[bt][1], afrag, bfrag[2], bfrag[3]);
                }
            }
            __syncthreads();
        }

        for (int bt = 0; bt < batches; ++bt) {

            {
                float* mine = c_tile + warp * 16 * 16;
                const int m = lane >> 2;
                const int col = (lane & 3) << 1;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    mine[m * 16 + t * 8 + col] = acc[bt][t][0];
                    mine[m * 16 + t * 8 + col + 1] = acc[bt][t][1];
                    mine[(m + 8) * 16 + t * 8 + col] = acc[bt][t][2];
                    mine[(m + 8) * 16 + t * 8 + col + 1] = acc[bt][t][3];
                }
            }
            __syncthreads();
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                const int m = idx / 16;
                const int nn = idx % 16;
                const int t = bt * kBatch + m;
                const int row = row0 + warp * 16 + nn;
                if (t < group && row < n) {
                    const int plane_route = order[g0 + t];
                    float v = c_tile[warp * 16 * 16 + idx];
                    if (b_row != nullptr) v += Elem<bf16>::to_f32(b_row[row]);
                    out[static_cast<long long>(plane_route) * n + row] = f32_to_bf16(v);
                }
            }
            __syncthreads();
        }
    }
}

template <int kBits, int kGroup>
__global__ void moe_matmul_select_mlxu4_wmma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{
    namespace wmma = ::nvcuda::wmma;
    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kBatch = 16;
    constexpr int kBatches = 2;
    constexpr int kRoutes = kBatch * kBatches;
    constexpr int kLd = kTileK + 8;
    constexpr int kWarps = kTileN / 16;
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;
    static_assert(kRoutes * kLd * 2 >= kWarps * 16 * 16 * 4, "the C tiles fit where the activations were");

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item / 65536;
    const int slice = item % 65536;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert] + slice * kRoutes;
    const int end = min(offsets[expert + 1], begin + kRoutes);
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int groups_per_row = k / kGroup;
    const int words_per_row = k / kPerWord;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }
    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * words_per_row * 4);
    const bf16* s16 = reinterpret_cast<const bf16*>(
        scales_at + static_cast<long long>(expert) * n * groups_per_row * 2);
    const bf16* b16 = reinterpret_cast<const bf16*>(
        biases_at + static_cast<long long>(expert) * n * groups_per_row * 2);

    __shared__ __align__(32) bf16 w_tile[kTileN][kLd];
    __shared__ __align__(32) bf16 x_tile[kRoutes][kLd];
    float* c_tile = reinterpret_cast<float*>(&x_tile[0][0]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const bf16 zero = f32_to_bf16(0.f);

    constexpr int kQuadWords = 4;
    constexpr int kQuadCodes = kQuadWords * kPerWord;
    constexpr int kQuadsPerTileK = kTileK / kQuadCodes;
    constexpr int kQuadsPerThread = kTileN * kQuadsPerTileK / 256;
    constexpr int kVec = 8;
    constexpr int kVecsPerTileK = kTileK / kVec;
    constexpr int kVecsPerThread = kRoutes * kVecsPerTileK / 256;
    static_assert(kGroup % kQuadCodes == 0, "a quad lies within one group");
    static_assert(kTileN * kQuadsPerTileK % 256 == 0 && kRoutes * kVecsPerTileK % 256 == 0, "even split");

    struct Staged {
        uint4 quad[kQuadsPerThread];
        unsigned short scale_bits[kQuadsPerThread];
        unsigned short zero_bits[kQuadsPerThread];
        bool quad_live[kQuadsPerThread];
        uint4 vec[kVecsPerThread];
    };
    const unsigned short* s_bits = reinterpret_cast<const unsigned short*>(s16);
    const unsigned short* b_bits = reinterpret_cast<const unsigned short*>(b16);

    {
        const int g0 = begin;
        const int group = end - g0;
        const int batches = (group + kBatch - 1) / kBatch;

        float acc[kBatches][2][4];
#pragma unroll
        for (int b = 0; b < kBatches; ++b)
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[b][t][e] = 0.f;

        const bf16* vec_src[kVecsPerThread];
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int t = (tid + i * 256) / kVecsPerTileK;
            vec_src[i] = nullptr;
            if (t < group) {
                const int plane_route = order[g0 + t];
                vec_src[i] = act + static_cast<long long>(plane_route / act_div) * k;
            }
        }

        auto fetch = [&](int k0, Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                const int kk = k0 + q * kQuadCodes;
                const int rr = row0 + r;
                st.quad_live[i] = kk + kQuadCodes <= k && rr < n;
                if (st.quad_live[i]) {
                    const long long fx = static_cast<long long>(rr) * groups_per_row + kk / kGroup;
                    st.scale_bits[i] = pie_ld_line_u16(s_bits + fx);
                    st.zero_bits[i] = pie_ld_line_u16(b_bits + fx);
                    st.quad[i] = pie_ld_line(reinterpret_cast<const uint4*>(
                        w32 + static_cast<long long>(rr) * words_per_row + kk / kPerWord));
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int v = (tid + i * 256) % kVecsPerTileK;
                const int kk = k0 + v * kVec;
                st.vec[i] = make_uint4(0u, 0u, 0u, 0u);
                if (vec_src[i] != nullptr && kk + kVec <= k) {
                    st.vec[i] = *reinterpret_cast<const uint4*>(vec_src[i] + kk);
                }
            }
        };

        auto store = [&](int k0, const Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                unsigned* dst = reinterpret_cast<unsigned*>(&w_tile[r][q * kQuadCodes]);
                if (st.quad_live[i]) {
                    const float sv = __uint_as_float(static_cast<unsigned>(st.scale_bits[i]) << 16);
                    const float zv = __uint_as_float(static_cast<unsigned>(st.zero_bits[i]) << 16);
                    const unsigned words[kQuadWords] = {st.quad[i].x, st.quad[i].y, st.quad[i].z, st.quad[i].w};
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
#pragma unroll
                        for (int j = 0; j < kPerWord; j += 2) {
                            const float c0 = static_cast<float>((words[w] >> (kBits * j)) & kMask);
                            const float c1 = static_cast<float>((words[w] >> (kBits * (j + 1))) & kMask);
                            dst[(w * kPerWord + j) / 2] = pack_bf16x2(fmaf(c0, sv, zv), fmaf(c1, sv, zv));
                        }
                    }
                } else {

                    const int kk = k0 + q * kQuadCodes;
                    const int rr = row0 + r;
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
                        const int kw = kk + w * kPerWord;
                        if (kw < k && rr < n) {
                            const long long fx = static_cast<long long>(rr) * groups_per_row + kw / kGroup;
                            const float sv = Elem<bf16>::to_f32(s16[fx]);
                            const float bv = Elem<bf16>::to_f32(b16[fx]);
                            const unsigned word =
                                w32[static_cast<long long>(rr) * words_per_row + kw / kPerWord];
#pragma unroll
                            for (int j = 0; j < kPerWord; ++j) {
                                const float code = static_cast<float>((word >> (kBits * j)) & kMask);
                                w_tile[r][q * kQuadCodes + w * kPerWord + j] = f32_to_bf16(fmaf(code, sv, bv));
                            }
                        } else {
#pragma unroll
                            for (int j = 0; j < kPerWord; ++j)
                                w_tile[r][q * kQuadCodes + w * kPerWord + j] = zero;
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int t = idx / kVecsPerTileK;
                const int v = idx % kVecsPerTileK;
                *reinterpret_cast<uint4*>(&x_tile[t][v * kVec]) = st.vec[i];
            }

            const int tail = k - k0;
            if (tail < kTileK && (tail % kVec) != 0) {
                const int from = (tail / kVec) * kVec;
                for (int idx = tid; idx < kRoutes * (kTileK - from); idx += blockDim.x) {
                    const int t = idx / (kTileK - from);
                    const int kk = from + idx % (kTileK - from);
                    bf16 vv = zero;
                    if (t < group && k0 + kk < k) {
                        const int plane_route = order[g0 + t];
                        vv = act[static_cast<long long>(plane_route / act_div) * k + k0 + kk];
                    }
                    x_tile[t][kk] = vv;
                }
            }
        };

        Staged staged;
        fetch(0, staged);
        for (int k0 = 0; k0 < k; k0 += kTileK) {
            store(k0, staged);
            __syncthreads();
            if (k0 + kTileK < k) fetch(k0 + kTileK, staged);
#pragma unroll
            for (int kk = 0; kk < kTileK; kk += 16) {

                unsigned bfrag[4];
                pie_ldmatrix_x4(bfrag, &w_tile[warp * 16 + (lane & 7) + ((lane >> 4) << 3)][kk + (((lane >> 3) & 1) << 3)]);

#pragma unroll
                for (int bt = 0; bt < kBatches; ++bt) {

                    unsigned afrag[4];
                    pie_ldmatrix_x4(afrag, &x_tile[bt * kBatch + (lane & 15)][kk + ((lane >> 4) << 3)]);
                    pie_mma_bf16_16816(acc[bt][0], afrag, bfrag[0], bfrag[1]);
                    pie_mma_bf16_16816(acc[bt][1], afrag, bfrag[2], bfrag[3]);
                }
            }
            __syncthreads();
        }

        for (int bt = 0; bt < batches; ++bt) {

            {
                float* mine = c_tile + warp * 16 * 16;
                const int m = lane >> 2;
                const int col = (lane & 3) << 1;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    mine[m * 16 + t * 8 + col] = acc[bt][t][0];
                    mine[m * 16 + t * 8 + col + 1] = acc[bt][t][1];
                    mine[(m + 8) * 16 + t * 8 + col] = acc[bt][t][2];
                    mine[(m + 8) * 16 + t * 8 + col + 1] = acc[bt][t][3];
                }
            }
            __syncthreads();
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                const int m = idx / 16;
                const int nn = idx % 16;
                const int t = bt * kBatch + m;
                const int row = row0 + warp * 16 + nn;
                if (t < group && row < n) {
                    const int plane_route = order[g0 + t];
                    out[static_cast<long long>(plane_route) * n + row] =
                        f32_to_bf16(c_tile[warp * 16 * 16 + idx]);
                }
            }
            __syncthreads();
        }
    }
}

template <int kW>
struct CodeChunk {
    unsigned w[kW];
};

template <int kW>
__device__ __forceinline__ CodeChunk<kW> load_code_chunk(const unsigned* at) {
    CodeChunk<kW> c;
    if constexpr (kW == 4) {
        const uint4 v = *reinterpret_cast<const uint4*>(at);
        c.w[0] = v.x; c.w[1] = v.y; c.w[2] = v.z; c.w[3] = v.w;
    } else if constexpr (kW == 2) {
        const uint2 v = *reinterpret_cast<const uint2*>(at);
        c.w[0] = v.x; c.w[1] = v.y;
    } else {
        c.w[0] = *at;
    }
    return c;
}

template <class T, int kBits, int kGroup, int kRowsT>
__global__ void moe_matmul_select_mlxu4(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ out,
    int top_k,
    int act_div,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;
    constexpr int kWordsPerGroup = kGroup / kPerWord;
    // A lane takes up to 16 bytes of one group's codes per load, and kU of
    // those chunks for every row before it computes, so the codes stream.
    constexpr int kW = kWordsPerGroup < 4 ? kWordsPerGroup : 4;
    constexpr int kChunk = kW * kPerWord;
    constexpr int kU = 2;
    static_assert(kChunk % 8 == 0, "a chunk's activations are whole 16-byte loads");
    const int route = blockIdx.x;

    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const int plane_route = win != nullptr
        ? route + static_cast<int>(win[1]) * top_k
        : route;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes[plane_route];

    moe_note_group(group_hits);

    const int groups_per_row = k / kGroup;
    const int words_per_row = k / kPerWord;
    const int chunks = k / kChunk;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }

    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * words_per_row * 4);
    const bf16* s16 = reinterpret_cast<const bf16*>(
        scales_at + static_cast<long long>(expert) * n * groups_per_row * 2);
    const bf16* b16 = reinterpret_cast<const bf16*>(
        biases_at + static_cast<long long>(expert) * n * groups_per_row * 2);

    const T* x = act + static_cast<long long>(plane_route / act_div) * k;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int c0 = lane_id; c0 < chunks; c0 += 32 * kU) {
        CodeChunk<kW> ww[kU][kRows];
        bf16 sv[kU][kRows];
        bf16 bv[kU][kRows];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = min(c0 + 32 * u, chunks - 1);
            const int g = c * kChunk / kGroup;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                ww[u][r] = load_code_chunk<kW>(
                    w32 + static_cast<long long>(row_of[r]) * words_per_row + c * kW);
                const long long fx = static_cast<long long>(row_of[r]) * groups_per_row + g;
                sv[u][r] = s16[fx];
                bv[u][r] = b16[fx];
            }
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = c0 + 32 * u;
            if (c >= chunks) break;
            float part[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[r] = 0.f;
            float xsum = 0.f;
#pragma unroll
            for (int v = 0; v < kChunk / 8; ++v) {
                const uint4 xq = x4[(c * kChunk) / 8 + v];
                const T* xe = reinterpret_cast<const T*>(&xq);
                float xv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    xv[j] = Elem<T>::to_f32(xe[j]);
                    xsum += xv[j];
                }
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int e = v * 8 + j;
                        const unsigned word = ww[u][r].w[e / kPerWord];
                        const float code =
                            static_cast<float>((word >> (kBits * (e % kPerWord))) & kMask);
                        part[r] = fmaf(code, xv[j], part[r]);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                acc[r] = fmaf(part[r], Elem<bf16>::to_f32(sv[u][r]), acc[r]);
                acc[r] = fmaf(xsum, Elem<bf16>::to_f32(bv[u][r]), acc[r]);
            }
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
                out[static_cast<long long>(plane_route) * n + row] =
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

constexpr int kOffPost = 0;
constexpr int kOffPreInt = 1;
constexpr int kOffPreReal = 2;
constexpr int kOffPreConst = 3;

template <class T, class F, int kBits, int kOffset, int kGroup, int kRowsT>
__global__ void matmul_affine(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ out,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;

    constexpr float kExcess = static_cast<float>(1 << (kBits - 1));
    const int token = blockIdx.x;

    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }

    const int groups_per_row = k / kGroup;
    constexpr int kWordsPerGroup = kGroup / kPerWord;
    const int words_per_row = k / kPerWord;
    const unsigned* w32 = reinterpret_cast<const unsigned*>(codes_at);
    const F* sf = reinterpret_cast<const F*>(scales_at);

    const F* bf = reinterpret_cast<const F*>(biases_at);
    const u8* zb = biases_at;
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < groups_per_row; g += 32) {

        float part[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) part[r] = 0.f;
        float xsum = 0.f;

#pragma unroll
        for (int q = 0; q < kWordsPerGroup; ++q) {
            float xv[kPerWord];
#pragma unroll
            for (int j = 0; j < kPerWord; ++j) {
                xv[j] = Elem<T>::to_f32(x[g * kGroup + q * kPerWord + j]);
                xsum += xv[j];
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const unsigned word =
                    w32[static_cast<long long>(row_of[r]) * words_per_row
                        + g * kWordsPerGroup + q];
#pragma unroll
                for (int j = 0; j < kPerWord; ++j) {
                    const float code =
                        static_cast<float>((word >> (kBits * j)) & kMask);
                    part[r] = fmaf(code, xv[j], part[r]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const long long fx =
                static_cast<long long>(row_of[r]) * groups_per_row + g;
            const float sv = Elem<F>::to_f32(sf[fx]);
            if constexpr (kOffset == kOffPost) {
                acc[r] = fmaf(part[r], sv, acc[r]);
                acc[r] = fmaf(xsum, Elem<F>::to_f32(bf[fx]), acc[r]);
            } else {

                float z;
                if constexpr (kOffset == kOffPreInt) {
                    z = static_cast<float>(zb[fx]);
                } else if constexpr (kOffset == kOffPreReal) {
                    z = Elem<F>::to_f32(bf[fx]);
                } else {
                    z = kExcess;
                }
                acc[r] = fmaf(sv, fmaf(-z, xsum, part[r]), acc[r]);
            }
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
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

template <class F, int kBits, int kOffset, int kGroup>
__global__ void dequant_affine(
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    bf16* __restrict__ out,
    int n,
    int k)
{
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;

    constexpr float kExcess = static_cast<float>(1 << (kBits - 1));

    const int words_per_row = k / kPerWord;
    const long long words = static_cast<long long>(n) * words_per_row;
    const long long at =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= words) return;

    const int row = static_cast<int>(at / words_per_row);
    const int word_in_row = static_cast<int>(at % words_per_row);

    const int groups_per_row = k / kGroup;
    const long long fx = static_cast<long long>(row) * groups_per_row
        + (word_in_row * kPerWord) / kGroup;

    const F* sf = reinterpret_cast<const F*>(scales);
    const F* bf = reinterpret_cast<const F*>(biases);
    const u8* zb = biases;
    const float sv = Elem<F>::to_f32(sf[fx]);
    float off;
    if constexpr (kOffset == kOffPost || kOffset == kOffPreReal) {
        off = Elem<F>::to_f32(bf[fx]);
    } else if constexpr (kOffset == kOffPreInt) {
        off = static_cast<float>(zb[fx]);
    } else {
        off = kExcess;
    }

    const unsigned word = reinterpret_cast<const unsigned*>(codes)[at];
    bf16* dst = out + static_cast<long long>(row) * k + word_in_row * kPerWord;

    unsigned int* dst2 = reinterpret_cast<unsigned int*>(dst);
#pragma unroll
    for (int j = 0; j < kPerWord; j += 2) {
        float v[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float code = static_cast<float>((word >> (kBits * (j + h))) & kMask);
            if constexpr (kOffset == kOffPost) {
                v[h] = fmaf(code, sv, off);
            } else {
                v[h] = sv * (code - off);
            }
        }
        dst2[j / 2] = pack_bf16x2(v[0], v[1]);
    }
}

using f32 = float;

template <class T>
struct Cast {
    static __device__ __forceinline__ float to_f32(T v) { return Elem<T>::to_f32(v); }
    static __device__ __forceinline__ T from_f32(float v) { return Elem<T>::from_f32(v); }
};

template <>
struct Cast<f32> {
    static __device__ __forceinline__ float to_f32(f32 v) { return v; }
    static __device__ __forceinline__ f32 from_f32(float v) { return v; }
};

template <class T>
__global__ void cast_f32_to(const float* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(src[i]);
}

template <class T>
__global__ void scale(
    const T* __restrict__ src, T* __restrict__ dst, usize n, float factor) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = Cast<T>::from_f32(Cast<T>::to_f32(src[i]) * factor);
}

template <class T>
__global__ void scale_rows(T* buf, const T* l, int width) {
    const int row = blockIdx.x;
    T* row_buf = buf + static_cast<usize>(row) * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        row_buf[c] =
            Cast<T>::from_f32(Cast<T>::to_f32(row_buf[c]) * Cast<T>::to_f32(l[c]));
    }
}

}
