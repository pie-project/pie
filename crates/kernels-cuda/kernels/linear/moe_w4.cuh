#pragma once

// A routed mxfp4 select at decode widths: each expert's weight rows stream
// once into registers, are dequantized there and meet the expert's few routes
// on mma.sync, with no shared-memory or TMEM round trip between.
//
// A warp holds two m16 tiles of weight rows (32 rows, or a GLU's 16 gate
// rows and their up rows) and walks K 128 codes at a time.
// In each 128-code chunk lane (g, c) owns the 32-code group c of rows g and
// g + 8: one 16-byte load a row. The MMA's K order is free as long as both
// operands agree, so k-step j reads word j / 2 of the group, and the pair
// whose low half is code 3 - p and high half code 7 - p of that word (p the
// pair's place, 0..3) is what one shift lays at bits 12..15 and 28..31: the
// e2m1 code there becomes a bf16 by two masks and a shift. The activations
// are permuted to match, two byte-perms a register.

#include <cuda_pipeline.h>

#include "prelude/device.cuh"
#include "prelude/glu.cuh"
#include "linear/grouped_fp8.cuh"

namespace pie::linear {

namespace w4mma {

using ::pie::ld_stream;

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// a * b, bf16 x 2, subnormals kept.
__device__ __forceinline__ unsigned mul2(unsigned a, unsigned b) {
    unsigned d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
}

/// The e2m1 codes at bits 12..15 and 28..31 of `q` as two bf16, each 2^-126
/// times its value (an exponent field of e, a code of exponent 0 subnormal).
__device__ __forceinline__ unsigned e2m1_pair(unsigned q) {
    return (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
}

__device__ __forceinline__ unsigned ld_scale2(const u8* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned ld_scale(const u8* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

using ::pie::glu_fn::glu;

__device__ __forceinline__ uint4 ld_act(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

/// An NVFP4 block scale (e4m3) as bf16 bits, times 2^118: an e2m1 code's
/// 2^-126 (`e2m1_pair`) times it is the product times 2^-8, exact and
/// normal across e4m3's range; the epilogue's factor restores the 2^8.
__device__ __forceinline__ unsigned nvfp4_scale_bf16(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0) {
        // m 2^-9 2^118, exact in f32.
        return __float_as_uint(static_cast<float>(m) * 6.4903710731685345e32f) >> 16;
    }
    return ((e + 238u) << 7) | (m << 4);
}

/// One 128-code chunk a lane holds: its rows' code groups and scales, and
/// its routes' activations over the group.
template <int kTiles, int kNb>
struct Chunk {
    uint4 w[kTiles][2];
    unsigned s[kTiles][2];
    uint4 x[kNb][4];
};

// 2^(s - 1) makes up a code's 2^-126 and the block's 2^(s - 127); past
// s = 128 that overflows bf16, and a warp with such a block takes it in
// two factors.
// NVFP4 (`kNv`): a lane's 32 codes are two 16-code blocks, words 0-1 and
// 2-3, the two e4m3 scales in the low and high byte of its scale word.
template <bool wide, bool kNv, int kTiles, int kNb>
__device__ __forceinline__ void compute(const Chunk<kTiles, kNb>& cc, float (&acc)[kTiles][kNb][4]) {
    unsigned s1[kTiles][2], s2[kTiles][2], nv[kTiles][2][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned s = cc.s[t][h];
            if constexpr (kNv) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const unsigned v = nvfp4_scale_bf16((s >> (8 * q)) & 0xFFu);
                    nv[t][h][q] = v | (v << 16);
                }
                s1[t][h] = s2[t][h] = 0u;
            } else {
                const unsigned a = !wide || s <= 128u ? (s + 126u) << 7 : 253u << 7;
                const unsigned b = s <= 128u ? 0x3F80u : s << 7;
                s1[t][h] = a | (a << 16);
                s2[t][h] = b | (b << 16);
            }
        }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if constexpr (kNv) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) s1[t][h] = nv[t][h][w >> 1];
        }
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            unsigned b0[kNb], b1[kNb];
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                const uint4 x = cc.x[b][w];
                const unsigned lo = jj == 0 ? x.y : x.x;
                const unsigned hi = jj == 0 ? x.w : x.z;
                b0[b] = __byte_perm(lo, hi, 0x7632);
                b1[b] = __byte_perm(lo, hi, 0x5410);
            }
            const int p0 = 2 * jj;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const unsigned w0 = (&cc.w[t][0].x)[w];
                const unsigned w1 = (&cc.w[t][1].x)[w];
                unsigned a0 = mul2(e2m1_pair(w0 << (4 * p0)), s1[t][0]);
                unsigned a1 = mul2(e2m1_pair(w1 << (4 * p0)), s1[t][1]);
                unsigned a2 = mul2(e2m1_pair(w0 << (4 * p0 + 4)), s1[t][0]);
                unsigned a3 = mul2(e2m1_pair(w1 << (4 * p0 + 4)), s1[t][1]);
                if constexpr (wide) {
                    a0 = mul2(a0, s2[t][0]);
                    a1 = mul2(a1, s2[t][1]);
                    a2 = mul2(a2, s2[t][0]);
                    a3 = mul2(a3, s2[t][1]);
                }
#pragma unroll
                for (int b = 0; b < kNb; ++b) mma(acc[t][b], a0, a1, a2, a3, b0[b], b1[b]);
            }
        }
    }
}

// A warp's sums landed at their routes' rows: each row's bias (`b_e`, or
// none) after its `factor`, the gate and up rows through the GLU into
// `inter` columns where `kGlu`, else `n` columns as they are.
template <int kGlu, int kTiles, int kNb, class Factor>
__device__ __forceinline__ void w4_land(const float (&acc)[kTiles][kNb][4], Factor factor,
                                        const bf16* b_e, const i32* __restrict__ order, int begin,
                                        int count, bf16* __restrict__ out, int n, int inter,
                                        int row_base, int g, int c, float limit, float galpha) {
    if constexpr (kGlu != 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = row_base + 8 * h + g;
            if (col >= inter) continue;
            const float bg = b_e != nullptr ? bf16_to_f32(b_e[col]) : 0.f;
            const float bu = b_e != nullptr ? bf16_to_f32(b_e[inter + col]) : 0.f;
            const float fg = factor(col), fu = factor(inter + col);
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * inter + col] = f32_to_bf16(glu<kGlu>(
                            acc[0][b][2 * h + e] * fg + bg, acc[1][b][2 * h + e] * fu + bu, limit,
                            galpha));
                    }
                }
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row_base + 16 * t + 8 * h + g;
            if (row >= n) continue;
            const float add = b_e != nullptr ? bf16_to_f32(b_e[row]) : 0.f;
            const float f = factor(row);
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * n + row] = f32_to_bf16(acc[t][b][2 * h + e] * f + add);
                    }
                }
        }
}

}  // namespace w4mma

// A routed bf16 select at decode widths: the bf16 analog of
// `moe_select_mxfp4_mma`, without its GLU. Each expert's weight rows and the
// routes' rows are staged a K-tile at a time into shared memory with
// `cp.async`, read to the mma fragments by `ldmatrix`, and met on mma.sync.
// The routing (order/offsets/work) and the epilogue (`w4mma::w4_land`) are
// shared with the mxfp4 kernel; the fragments are the ones `attn/mla.cuh`
// loads, so the output fragment is the layout `w4_land` reads.
// The bf16 select's K step and its pipeline's stages (`moe_select_bf16_mma`).
constexpr int kBf16MmaBK = 64;
constexpr int kBf16MmaStages = 3;

namespace bf16mma {
__device__ __forceinline__ void ld_a(unsigned (&a)[4], const bf16* base, int ld, int lane) {
    const bf16* r = base + static_cast<long long>(lane & 15) * ld + ((lane & 16) ? 8 : 0);
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(r));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}
__device__ __forceinline__ void ld_b(unsigned (&b)[2], const bf16* base, int ld, int lane) {
    const bf16* q = base + static_cast<long long>(lane & 7) * ld + ((lane & 8) ? 8 : 0);
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(q));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
}
}  // namespace bf16mma

template <int kNb>
__global__ void __launch_bounds__(128) moe_select_bf16_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const bf16* __restrict__ weight,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div, int n, int k, int num_experts)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;   // 128 weight rows a CTA
    constexpr int kSlice = 8 * kNb;           // routes a slice (B of the GEMM)
    constexpr int kBK = kBf16MmaBK;           // K staged a step (four mma k)
    constexpr int kPad = kBK + 8;             // smem row stride (ldmatrix)
    constexpr int kStages = kBf16MmaStages;
    constexpr int kChunks = kBK / 8;          // 16-byte chunks a row a step
    constexpr int kStage = (kRowsCta + kSlice) * kPad;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    // The CTA's 128 rows and this warp's 32 of them. A warp whose rows run
    // past the bank still joins the cooperative load and the barriers;
    // `w4_land` drops its out-of-range rows.
    const int cta_row0 = static_cast<int>(blockIdx.y) * kRowsCta;
    const int row_base = cta_row0 + warp * kRowsWarp;

    const bf16* w_e = weight + static_cast<long long>(expert) * n * k;
    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;

    // `kStages` stages, each 128 weight rows then the slice's route rows.
    extern __shared__ bf16 smem[];

    // Stage `st` takes K-step `step`: 16 bytes a cp.async. Rows past the
    // bank or routes past the slice are zeroed so the mma reads defined
    // smem (their outputs are not stored).
    auto load = [&](int step, int st) {
        bf16* sW = smem + st * kStage;
        bf16* sX = sW + kRowsCta * kPad;
        const int k0 = step * kBK;
#pragma unroll
        for (int i = tid; i < kRowsCta * kChunks; i += 128) {
            const int row = i / kChunks;
            const int ch = (i % kChunks) * 8;
            bf16* dst = sW + row * kPad + ch;
            if (cta_row0 + row < n) {
                __pipeline_memcpy_async(
                    dst, w_e + static_cast<long long>(cta_row0 + row) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        for (int i = tid; i < kSlice * kChunks; i += 128) {
            const int tok = i / kChunks;
            const int ch = (i % kChunks) * 8;
            bf16* dst = sX + tok * kPad + ch;
            if (tok < count) {
                const long long r = static_cast<long long>(order[begin + tok] / act_div);
                __pipeline_memcpy_async(dst, act + r * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

    // Every step commits a group (empty past the end), so the wait for
    // step `s` is always the same count behind.
    const int steps = k / kBK;
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) load(s, s);
        __pipeline_commit();
    }
    for (int s = 0; s < steps; ++s) {
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        // The stage refilled here was read at step s - 1, behind the barrier.
        const int next = s + kStages - 1;
        if (next < steps) load(next, next % kStages);
        __pipeline_commit();

        const bf16* sW = smem + (s % kStages) * kStage;
        const bf16* sX = sW + kRowsCta * kPad;
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned bk[kNb][2];
#pragma unroll
            for (int b = 0; b < kNb; ++b) bf16mma::ld_b(bk[b], sX + 8 * b * kPad + kk, kPad, lane);
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                unsigned a[4];
                bf16mma::ld_a(a, sW + (warp * kRowsWarp + 16 * t) * kPad + kk, kPad, lane);
#pragma unroll
                for (int b = 0; b < kNb; ++b) {
                    w4mma::mma(acc[t][b], a[0], a[1], a[2], a[3], bk[b][0], bk[b][1]);
                }
            }
        }
    }

    auto factor = [&](int) { return 1.f; };
    w4mma::w4_land<0, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, n, row_base,
                                   g, c, 0.f, 1.f);
}

/// `y[route] = x[route / act_div] . W[expert]^T (+ bias[expert])` for the
/// routes `order` lists by expert, `work` naming (expert, slice of
/// `slice_routes` routes) per CTA (from `moe_route_order`). A CTA covers 128
/// weight rows; `kNb` 8-route blocks a slice (`slice_routes` = 8 kNb).
/// With `kGlu`, the bank's `n` rows are `inter` gate rows and as many up
/// rows: a warp's two tiles are 16 gate rows and their up rows, and `out`
/// is the GLU, `inter` wide (a CTA 64 of it).
/// A block-wide exclusive scan of one value a thread (blockDim.x a whole
/// number of warps, at most 1024); the total lands in `*total`.
__device__ __forceinline__ int w4_block_scan(int v, int* warp_sums, int* total) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warps = static_cast<int>(blockDim.x) >> 5;
    int x = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < warps ? warp_sums[lane] : 0;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += y;
        }
        if (lane < warps) warp_sums[lane] = w;
        if (lane == 31) *total = w;
    }
    __syncthreads();
    const int before = warp > 0 ? warp_sums[warp - 1] : 0;
    return before + x - v;
}

/// `moe_route_order`'s product (routes sorted by expert, each expert's
/// offset, the work list of (expert, slice) items in expert order) with the
/// two serial walks over the experts done as block scans; `offsets` holds
/// `num_experts + 3`: the experts' starts and the dropped routes' (those
/// naming no expert), the work items' count, and the live routes' count.
__global__ void moe_route_order_scan(
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
    __shared__ int warp_sums[32];
    __shared__ int total;
    const int live = win != nullptr
        ? min(static_cast<int>(win[0]) * top_k, route_count)
        : route_count;
    const int base = win != nullptr ? static_cast<int>(win[1]) * top_k : 0;
    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    for (int e = tid; e <= num_experts; e += threads) {
        counts[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
    for (int r = tid; r < live; r += threads) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        atomicAdd(counts + e, 1);
    }
    __syncthreads();
    // Each thread a contiguous run of experts: their routes and work items.
    const int per = (num_experts + 1 + threads - 1) / threads;
    const int e0 = min(tid * per, num_experts + 1);
    const int e1 = min(e0 + per, num_experts + 1);
    int mine = 0, items = 0;
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        mine += c;
        if (e < num_experts) items += (c + group_routes - 1) / group_routes;
    }
    int route_at = w4_block_scan(mine, warp_sums, &total);
    __syncthreads();
    int item_at = w4_block_scan(items, warp_sums, &total);
    const int items_total = total;
    __syncthreads();
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        counts[e] = route_at;
        offsets[e] = route_at;
        route_at += c;
        if (e < num_experts) {
            for (int g = 0; g * group_routes < c; ++g, ++item_at)
                if (item_at < work_cap) work[item_at] = e * 65536 + g;
        }
    }
    if (tid == 0) {
        offsets[num_experts + 1] = min(items_total, work_cap);
        // The dropped routes' bucket ends at the live routes' count.
        offsets[num_experts + 2] = live;
    }
    __syncthreads();
    for (int r = tid; r < live; r += threads) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        const int pos = counts[e] + atomicAdd(fill + e, 1);
        order[pos] = base + r;
    }
}

// `kNv`: an NVFP4 bank (an e4m3 scale a 16 codes, `tscale` an f32 tensor
// scale a weight row) in place of mxfp4's e8m0 a 32.
template <int kNb, int kDepth, int kGlu, bool kNv = false>
__global__ void __launch_bounds__(128) moe_select_mxfp4_mma(
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
    int inter,
    float limit,
    float galpha,
    const float* __restrict__ tscale)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;
    constexpr int kSlice = 8 * kNb;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(blockIdx.y) * kRowsCta + warp * kRowsWarp;
    // A tile's first row: the next 16 rows, or a GLU's up rows `inter` on.
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    if (row_base >= span) return;

    const int groups = k >> 5;
    const int chunks = (groups + 3) >> 2;
    const long long row_bytes = k >> 1;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    // A row's scales: one a 32-code group (mxfp4), two (NVFP4).
    const int sf_row = kNv ? 2 * groups : groups;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;

    const u8* wrow[kTiles][2];
    const u8* srow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(tile_row(t) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + r * row_bytes + c * 16;
            srow[t][h] = s_e + static_cast<long long>(r) * sf_row + (kNv ? 2 * c : c);
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 32
            : nullptr;
    }

    using Chunk = w4mma::Chunk<kTiles, kNb>;
    auto load = [&](int ch, Chunk& cc) {
        const bool live = ch * 4 + c < groups;
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cc.w[t][h] = live ? w4mma::ld_stream(wrow[t][h] + ch * 64) : make_uint4(0, 0, 0, 0);
                if constexpr (kNv) {
                    cc.s[t][h] = live ? w4mma::ld_scale2(srow[t][h] + ch * 8) : 0u;
                } else {
                    cc.s[t][h] = live ? w4mma::ld_scale(srow[t][h] + ch * 4) : 127u;
                }
            }
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                cc.x[b][q] = live && xrow[b] != nullptr
                    ? w4mma::ld_act(xrow[b] + ch * 128 + q * 8)
                    : make_uint4(0, 0, 0, 0);
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

    Chunk ring[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
        if (d < chunks) load(d, ring[d]);
    for (int ch0 = 0; ch0 < chunks; ch0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int ch = ch0 + d;
            if (ch < chunks) {
                bool wide = false;
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
                    wide |= ring[d].s[t][0] > 128u || ring[d].s[t][1] > 128u;
                if constexpr (kNv) {
                    w4mma::compute<false, true>(ring[d], acc);
                } else if (__any_sync(0xffffffffu, wide)) {
                    w4mma::compute<true, false>(ring[d], acc);
                } else {
                    w4mma::compute<false, false>(ring[d], acc);
                }
                if (ch + kDepth < chunks) load(ch + kDepth, ring[d]);
            }
        }
    }

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    // NVFP4: a weight row's tensor scale, and the 2^8 the scales held back.
    const float* t_e = kNv ? tscale + static_cast<long long>(expert) * n : nullptr;
    auto factor = [&](int row) { return kNv ? t_e[row] * 256.f : 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, inter,
                                      row_base, g, c, limit, galpha);
}


/// Rows `[rows, k]` of bf16 as e4m3 in 1x128 blocks, as the tcgen05 path
/// quantizes them (`E4m3Block`): codes row-major into `codes`, each block's
/// biased exponent a byte into `exps` (`[rows, k / 128]`). 16 lanes a
/// block; 128 threads, `ceil(rows k / 1024)` blocks.
__global__ void __launch_bounds__(128) quantize_rows_e4m3b128(
    const bf16* __restrict__ x, u8* __restrict__ codes, u8* __restrict__ exps, int rows, int k)
{
    using Block = E4m3Block<128>;
    const int blocks = k / 128;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    // Every lane joins the block's shuffles; a slot past the rows reads row 0.
    const bool live = slot < rows * blocks;
    const int sl = live ? slot : 0;
    const long long at = static_cast<long long>(sl / blocks) * k + (sl % blocks) * 128;
    Block q;
    q.load(x + at, lane);
    if (!live) return;
    *reinterpret_cast<unsigned long long*>(codes + at + (lane % Block::kLanes) * 8) = q.codes();
    if (lane % Block::kLanes == 0) exps[sl] = static_cast<u8>(q.e + 127);
}

// The fp8 select's pipeline: K steps of one 128-wide block, three stages.
constexpr int kFp8MmaStages = 3;

/// `y[route] = q(x[route / act_div]) . (W[expert] * 2^(s - 127))^T` for a
/// 128x128-block FP8 bank (e4m3 codes, each block's ue8m0 exponent spread to
/// its rows in tcgen05's atoms, a weight row an atom row) against the rows
/// quantized by `quantize_rows_e4m3b128` (W8A8): `moe_select_bf16_mma`'s
/// staging over e4m3 x e4m3 mma.sync, each 128-wide block's sums scaled by
/// its two exponents. `kDense`: one plane, rows in order, no routing (`order`,
/// `offsets`, `work` unread, a CTA the `kSlice` rows at `blockIdx.x`), K
/// split `gridDim.z` ways: each split's f32 sums into `part`
/// (`[splits][rows][n]`) for `fp8_mma_fold`, or one split's straight out.
template <int kNb, bool kDense>
__global__ void __launch_bounds__(128) moe_select_fp8_mma(
    const u8* __restrict__ xq,
    const u8* __restrict__ xe,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ weight,
    const u8* __restrict__ atoms,
    bf16* __restrict__ out,
    float* __restrict__ part,
    int act_div, int n, int k, int num_experts, int rows)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;   // 128 weight rows a CTA: one block's
    constexpr int kSlice = 8 * kNb;           // routes a slice (B of the GEMM)
    constexpr int kBK = 128;                  // K a step: one block's codes
    constexpr int kPad = kBK + 16;            // smem row stride in bytes (ldmatrix)
    constexpr int kStages = kFp8MmaStages;
    constexpr int kChunks = kBK / 16;
    constexpr int kStage = (kRowsCta + kSlice) * kPad;

    int expert = 0, begin = 0, count = 0;
    if constexpr (kDense) {
        begin = static_cast<int>(blockIdx.x) * kSlice;
        count = min(rows - begin, kSlice);
    } else {
        if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
        const int item = work[blockIdx.x];
        expert = item >> 16;
        begin = offsets[expert] + (item & 0xFFFF) * kSlice;
        count = min(offsets[expert + 1] - begin, kSlice);
    }
    if (count <= 0) return;
    auto act_row = [&](int tok) {
        if constexpr (kDense) return begin + tok;
        else return order[begin + tok] / act_div;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int cta_row0 = static_cast<int>(blockIdx.y) * kRowsCta;
    const int row_base = cta_row0 + warp * kRowsWarp;
    const int blocks = k / kBK;
    // This CTA's share of K, in blocks: all of it but a split dense fire.
    const int per = (blocks + static_cast<int>(gridDim.z) - 1) / static_cast<int>(gridDim.z);
    const int s_lo = min(blocks, static_cast<int>(blockIdx.z) * per);
    const int s_hi = min(blocks, s_lo + per);

    const u8* w_e = weight + static_cast<long long>(expert) * n * k;
    const u8* atoms_at = atoms + atom_offset(expert * n + cta_row0, 0, blocks);

    extern __shared__ uint4 fp8mma_smem[];
    u8* smem = reinterpret_cast<u8*>(fp8mma_smem);

    auto load = [&](int step, int st) {
        u8* sW = smem + st * kStage;
        u8* sX = sW + kRowsCta * kPad;
        const int k0 = step * kBK;
#pragma unroll
        for (int i = tid; i < kRowsCta * kChunks; i += 128) {
            const int row = i / kChunks;
            const int ch = (i % kChunks) * 16;
            u8* dst = sW + row * kPad + ch;
            if (cta_row0 + row < n) {
                __pipeline_memcpy_async(
                    dst, w_e + static_cast<long long>(cta_row0 + row) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        for (int i = tid; i < kSlice * kChunks; i += 128) {
            const int tok = i / kChunks;
            const int ch = (i % kChunks) * 16;
            u8* dst = sX + tok * kPad + ch;
            if (tok < count) {
                __pipeline_memcpy_async(
                    dst, xq + static_cast<long long>(act_row(tok)) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // The activation rows of this lane's two columns in each 8-route block
    // (C's columns 2c and 2c + 1), -1 past the slice.
    int xrow[kNb][2];
#pragma unroll
    for (int b = 0; b < kNb; ++b)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tok = 8 * b + 2 * c + h;
            xrow[b][h] = tok < count ? act_row(tok) : -1;
        }

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (s_lo + i < s_hi) load(s_lo + i, i);
        __pipeline_commit();
    }
    for (int s = s_lo; s < s_hi; ++s) {
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        const int next = s + kStages - 1;
        if (next < s_hi) load(next, (next - s_lo) % kStages);
        __pipeline_commit();

        const u8* sW = smem + ((s - s_lo) % kStages) * kStage;
        const u8* sX = sW + kRowsCta * kPad;
        float blk[kTiles][kNb][4];
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) blk[t][b][e] = 0.f;
        // The bf16 ldmatrix loads read e4m3 fragments: a 16-byte row of an
        // 8x8 b16 matrix is a 16-code row of K, 4 codes a lane, the order
        // m16n8k32's A and B fragments take.
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 32) {
            unsigned bk[kNb][2];
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                bf16mma::ld_b(bk[b], reinterpret_cast<const bf16*>(sX + 8 * b * kPad + kk),
                              kPad / 2, lane);
            }
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                unsigned a[4];
                bf16mma::ld_a(a,
                              reinterpret_cast<const bf16*>(sW + (warp * kRowsWarp + 16 * t) * kPad + kk),
                              kPad / 2, lane);
#pragma unroll
                for (int b = 0; b < kNb; ++b) {
                    gfp8::mma(blk[t][b], a[0], a[1], a[2], a[3], bk[b][0], bk[b][1]);
                }
            }
        }
        // The CTA's 128 rows are one block's: one weight exponent a step.
        const float sw = gfp8::exp2_biased(atoms_at[(s / 4) * 512 + s % 4]);
#pragma unroll
        for (int b = 0; b < kNb; ++b) {
            const float s0 = xrow[b][0] >= 0 ? sw * gfp8::exp2_biased(xe[xrow[b][0] * blocks + s]) : 0.f;
            const float s1 = xrow[b][1] >= 0 ? sw * gfp8::exp2_biased(xe[xrow[b][1] * blocks + s]) : 0.f;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                acc[t][b][0] = fmaf(blk[t][b][0], s0, acc[t][b][0]);
                acc[t][b][1] = fmaf(blk[t][b][1], s1, acc[t][b][1]);
                acc[t][b][2] = fmaf(blk[t][b][2], s0, acc[t][b][2]);
                acc[t][b][3] = fmaf(blk[t][b][3], s1, acc[t][b][3]);
            }
        }
    }

    if constexpr (kDense) {
        // Rows in order: output row `begin + col`.
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int tok = 8 * b + 2 * c + (e & 1);
                    const int row = row_base + 16 * t + g + (e >= 2 ? 8 : 0);
                    if (tok < count && row < n) {
                        const long long at = static_cast<long long>(begin + tok) * n + row;
                        if (gridDim.z > 1) {
                            part[blockIdx.z * static_cast<long long>(rows) * n + at] = acc[t][b][e];
                        } else {
                            out[at] = f32_to_bf16(acc[t][b][e]);
                        }
                    }
                }
    } else {
        auto factor = [&](int) { return 1.f; };
        w4mma::w4_land<0, kTiles, kNb>(acc, factor, nullptr, order, begin, count, out, n, n,
                                       row_base, g, c, 0.f, 1.f);
    }
}

/// A split `moe_select_fp8_mma` dense fire's `splits` f32 partials
/// (`[splits][count]`) summed in split order into bf16 `out`. Four a thread.
__global__ void __launch_bounds__(256) fp8_mma_fold(
    const float* __restrict__ part, bf16* __restrict__ out, long long count, int splits)
{
    const long long at = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (at >= count) return;
    float4 sum = *reinterpret_cast<const float4*>(part + at);
    for (int z = 1; z < splits; ++z) {
        const float4 v = *reinterpret_cast<const float4*>(part + z * count + at);
        sum.x += v.x;
        sum.y += v.y;
        sum.z += v.z;
        sum.w += v.w;
    }
    uint2 packed;
    packed.x = pack_bf16x2(sum.x, sum.y);
    packed.y = pack_bf16x2(sum.z, sum.w);
    *reinterpret_cast<uint2*>(out + at) = packed;
}

}  // namespace pie::linear
