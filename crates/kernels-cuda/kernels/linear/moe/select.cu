#include "linear/codec.cuh"
#include "prim/device.cuh"
#ifdef __CUDACC_RTC__

#include "prim/mma_sm80.cuh"
#else

#include <cuda_bf16.h>
#include <mma.h>
#endif
#include "linear/moe_w4.cuh"
#include "linear/w8a16.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

[[maybe_unused]] constexpr int kMoeVecWidth = 8;

constexpr int kGemvWarps = 4;

template <class T>
__device__ __forceinline__ const T* moe_expert_base(
    const T* __restrict__ weight_base,
    const T* const* __restrict__ expert_table,
    int expert,
    long long expert_stride)
{
    return expert_table != nullptr
        ? expert_table[expert]
        : weight_base + static_cast<long long>(expert) * expert_stride;
}

__device__ __forceinline__ void moe_note_expert(
    unsigned int* __restrict__ expert_hits, int expert)
{
    if (expert_hits != nullptr) {
        atomicAdd(&expert_hits[expert], 1u);
    }
}

// `y[route] = x[route or its token] . W[routes[route]]^T`, a warp `kRows`
// weight rows: a lane's activations are read once and meet every row, and
// the rows' loads issue together. A route run past grid.y's 65535 rows
// continues on grid.z.
template <class T, bool ActByToken, int kRows>
__device__ __forceinline__ void moe_matmul_select_gemv_body(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride,
    const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits,
    const u32* __restrict__ win,
    int route_count)
{
    const int route = static_cast<int>(blockIdx.y + blockIdx.z * gridDim.y);
    if (route >= route_count) return;
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const int plane_route = win != nullptr
        ? route + static_cast<int>(win[1]) * top_k
        : route;
    const int row0 = (static_cast<int>(blockIdx.x) * kGemvWarps + static_cast<int>(threadIdx.y)) * kRows;
    if (row0 >= N) return;
    const int lane = threadIdx.x;

    const int expert = topk_idx[plane_route];
    if (expert < 0) {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                if (row0 + r < N) out[(long long)plane_route * N + row0 + r] = Elem<T>::from_f32(0.f);
        }
        return;
    }
    if (blockIdx.x == 0 && threadIdx.y == 0 && lane == 0) {
        moe_note_expert(expert_hits, expert);
    }
    const T* w = moe_expert_base(weight_base, expert_table, expert, expert_stride);
    const float4* w4[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r)
        w4[r] = reinterpret_cast<const float4*>(w + (long long)min(row0 + r, N - 1) * K);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + (long long)(ActByToken ? plane_route / top_k : plane_route) * K);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
    const int vec = K / kMoeVecWidth;
    for (int i = lane; i < vec; i += 32) {
        float4 wv[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) wv[r] = w4[r][i];
        const float4 xv = x4[i];
        const T* xb = reinterpret_cast<const T*>(&xv);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const T* wb = reinterpret_cast<const T*>(&wv[r]);
#pragma unroll
            for (int j = 0; j < kMoeVecWidth; ++j) {
                acc[r] += Elem<T>::to_f32(wb[j]) * Elem<T>::to_f32(xb[j]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            if (row0 + r < N) out[(long long)plane_route * N + row0 + r] = Elem<T>::from_f32(acc[r]);
    }
}

template <class T, int kRows>
__global__ void moe_matmul_select_gemv_by_token(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride,
    const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits,
    int route_count,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    moe_matmul_select_gemv_body<T, true, kRows>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride,
        expert_table, expert_hits, win, route_count);
}

template <class T, int kRows>
__global__ void moe_matmul_select_gemv_by_route(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride,
    const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits,
    int route_count,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    moe_matmul_select_gemv_body<T, false, kRows>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride,
        expert_table, expert_hits, win, route_count);
}

/// The same walk over a bf16 bank (`[experts, n, k]` bf16): no widening, 32
/// columns of K a lane's 16-byte load a row, the mma's step q (of two) lane
/// c element j reading column 8c + 4q + j.
// `kTiles` m16 tiles a warp (1 halves a CTA's rows, doubling a decode's
// CTAs; a GLU's warp is always its 16 gate rows and their up rows), and K
// split across `kKSplit` groups of four warps, folded through shared memory
// before the first lands them.
template <int kNb, int kDepth, int kGlu, int kTiles = 2, int kKSplit = 1>
__global__ void __launch_bounds__(128 * kKSplit) moe_select_bf16_held(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const bf16* __restrict__ codes,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha)
{
    static_assert(kGlu == 0 || kTiles == 2, "a GLU warp is a gate tile and its up tile");
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

    const int warp = (static_cast<int>(threadIdx.x) >> 5) & 3;
    const int part = static_cast<int>(threadIdx.x) >> 7;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(blockIdx.y) * kRowsCta + warp * kRowsWarp;
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    if (kKSplit == 1 && row_base >= span) return;
    const bool rows_live = row_base < span;

    const int chunks = k >> 5;
    const bf16* w_e = codes + static_cast<long long>(expert) * n * k;
    const bf16* wrow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(tile_row(t) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + static_cast<long long>(r) * k + c * 8;
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 8
            : nullptr;
    }

    struct Chunk {
        uint4 w[kTiles][2];
        uint4 x[kNb];
    };
    auto load = [&](int ch, Chunk& cc) {
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) cc.w[t][h] = w4mma::ld_act(wrow[t][h] + ch * 32);
#pragma unroll
        for (int b = 0; b < kNb; ++b)
            cc.x[b] = xrow[b] != nullptr ? w4mma::ld_act(xrow[b] + ch * 32) : make_uint4(0, 0, 0, 0);
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

    // This group's chunks: part, part + kKSplit, ...
    const int mine = rows_live && part < chunks ? (chunks - part + kKSplit - 1) / kKSplit : 0;
    Chunk ring[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
        if (d < mine) load(part + d * kKSplit, ring[d]);
    for (int i0 = 0; i0 < mine; i0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int i = i0 + d;
            if (i < mine) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
#pragma unroll
                    for (int b = 0; b < kNb; ++b) {
                        const unsigned b0 = q ? ring[d].x[b].z : ring[d].x[b].x;
                        const unsigned b1 = q ? ring[d].x[b].w : ring[d].x[b].y;
#pragma unroll
                        for (int t = 0; t < kTiles; ++t) {
                            const uint4& lo = ring[d].w[t][0];
                            const uint4& hi = ring[d].w[t][1];
                            mma_m16n8k16(acc[t][b], {q ? lo.z : lo.x, q ? hi.z : hi.x, q ? lo.w : lo.y, q ? hi.w : hi.y}, {b0, b1});
                        }
                    }
                }
                if (i + kDepth < mine) load(part + (i + kDepth) * kKSplit, ring[d]);
            }
        }
    }
    if (!w4mma::fold_ksplit<kKSplit>(acc, part, warp, lane, rows_live)) return;

    w4mma::w4_land<kGlu, kTiles, kNb>(acc, [](int) { return 1.f; }, nullptr, order, begin, count,
                                      out, n, inter, row_base, g, c, limit, galpha);
}

// One block a route and `kRowsT` rows a warp, the route's expert read once a
// route: the decode arm. A null `routes` reads one expert, route `r` being
// activation row `r`, which is the dense product. With kLean (A100's
// profile) one 32-code chunk a row in flight a lane and the codes widened by
// integer ops as `moe_matmul_select_mxfp4`'s lean arm widens them.
template <class T, int kRowsT, bool kLean = false>
__global__ void moe_matmul_select_nvfp4(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const float* __restrict__ tensor_scales,
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
    const int expert = routes != nullptr ? routes[plane_route] : 0;
    T* y = out + static_cast<long long>(plane_route) * n;
    if (expert < 0) {
        if (lane_id == 0) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                if (row0 + r < n) y[row0 + r] = Elem<T>::from_f32(0.f);
        }
        return;
    }

    moe_note_group(group_hits);

    // A 16-byte load is 32 codes, two groups of a row.
    const int chunks = k / 32;
    const int groups = k / 16;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const float* tensor_at = tensor_scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        tensor_at = reinterpret_cast<const float*>(seat.biases);
    }

    const uint4* wq = reinterpret_cast<const uint4*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        scales_at + static_cast<long long>(expert) * n * groups);
    const float* g = tensor_at + static_cast<long long>(expert) * n;

    const T* x = act + static_cast<long long>(plane_route / act_div) * k;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    constexpr int kU = kLean ? 1 : 2;
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int c0 = lane_id; c0 < chunks; c0 += 32 * kU) {
        uint4 ww[kU][kRows];
        unsigned short sc[kU][kRows];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = min(c0 + 32 * u, chunks - 1);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const long long at = static_cast<long long>(row_of[r]) * chunks + c;
                ww[u][r] = wq[at];
                sc[u][r] = s2[at];
            }
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = c0 + 32 * u;
            if (c >= chunks) break;
            float part[2][kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[0][r] = part[1][r] = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xq = x4[c * 4 + q];
                const T* xe = reinterpret_cast<const T*>(&xq);
                float xv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) xv[j] = Elem<T>::to_f32(xe[j]) * (kLean ? 0x1p100f : 1.f);
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if constexpr (kLean) {
                        part[q >> 1][r] = e2m1x8_dot_lean((&ww[u][r].x)[q], xv, part[q >> 1][r]);
                    } else {
                        __half2 qd[4];
                        mxfp4_unpack8((&ww[u][r].x)[q], qd);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f = __half22float2(qd[j]);
                            part[q >> 1][r] = fmaf(f.x, xv[2 * j], part[q >> 1][r]);
                            part[q >> 1][r] = fmaf(f.y, xv[2 * j + 1], part[q >> 1][r]);
                        }
                    }
                }
            }
            // The lean arm's products are 2^-26 short: the scale takes it back.
            constexpr float kBack = kLean ? 0x1p26f : 1.f;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                acc[r] = fmaf(part[0][r], nvfp4_scale(static_cast<u8>(sc[u][r] & 0xFFu)) * kBack, acc[r]);
                acc[r] = fmaf(part[1][r], nvfp4_scale(static_cast<u8>(sc[u][r] >> 8)) * kBack, acc[r]);
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
            if (row >= n) break;
            y[row] = Elem<T>::from_f32(acc[r] * g[row]);
        }
    }
}

constexpr int kBlock = 256;

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

}
