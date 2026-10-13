#pragma once

// Quantized linear kernels.

#include "prelude/device.hip"

// ================================================================ MoE ====
//
// Packed expert banks (mxfp4, MLX affine) for the routed matmul. The decode
// legs stream (linear/moe_kit.hip); the grouped legs take routes ordered by
// expert from `moe_route_order`. A seat's `bases`, when given, names the
// planes in place of the resident ones; `group_hits` counts the routes read.

#include "linear/moe_kit.hip"

namespace pie::linear {

// Routes in expert order: order[offsets[e] .. offsets[e+1]) are expert e's
// routes (plane route indices), offsets[num_experts] starts the routes no
// expert takes, and `work` lists one item (e << 16 | slice) for each
// `group_routes` routes of an expert, at most `work_cap`, their count in
// offsets[num_experts + 1]. One block; dynamic LDS of 2·(num_experts + 1)
// ints.
__global__ __launch_bounds__(1024) void moe_route_order(const i32* __restrict__ routes, i32* __restrict__ order,
                                                        i32* __restrict__ offsets, i32* __restrict__ work, int work_cap,
                                                        int group_routes, int top_k, int route_count, int num_experts,
                                                        const u32* __restrict__ win) {
    extern __shared__ i32 order_smem[];
    __shared__ i32 wave_totals[1024 / 32];
    i32* counts = order_smem;
    i32* fill = counts + num_experts + 1;
    const int live = win != nullptr ? min(static_cast<int>(win[0]) * top_k, route_count) : route_count;
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

    // Prefix the counts (in place, into offsets) and the work items.
    int run = 0, items = 0;
    for (int e0 = 0; e0 <= num_experts; e0 += blockDim.x) {
        const int e = e0 + threadIdx.x;
        const int c = e <= num_experts ? counts[e] : 0;
        const int slices = e < num_experts ? (c + group_routes - 1) / group_routes : 0;
        int total, slice_total;
        const int at = run + block_scan(c, wave_totals, total);
        const int first = items + block_scan(slices, wave_totals, slice_total);
        if (e <= num_experts) {
            counts[e] = at;
            offsets[e] = at;
        }
        for (int s = 0; s < slices && first + s < work_cap; ++s) work[first + s] = e * 65536 + s;
        run += total;
        items += slice_total;
    }
    if (threadIdx.x == 0) offsets[num_experts + 1] = min(items, work_cap);
    __syncthreads();

    for (int r = threadIdx.x; r < live; r += blockDim.x) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        order[counts[e] + atomicAdd(fill + e, 1)] = base + r;
    }
}

// The decode legs: y[route] = W[expert(route)]·act[route / act_div], one
// route a block row (grid (routes, ⌈n / (4·R)⌉), 256 threads). A dead route
// (expert < 0) lands zeros.
template <class C, class T, class F>
__device__ __forceinline__ void select_stream(const T* __restrict__ act, const i32* __restrict__ routes,
                                              T* __restrict__ out, int top_k, int act_div, int n, int k,
                                              unsigned int* __restrict__ group_hits, int rows_per_wave,
                                              const u32* __restrict__ win, F&& planes) {
    extern __shared__ __align__(16) unsigned char stream_smem[];
    const int route = blockIdx.x;
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const long long plane_route = win != nullptr ? route + static_cast<long long>(win[1]) * top_k : route;
    const int expert = routes[plane_route];
    T* y = out + plane_route * n;
    if (expert < 0) {
        stream_zero(y, n, rows_per_wave);
        return;
    }
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) atomicAdd(group_hits, 1u);
    const T* bias = nullptr;
    const typename C::Planes p = planes(expert, bias);
    stream_rows<C, T>(p, act + plane_route / act_div * k, y, bias, n, k, rows_per_wave, stream_smem);
}

template <class T>
__global__ __launch_bounds__(256) void moe_matmul_select_mxfp4(
    const T* __restrict__ act, const i32* __restrict__ routes, const u8* __restrict__ codes,
    const u8* __restrict__ scales, const T* __restrict__ bias, T* __restrict__ out, int top_k, int act_div, int n,
    int k, const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits, int rows_per_wave,
    const u32* __restrict__ win) {
    select_stream<StreamMxfp4<T>, T>(
        act, routes, out, top_k, act_div, n, k, group_hits, rows_per_wave, win, [&](int e, const T*& b) {
            const u8* c = bases != nullptr ? bases->codes : codes;
            const u8* s = bases != nullptr ? bases->scales : scales;
            const long long rows = static_cast<long long>(e) * n;
            if (bias != nullptr) b = bias + rows;
            return typename StreamMxfp4<T>::Planes{c + rows * (k / 2), s + rows * (k / 32)};
        });
}

template <class T, int BITS, int GROUP>
__global__ __launch_bounds__(256) void moe_matmul_select_mlxu4(
    const T* __restrict__ act, const i32* __restrict__ routes, const u8* __restrict__ codes,
    const u8* __restrict__ scales, const u8* __restrict__ biases, T* __restrict__ out, int top_k, int act_div, int n,
    int k, const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits, int rows_per_wave,
    const u32* __restrict__ win) {
    using C = StreamAffine<T, BITS, GROUP>;
    select_stream<C, T>(act, routes, out, top_k, act_div, n, k, group_hits, rows_per_wave, win,
                        [&](int e, const T*&) {
                            const u8* c = bases != nullptr ? bases->codes : codes;
                            const u8* s = bases != nullptr ? bases->scales : scales;
                            const u8* b = bases != nullptr ? bases->biases : biases;
                            const long long rows = static_cast<long long>(e) * n;
                            const long long factors = rows * (k / GROUP);
                            return typename C::Planes{reinterpret_cast<const u32*>(c) + rows * (k * BITS / 32),
                                                      reinterpret_cast<const bf16*>(s) + factors,
                                                      reinterpret_cast<const bf16*>(b) + factors};
                        });
}

// The grouped legs on the matrix core (bf16): one work item of up to MT
// routes a block, 128 weight rows (grid (work cap, ⌈n / 128⌉), 256 threads).
template <int MT>
__global__ __launch_bounds__(256) void moe_matmul_select_mxfp4_wmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const bf16* __restrict__ bias, bf16* __restrict__ out, int act_div, int n, int k, int num_experts,
    const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* s = bases != nullptr ? bases->scales : scales;
    const long long rows = static_cast<long long>(item.expert) * n;
    grouped_tile<GroupMxfp4, bf16, MT>(act, order, item.begin, item.count, {c + rows * (k / 2), s + rows * (k / 32)},
                                   bias != nullptr ? bias + rows : nullptr, out, act_div, n, k);
}

template <int BITS, int GROUP, int MT>
__global__ __launch_bounds__(256) void moe_matmul_select_mlxu4_wmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const u8* __restrict__ biases, bf16* __restrict__ out, int act_div, int n, int k, int num_experts,
    const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* s = bases != nullptr ? bases->scales : scales;
    const u8* b = bases != nullptr ? bases->biases : biases;
    const long long rows = static_cast<long long>(item.expert) * n;
    const long long factors = rows * (k / GROUP);
    grouped_tile<GroupAffine<BITS, GROUP>, bf16, MT>(
        act, order, item.begin, item.count,
        {reinterpret_cast<const u32*>(c) + rows * (k * BITS / 32), reinterpret_cast<const bf16*>(s) + factors,
         reinterpret_cast<const bf16*>(b) + factors},
        nullptr, out, act_div, n, k);
}

// The grouped legs off the matrix core, any T, f32 throughout: one expert a
// block (grid (experts, ⌈n / 128⌉), 256 threads), its routes 16 at a time
// against a 128 x 64 tile of the dequantized weight in LDS. Each thread
// owns two weight rows and four of the batch's routes.
constexpr int kSimtTileN = 128;
constexpr int kSimtTileK = 64;
constexpr int kSimtBatch = 16;

__constant__ float kFp4Values[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                     -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// `dequant(row, kk, w)` writes codes kk .. kk + 7 of weight row `row`.
template <class T, class D>
__device__ __forceinline__ void grouped_simt(const T* __restrict__ act, const i32* __restrict__ order, int begin,
                                             int end, const T* __restrict__ bias, T* __restrict__ out, int act_div,
                                             int n, int k, D&& dequant) {
    __shared__ float w_tile[kSimtTileN][kSimtTileK + 1];
    __shared__ float x_tile[kSimtBatch][kSimtTileK];
    const int row0 = blockIdx.y * kSimtTileN;
    const int tid = threadIdx.x;
    const int pair = tid % (kSimtTileN / 2), slot = tid / (kSimtTileN / 2);
    const int ra = 2 * pair, rb = ra + 1;
    const float bias_a = bias != nullptr && row0 + ra < n ? to_f32(bias[row0 + ra]) : 0.f;
    const float bias_b = bias != nullptr && row0 + rb < n ? to_f32(bias[row0 + rb]) : 0.f;
    for (int b0 = begin; b0 < end; b0 += kSimtBatch) {
        const int batch = min(kSimtBatch, end - b0);
        float acc[2][kSimtBatch / 4] = {};
        for (int k0 = 0; k0 < k; k0 += kSimtTileK) {
            for (int idx = tid; idx < kSimtTileN * kSimtTileK / 8; idx += blockDim.x) {
                const int r = idx / (kSimtTileK / 8), kk = (idx % (kSimtTileK / 8)) * 8;
                float w[8] = {};
                if (k0 + kk < k) dequant(min(row0 + r, n - 1), k0 + kk, w);
#pragma unroll
                for (int j = 0; j < 8; ++j) w_tile[r][kk + j] = w[j];
            }
            for (int idx = tid; idx < batch * kSimtTileK; idx += blockDim.x) {
                const int t = idx / kSimtTileK, kk = idx % kSimtTileK;
                const T* x = act + static_cast<long long>(order[b0 + t] / act_div) * k;
                x_tile[t][kk] = k0 + kk < k ? to_f32(x[k0 + kk]) : 0.f;
            }
            __syncthreads();
            for (int kk = 0; kk < kSimtTileK; ++kk) {
                const float wa = w_tile[ra][kk], wb = w_tile[rb][kk];
#pragma unroll
                for (int j = 0; j < kSimtBatch / 4; ++j) {
                    const float xv = x_tile[slot + 4 * j][kk];
                    acc[0][j] = fmaf(wa, xv, acc[0][j]);
                    acc[1][j] = fmaf(wb, xv, acc[1][j]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < kSimtBatch / 4; ++j) {
            const int t = slot + 4 * j;
            if (t >= batch) continue;
            T* o = out + static_cast<long long>(order[b0 + t]) * n + row0;
            if (row0 + ra < n) o[ra] = from_f32<T>(acc[0][j] + bias_a);
            if (row0 + rb < n) o[rb] = from_f32<T>(acc[1][j] + bias_b);
        }
    }
}

template <class T>
__global__ __launch_bounds__(256) void moe_matmul_select_mxfp4_grouped(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const u8* __restrict__ codes, const u8* __restrict__ scales, const T* __restrict__ bias, T* __restrict__ out,
    int act_div, int n, int k, int num_experts, const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits) {
    const int expert = blockIdx.x;
    if (expert >= num_experts || static_cast<int>(blockIdx.y) * kSimtTileN >= n) return;
    const int begin = offsets[expert], end = offsets[expert + 1];
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));
    const long long rows = static_cast<long long>(expert) * n;
    const u32* w32 = reinterpret_cast<const u32*>((bases != nullptr ? bases->codes : codes) + rows * (k / 2));
    const u8* s8 = (bases != nullptr ? bases->scales : scales) + rows * (k / 32);
    grouped_simt<T>(act, order, begin, end, bias != nullptr ? bias + rows : nullptr, out, act_div, n, k,
                    [&](int row, int kk, float* w) {
                        const float sc = mxfp4_scale(s8[static_cast<long long>(row) * (k / 32) + kk / 32]);
                        const u32 word = w32[static_cast<long long>(row) * (k / 8) + kk / 8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) w[j] = kFp4Values[(word >> (4 * j)) & 15u] * sc;
                    });
}

template <class T, int BITS, int GROUP>
__global__ __launch_bounds__(256) void moe_matmul_select_mlxu4_grouped(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const u8* __restrict__ codes, const u8* __restrict__ scales, const u8* __restrict__ biases, T* __restrict__ out,
    int act_div, int n, int k, int num_experts, const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits) {
    const int expert = blockIdx.x;
    if (expert >= num_experts || static_cast<int>(blockIdx.y) * kSimtTileN >= n) return;
    const int begin = offsets[expert], end = offsets[expert + 1];
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));
    const long long rows = static_cast<long long>(expert) * n;
    const long long factors = rows * (k / GROUP);
    const u32* w32 = reinterpret_cast<const u32*>(bases != nullptr ? bases->codes : codes) + rows * (k * BITS / 32);
    const bf16* s16 = reinterpret_cast<const bf16*>(bases != nullptr ? bases->scales : scales) + factors;
    const bf16* b16 = reinterpret_cast<const bf16*>(bases != nullptr ? bases->biases : biases) + factors;
    grouped_simt<T>(act, order, begin, end, nullptr, out, act_div, n, k, [&](int row, int kk, float* w) {
        const long long f = static_cast<long long>(row) * (k / GROUP) + kk / GROUP;
        const float sv = to_f32(s16[f]), bv = to_f32(b16[f]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const long long bit = (static_cast<long long>(row) * k + kk + j) * BITS;
            const u32 code = (w32[bit / 32] >> (bit % 32)) & ((1u << BITS) - 1u);
            w[j] = fmaf(static_cast<float>(code), sv, bv);
        }
    });
}

}  // namespace pie::linear
