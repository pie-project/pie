#pragma once

#include <cstdint>

#include "prim/kv_rows.cuh"
#include "prim/device.cuh"
#include "prim/rope.cuh"

namespace pie::attn::index {

constexpr int kBlock = 256;

// The `topk` best of a row's `nkeys` scores, ascending, `-1` past the end:
// the threshold the counts bisect to, then the scores at or above it in order.
__device__ __forceinline__ void select_top(const float* __restrict__ frow, int nkeys, int topk,
                                           i32* __restrict__ srow, int tid)
{
    __shared__ float red[kBlock];
    float lo_l = pos_inf(), hi_l = -pos_inf();
    for (int j = tid; j < nkeys; j += kBlock) {
        lo_l = fminf(lo_l, frow[j]);
        hi_l = fmaxf(hi_l, frow[j]);
    }
    red[tid] = lo_l; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fminf(red[tid], red[tid + o]); __syncthreads(); }
    float lo = red[0];
    __syncthreads();
    red[tid] = hi_l; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    float hi = red[0];
    __syncthreads();

    __shared__ int cnt_s;
    float thr = hi;
    for (int it = 0; it < 40; ++it) {
        const float mid = 0.5f * (lo + hi);
        if (tid == 0) cnt_s = 0;
        __syncthreads();
        int c = 0;
        for (int j = tid; j < nkeys; j += kBlock) if (frow[j] >= mid) c++;
        atomicAdd(&cnt_s, c);
        __syncthreads();
        const int cnt = cnt_s;
        if (cnt > topk) lo = mid; else hi = mid;
        __syncthreads();
        thr = hi;
    }

    if (tid == 0) {
        int n = 0;
        for (int j = 0; j < nkeys && n < topk; ++j) {
            if (frow[j] >= thr) srow[n++] = j;
        }
        for (; n < topk; ++n) srow[n] = -1;
    }
}

// `select_top` by radix: the `topk` largest of `nkeys` scores, ascending by
// index, `-1` past the end. The k-th largest score's bits are found a byte a
// pass (a 256-bin histogram of the survivors, most significant byte first),
// then every score above it and the first ties at it (in index order) take
// their places by a block scan. Four passes over the scores, where the
// bisection walks forty.
__device__ __forceinline__ u32 order_bits(float v) {
    const u32 b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Block-wide exclusive prefix of `flag`, and the block's total.
__device__ __forceinline__ int block_prefix(bool flag, int tid, int* warp_sums, int& total) {
    const int lane = tid & 31, warp = tid >> 5;
    const unsigned ballot = __ballot_sync(0xffffffffu, flag);
    const int before = __popc(ballot & ((1u << lane) - 1u));
    if (lane == 0) warp_sums[warp] = __popc(ballot);
    __syncthreads();
    int base = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < kBlock / 32; ++w) {
        const int c = warp_sums[w];
        if (w < warp) base += c;
        total += c;
    }
    __syncthreads();
    return base + before;
}

__device__ __forceinline__ void select_top_radix(const float* __restrict__ frow, int nkeys,
                                                 int topk, i32* __restrict__ srow, int tid)
{
    __shared__ int hist[256];
    __shared__ int s_digit, s_need;
    __shared__ int warp_sums[kBlock / 32];
    u32 prefix = 0, mask = 0;
    int need = topk;  // how many of the survivors (bits matching prefix under mask) to take
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += kBlock) hist[i] = 0;
        __syncthreads();
        // Scores crowd a few bins (the top byte is sign and exponent): a
        // warp adds each bin once, its lanes in that bin counted together.
        for (int j0 = 0; j0 < nkeys; j0 += kBlock) {
            const int j = j0 + tid;
            const u32 u = j < nkeys ? order_bits(frow[j]) : 0u;
            const bool in = j < nkeys && (u & mask) == prefix;
            const int bin = in ? static_cast<int>((u >> shift) & 255) : -1;
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (in && (tid & 31) == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
        }
        __syncthreads();
        if (tid < 32) {
            // From the top bin down, the bin where the count reaches `need`.
            int run = 0;
            int found = -1, found_need = 0;
            // Each lane holds eight bins, highest first across the warp.
            int c[8];
            int lane_sum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c[i] = hist[255 - (tid * 8 + i)];
                lane_sum += c[i];
            }
            int incl = lane_sum;
#pragma unroll
            for (int sh = 1; sh < 32; sh <<= 1) {
                const int x = __shfl_up_sync(0xffffffffu, incl, sh);
                if (tid >= sh) incl += x;
            }
            run = incl - lane_sum;
            if (run < need && incl >= need) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (found < 0 && run + c[i] >= need) {
                        found = 255 - (tid * 8 + i);
                        found_need = need - run;
                    }
                    run += c[i];
                }
                s_digit = found;
                s_need = found_need;
            }
        }
        __syncthreads();
        prefix |= static_cast<u32>(s_digit) << shift;
        mask |= 255u << shift;
        need = s_need;
        __syncthreads();
    }
    // `prefix` is the k-th largest score's bits; `need` of its ties are taken.
    int written = 0, ties_seen = 0;
#pragma unroll 1
    for (int j0 = 0; j0 < nkeys; j0 += kBlock) {
        const int j = j0 + tid;
        const u32 u = j < nkeys ? order_bits(frow[j]) : 0u;
        const bool tie = j < nkeys && u == prefix;
        int ties;
        const int tie_rank = ties_seen + block_prefix(tie, tid, warp_sums, ties);
        ties_seen += ties;
        const bool take = j < nkeys && (u > prefix || (tie && tie_rank < need));
        int took;
        const int at = written + block_prefix(take, tid, warp_sums, took);
        if (take) srow[at] = j;
        written += took;
    }
    for (int n = written + tid; n < topk; n += kBlock) srow[n] = -1;
}

}
