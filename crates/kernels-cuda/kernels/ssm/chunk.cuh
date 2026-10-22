#pragma once

#include "ssm/state.cuh"

namespace pie::ssm {

constexpr int GDN_CH = 64;

struct GdnLane {
    int t0;
    int T;
    int rl;
    int slot;
};

__device__ __forceinline__ GdnLane gdn_lane(
    int r,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ slot_ids,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    const u32* __restrict__ win)
{
    GdnLane lane;
    lane.rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    lane.t0 = static_cast<int>(qo_indptr[r]);
    lane.T = static_cast<int>(qo_indptr[r + 1]) - lane.t0;
    if (begin_at != nullptr) {
        int b = begin_at[lane.rl];
        if (b > lane.T) b = lane.T;
        if (b > 0) { lane.t0 += b; lane.T -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[lane.rl];
        if (c < lane.T) lane.T = c;
    }
    lane.slot = lane.T > 0 ? slot_ids[lane.rl] : -1;
    return lane;
}

template <int CH = GDN_CH>
__device__ __forceinline__ int gdn_chunk_base(
    int r,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ begin_at,
    const u32* __restrict__ win)
{
    int t0 = static_cast<int>(qo_indptr[r]);
    if (begin_at != nullptr) {
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const int T = static_cast<int>(qo_indptr[r + 1]) - t0;
        int b = begin_at[rl];
        if (b > T) b = T;
        if (b > 0) t0 += b;
    }
    return t0 / CH + r;
}

template <int CH = GDN_CH>
__device__ __forceinline__ int gdn_slot_lane(
    int s, int lanes,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ begin_at,
    const u32* __restrict__ win)
{
    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (gdn_chunk_base<CH>(mid, qo_indptr, begin_at, win) <= s) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// A snapshot row is a chunk boundary: the rows between two are chunked from
// the first, as a pass cut there would chunk them, so a request takes at most
// one more chunk slot a snapshot (`snap[rl]` of them before lane `rl`'s).
__device__ __forceinline__ int gdn_chunk_base_snap(
    int r,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ begin_at,
    const int* __restrict__ snap,
    const u32* __restrict__ win)
{
    const int base = gdn_chunk_base(r, qo_indptr, begin_at, win);
    if (snap == nullptr) return base;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    return base + snap[rl];
}

__device__ __forceinline__ int gdn_slot_lane_snap(
    int s, int lanes,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ begin_at,
    const int* __restrict__ snap,
    const u32* __restrict__ win)
{
    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (gdn_chunk_base_snap(mid, qo_indptr, begin_at, snap, win) <= s) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// Chunk j of a lane: its first row, its rows (none past the last chunk),
// and the snapshot its end lands (-1: none).
struct GdnChunk {
    int ta;
    int L;
    int snap;
};

__device__ __forceinline__ GdnChunk gdn_chunk(
    const GdnLane& ln, Snaps sn, int j, const int* __restrict__ snap_rows)
{
    int a = 0;
    for (int e = sn.e0; e < sn.e1; ++e) {
        const int b = snap_rows[e];
        if (b >= ln.T) break;
        const int n = (b - a + GDN_CH - 1) / GDN_CH;
        if (j < n) {
            return {ln.t0 + a + j * GDN_CH, min(GDN_CH, b - a - j * GDN_CH), j == n - 1 ? e : -1};
        }
        j -= n;
        a = b;
    }
    return {ln.t0 + a + j * GDN_CH, min(GDN_CH, ln.T - a - j * GDN_CH), -1};
}

__device__ __forceinline__ int gdn_chunks(
    const GdnLane& ln, Snaps sn, const int* __restrict__ snap_rows)
{
    int a = 0, n = 0;
    for (int e = sn.e0; e < sn.e1 && snap_rows[e] < ln.T; ++e) {
        n += (snap_rows[e] - a + GDN_CH - 1) / GDN_CH;
        a = snap_rows[e];
    }
    return n + (ln.T - a + GDN_CH - 1) / GDN_CH;
}

}
