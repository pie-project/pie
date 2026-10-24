#pragma once

// The selector's greedy walk over a request's slot rows: each row picks,
// among its k candidate ids, the one maximizing
//
//   unary[row][c] + Σ_d pred[prev][d] · (hp[row][d] ·) succ[cand][d]
//
// where prev is the previous row's pick (the request's first token for the
// first row). The rows chain through their picks, so a request is one block
// that walks its rows in order; within a row, 16 lanes score each candidate.

#include "prim/device.hip"

namespace pie::attn {

constexpr int kWalkThreads = 256;
constexpr int kWalkLanes = 16;
constexpr int kWalkMaxK = kWalkThreads / kWalkLanes;

__global__ __launch_bounds__(kWalkThreads) void selector_walk(
    const i32* __restrict__ cand, const i32* __restrict__ indptr, const float* __restrict__ unary,
    const bf16* __restrict__ hp, const i32* __restrict__ tokens, const bf16* __restrict__ pred,
    const bf16* __restrict__ succ, i32* __restrict__ picks, int k, int rank, int vocab, int has_hp,
    int first, const u32* __restrict__ win) {
    const int r = static_cast<int>(blockIdx.x);
    if (win != nullptr && blockIdx.x >= win[2]) return;
    const int c = static_cast<int>(threadIdx.x) / kWalkLanes;
    const int sub = static_cast<int>(threadIdx.x) % kWalkLanes;
    const int begin = indptr[r];
    const int end = indptr[r + 1];
    if (end <= begin) return;

    __shared__ float score[kWalkMaxK];
    __shared__ int prev_id;
    if (threadIdx.x == 0) {
        // Row 0 is the anchor: with `first` = 1 it is its own first candidate.
        if (first > 0) picks[begin] = cand[static_cast<long long>(begin) * k];
        prev_id = tokens[begin];
    }
    __syncthreads();

    for (int row = begin + first; row < end; ++row) {
        const int prev = prev_id;
        const long long at = static_cast<long long>(row) * k;
        float partial = 0.f;
        if (c < k) {
            const int id = cand[at + c];
            if (prev >= 0 && prev < vocab && id >= 0 && id < vocab) {
                const bf16* a = pred + static_cast<long long>(prev) * rank;
                const bf16* b = succ + static_cast<long long>(id) * rank;
                if (has_hp) {
                    const bf16* h = hp + static_cast<long long>(row) * rank;
                    for (int d = sub; d < rank; d += kWalkLanes) partial += to_f32(a[d]) * to_f32(h[d]) * to_f32(b[d]);
                } else {
                    for (int d = sub; d < rank; d += kWalkLanes) partial += to_f32(a[d]) * to_f32(b[d]);
                }
            }
        }
#pragma unroll
        for (int off = kWalkLanes / 2; off > 0; off >>= 1) partial += __shfl_xor(partial, off, kWalkLanes);
        if (sub == 0 && c < k) score[c] = unary[at + c] + partial;
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int j = 1; j < k; ++j)
                if (score[j] > score[best]) best = j;
            const int pick = cand[at + best];
            picks[row] = pick;
            prev_id = pick;
        }
        __syncthreads();
    }
}

}  // namespace pie::attn
