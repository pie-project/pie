#pragma once


#include "prelude/device.cuh"

namespace pie::attn {

template <class T>
using Elem = ::pie::Elem<T>;

constexpr unsigned WALK_THREADS = 256;
constexpr unsigned WALK_LANES = 16;
constexpr unsigned WALK_MAX_K = WALK_THREADS / WALK_LANES;

template <class T>
__global__ void __launch_bounds__(WALK_THREADS) selector_walk(
    const i32* __restrict__ cand,
    const i32* __restrict__ indptr,
    const float* __restrict__ unary,
    const T* __restrict__ hp,
    const i32* __restrict__ tokens,
    const T* __restrict__ pred,
    const T* __restrict__ succ,
    i32* __restrict__ picks,
    int k,
    int rank,
    int vocab,
    int has_hp,
    int first,
    const u32* __restrict__ win)
{
    const int r = static_cast<int>(blockIdx.x);
    if (win != nullptr && blockIdx.x >= win[2]) return;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid / WALK_LANES;
    const unsigned lane = tid % WALK_LANES;
    const int begin = indptr[r];
    const int end = indptr[r + 1];
    if (end <= begin) return;

    __shared__ float score[WALK_MAX_K];
    __shared__ int prev_id;
    if (tid == 0) {

        if (first > 0) {
            picks[begin] = cand[static_cast<size_t>(begin) * static_cast<size_t>(k)];
        }
        prev_id = tokens[begin];
    }
    __syncthreads();

    for (int row = begin + first; row < end; ++row) {
        const int my_prev = prev_id;
        float partial = 0.0f;
        if (static_cast<int>(c) < k) {
            const int cid = cand[static_cast<size_t>(row) * static_cast<size_t>(k) + c];
            const bool live = my_prev >= 0 && my_prev < vocab && cid >= 0 && cid < vocab;
            if (live) {
                const T* a = pred + static_cast<size_t>(my_prev) * static_cast<size_t>(rank);
                const T* b = succ + static_cast<size_t>(cid) * static_cast<size_t>(rank);
                if (has_hp) {
                    const T* h = hp + static_cast<size_t>(row) * static_cast<size_t>(rank);
                    for (int d = static_cast<int>(lane); d < rank; d += static_cast<int>(WALK_LANES)) {
                        partial += Elem<T>::to_f32(a[d]) * Elem<T>::to_f32(h[d]) * Elem<T>::to_f32(b[d]);
                    }
                } else {
                    for (int d = static_cast<int>(lane); d < rank; d += static_cast<int>(WALK_LANES)) {
                        partial += Elem<T>::to_f32(a[d]) * Elem<T>::to_f32(b[d]);
                    }
                }
            }
        }

        partial += __shfl_xor_sync(0xffffffffu, partial, 8);
        partial += __shfl_xor_sync(0xffffffffu, partial, 4);
        partial += __shfl_xor_sync(0xffffffffu, partial, 2);
        partial += __shfl_xor_sync(0xffffffffu, partial, 1);
        if (lane == 0 && static_cast<int>(c) < k) {
            score[c] = unary[static_cast<size_t>(row) * static_cast<size_t>(k) + c] + partial;
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            float best_v = score[0];
            for (int j = 1; j < k; ++j) {
                if (score[j] > best_v) {
                    best_v = score[j];
                    best = j;
                }
            }
            const int pick = cand[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(best)];
            picks[row] = pick;
            prev_id = pick;
        }
        __syncthreads();
    }
}

}
