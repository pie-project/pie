#pragma once


#include "prelude/device.cuh"

namespace pie::layout {

__device__ __forceinline__ float topk_load(bf16 v) { return ::pie::Elem<bf16>::to_f32(v); }
__device__ __forceinline__ float topk_load(float v) { return v; }

constexpr unsigned TOPK_NONE = 0xFFFFFFFFu;

__device__ __forceinline__ bool topk_beats(float av, unsigned ai, float bv, unsigned bi)
{
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ void warp_best(float& v, unsigned& i)
{
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, i, off);
        if (oi != TOPK_NONE && (i == TOPK_NONE || topk_beats(ov, oi, v, i))) {
            v = ov;
            i = oi;
        }
    }
}

template <class T>
__global__ void argmax_rows(
    const T* __restrict__ x,
    int* __restrict__ y,
    int width,
    int depth,
    int column,
    const u32* __restrict__ win)
{
    const unsigned row = blockIdx.x;
    if (win != nullptr && row >= win[0]) return;
    const T* src = x + static_cast<size_t>(row) * static_cast<size_t>(width);

    float best = neg_inf();
    unsigned best_i = TOPK_NONE;
    for (unsigned c = threadIdx.x; c < static_cast<unsigned>(width); c += blockDim.x) {
        const float v = topk_load(src[c]);
        if (!isnan(v) && (best_i == TOPK_NONE || topk_beats(v, c, best, best_i))) {
            best = v;
            best_i = c;
        }
    }

    __shared__ float part_v[32];
    __shared__ unsigned part_i[32];
    warp_best(best, best_i);
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    if (lane == 0) {
        part_v[warp] = best;
        part_i[warp] = best_i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned warps = (blockDim.x + 31u) >> 5;
        float top = neg_inf();
        unsigned top_i = TOPK_NONE;
        for (unsigned w = 0; w < warps; ++w) {
            const unsigned i = part_i[w];
            if (i != TOPK_NONE && (top_i == TOPK_NONE || topk_beats(part_v[w], i, top, top_i))) {
                top = part_v[w];
                top_i = i;
            }
        }
        y[static_cast<size_t>(row) * static_cast<size_t>(depth) + static_cast<size_t>(column)] =
            static_cast<int>(top_i == TOPK_NONE ? 0u : top_i);
    }
}

constexpr unsigned TOPK_THREADS = 128;
constexpr unsigned TOPK_WARPS = TOPK_THREADS / 32;

template <class T, int K>
__global__ void __launch_bounds__(TOPK_THREADS) topk_rows(
    const T* __restrict__ x,
    float* __restrict__ values,
    int* __restrict__ indices,
    int width,
    const u32* __restrict__ win)
{
    const unsigned row = blockIdx.x;
    if (win != nullptr && row >= win[0]) return;
    const unsigned lid = threadIdx.x;
    const T* src = x + static_cast<size_t>(row) * static_cast<size_t>(width);

    float lv[K];
    unsigned li[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        lv[j] = neg_inf();
        li[j] = TOPK_NONE;
    }
    for (unsigned c = lid; c < static_cast<unsigned>(width); c += TOPK_THREADS) {
        const float v = topk_load(src[c]);
        if (isnan(v)) continue;
        if (li[K - 1] != TOPK_NONE && !topk_beats(v, c, lv[K - 1], li[K - 1])) continue;

        int at = K - 1;
        while (at > 0 && (li[at - 1] == TOPK_NONE || topk_beats(v, c, lv[at - 1], li[at - 1]))) {
            lv[at] = lv[at - 1];
            li[at] = li[at - 1];
            --at;
        }
        lv[at] = v;
        li[at] = c;
    }

    __shared__ float lists_v[TOPK_THREADS * K];
    __shared__ unsigned lists_i[TOPK_THREADS * K];
    __shared__ float part_v[TOPK_WARPS];
    __shared__ unsigned part_i[TOPK_WARPS];
    __shared__ unsigned part_owner[TOPK_WARPS];
    __shared__ unsigned winner;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        lists_v[lid * K + j] = lv[j];
        lists_i[lid * K + j] = li[j];
    }
    unsigned head = 0;
    const unsigned warp = lid >> 5;
    const unsigned lane = lid & 31;
    __syncthreads();

    for (int j = 0; j < K; ++j) {

        float hv = head < static_cast<unsigned>(K) ? lists_v[lid * K + head] : neg_inf();
        unsigned hi = head < static_cast<unsigned>(K) ? lists_i[lid * K + head] : TOPK_NONE;
        const unsigned my_i = hi;
        warp_best(hv, hi);

        unsigned owner = (my_i != TOPK_NONE && my_i == hi) ? lid : TOPK_NONE;
        for (int off = 16; off > 0; off >>= 1) {
            owner = min(owner, __shfl_xor_sync(0xffffffffu, owner, off));
        }
        if (lane == 0) {
            part_v[warp] = hv;
            part_i[warp] = hi;
            part_owner[warp] = owner;
        }
        __syncthreads();
        if (lid == 0) {
            float top = neg_inf();
            unsigned top_i = TOPK_NONE;
            unsigned top_owner = TOPK_NONE;
            for (unsigned w = 0; w < TOPK_WARPS; ++w) {
                if (part_i[w] != TOPK_NONE
                    && (top_i == TOPK_NONE || topk_beats(part_v[w], part_i[w], top, top_i))) {
                    top = part_v[w];
                    top_i = part_i[w];
                    top_owner = part_owner[w];
                }
            }
            values[static_cast<size_t>(row) * K + j] = top_i == TOPK_NONE ? 0.0f : top;
            indices[static_cast<size_t>(row) * K + j] = static_cast<int>(top_i == TOPK_NONE ? 0u : top_i);
            winner = top_owner;
        }
        __syncthreads();
        if (winner == lid) {
            ++head;
        }
    }
}

}
