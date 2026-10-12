#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

__global__ void ragged_schedule(
    const i32* __restrict__ q_indptr,
    int groups,
    int padded,
    int group_size,
    int cta_tile_q,
    int kv_chunk_sentinel,
    i32* __restrict__ request_indices,
    i32* __restrict__ qo_tile_indices,
    i32* __restrict__ kv_tile_indices,
    unsigned char* __restrict__ block_valid_mask,
    i32* __restrict__ kv_chunk_size,
    const u32* __restrict__ win)
{
    extern __shared__ int scan[];
    const int threads = static_cast<int>(blockDim.x);
    const int tid = static_cast<int>(threadIdx.x);

    int live_groups = groups;
    int first_group = 0;
    if (win != nullptr) {
        live_groups = static_cast<int>(win[2]);
        first_group = static_cast<int>(win[3]);
        if (first_group + live_groups > groups) live_groups = groups - first_group;
        if (live_groups < 0) live_groups = 0;
    }

    int total = 0;
    for (int chunk = 0; chunk < live_groups; chunk += threads) {
        const int g = chunk + tid;
        int tiles = 0;
        if (g < live_groups) {
            const int at = first_group + g;
            const int q_len = q_indptr[at + 1] - q_indptr[at];
            if (q_len > 0) {
                tiles = (q_len * group_size + cta_tile_q - 1) / cta_tile_q;
            }
        }
        scan[tid] = tiles;
        __syncthreads();

        for (int stride = 1; stride < threads; stride <<= 1) {
            const int left = tid >= stride ? scan[tid - stride] : 0;
            __syncthreads();
            scan[tid] += left;
            __syncthreads();
        }
        const int inclusive = scan[tid];
        const int chunk_total = scan[threads - 1];
        __syncthreads();
        if (g < live_groups && tiles > 0) {
            const int base = total + inclusive - tiles;
            const int at = first_group + g;
            for (int t = 0; t < tiles; ++t) {
                const int item = base + t;
                if (item >= padded) break;
                request_indices[item] = at;
                qo_tile_indices[item] = t;
                kv_tile_indices[item] = 0;
                block_valid_mask[item] = 1;
            }
        }
        total += chunk_total;
    }
    if (total > padded) total = padded;

    for (int item = total + tid; item < padded; item += threads) {
        request_indices[item] = 0;
        qo_tile_indices[item] = 0;
        kv_tile_indices[item] = 0;
        block_valid_mask[item] = 0;
    }
    if (tid == 0) *kv_chunk_size = kv_chunk_sentinel;
}

}
