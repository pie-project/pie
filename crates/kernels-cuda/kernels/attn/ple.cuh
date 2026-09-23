#pragma once

#include "prelude/device.cuh"

namespace pie::attn {


constexpr int PLE_MAX_NGRAM = 4;
constexpr int PLE_MAX_HEADS = 32;

struct PleHash {
    unsigned long long mults[PLE_MAX_NGRAM];
    unsigned long long primes[PLE_MAX_HEADS];
    unsigned long long offsets[PLE_MAX_HEADS];
    int ngram;
    int heads;
    int heads_per_ngram;
    int eos;
};

__device__ __forceinline__ void ple_hash_row(
    const PleHash& h, const int* window, int* out)
{
    for (int order = 2; order <= h.ngram; ++order) {
        unsigned long long mixed = (unsigned long long)window[0] * h.mults[0];
        for (int p = 1; p < order; ++p) {
            mixed ^= (unsigned long long)window[p] * h.mults[p];
        }
        const int base = (order - 2) * h.heads_per_ngram;
        for (int k = 0; k < h.heads_per_ngram; ++k) {
            const int head = base + k;
            const unsigned long long id = mixed % h.primes[head] + h.offsets[head];
            out[head] = (int)id;
        }
    }
}

__device__ __forceinline__ void ple_mask_window(
    const PleHash& h, int* window)
{
    bool crossed = false;
    for (int p = 1; p < h.ngram; ++p) {
        if (crossed) window[p] = h.eos;
        if (window[p] == h.eos) crossed = true;
    }
}

// Engram hashes tokenizer-compressed ids: the window holds raw ids (so the
// state and the eos barrier stay in the tokenizer's id space) and is mapped
// after masking, so the pre-start padding maps like every other id.
__device__ __forceinline__ void ple_map_window(
    const PleHash& h, const int* __restrict__ map, int* window)
{
    if (map == nullptr) return;
    for (int p = 0; p < h.ngram; ++p) window[p] = map[window[p]];
}

__global__ void ple_ngram_ids_update(
    const int* __restrict__ ids,
    int* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    int* __restrict__ ngram_ids,
    int rows,
    PleHash h,
    const int* __restrict__ map,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int slot = slot_ids[r];
    if (slot < 0) return;
    int* state = state_base + (long long)slot * slot_stride_elems;

    const int span = h.ngram - 1;
    int window[PLE_MAX_NGRAM];
    window[0] = ids[r_row];
    for (int p = 1; p <= span; ++p) {
        const int cell = state[span - p];
        window[p] = cell == 0 ? h.eos : cell - 1;
    }
    ple_mask_window(h, window);
    ple_map_window(h, map, window);

    int out[PLE_MAX_HEADS];
    ple_hash_row(h, window, out);
    for (int k = 0; k < h.heads; ++k) {
        ngram_ids[(long long)r_row * h.heads + k] = out[k];
    }

    for (int p = 0; p < span - 1; ++p) state[p] = state[p + 1];
    state[span - 1] = ids[r_row] + 1;
}

__global__ void ple_ngram_ids_chunked(
    const int* __restrict__ ids,
    int* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int* __restrict__ ngram_ids,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    PleHash h,
    const int* __restrict__ map,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;

    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;

    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = (int)qo_indptr[r] + row0;
    int Nr = (int)qo_indptr[r + 1] - (int)qo_indptr[r];

    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > Nr) b = Nr;
        if (b > 0) { t0 += b; Nr -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[rl];
        if (c < Nr) Nr = c;
    }
    if (Nr <= 0) return;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    int* state = state_base + (long long)slot * slot_stride_elems;

    const int span = h.ngram - 1;
    const int tid = threadIdx.x;

    for (int t = tid; t < Nr; t += blockDim.x) {
        int window[PLE_MAX_NGRAM];
        window[0] = ids[t0 + t];
        for (int p = 1; p <= span; ++p) {
            if (t - p >= 0) {
                window[p] = ids[t0 + t - p];
            } else {
                const int cell = state[span - (p - t)];
                window[p] = cell == 0 ? h.eos : cell - 1;
            }
        }
        ple_mask_window(h, window);
        ple_map_window(h, map, window);
        int out[PLE_MAX_HEADS];
        ple_hash_row(h, window, out);
        for (int k = 0; k < h.heads; ++k) {
            ngram_ids[(long long)(t0 + t) * h.heads + k] = out[k];
        }
    }

    __syncthreads();

    if (write_state &&
        (write_state_mask == nullptr || write_state_mask[rl] != 0) &&
        tid == 0) {

        int next[PLE_MAX_NGRAM];
        for (int p = 0; p < span; ++p) {
            const int src_t = Nr - span + p;
            next[p] = src_t >= 0 ? ids[t0 + src_t] + 1 : state[p + Nr];
        }
        for (int p = 0; p < span; ++p) state[p] = next[p];
    }
}

}
