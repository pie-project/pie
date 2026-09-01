#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

// The PLE n-gram hasher (qwen4): every token's hashed n-gram table rows,
// one column per head. The hash is the reference's own — token ids
// multiplied by seed-derived odd constants, xor-folded, reduced modulo a
// per-head prime plus a per-head offset — and its constants arrive by value
// in one aggregate parameter, because no checkpoint plane needs to be read
// to know them.
//
// The window cache is a per-lane state slab of `ngram - 1` i32 cells storing
// PREVIOUS token ids as `id + 1`, so a zeroed slot reads as "no history" and
// the reference's eos padding falls out of the sentinel rather than out of a
// separate reset. The eos-segmentation rule reduces, for a window of two,
// to: the id one back is itself; the id two back is eos when the id one
// back is eos.

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

// Hash the window [t, p1, p2, ...] (newest first) for every head.
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

// Apply the eos-segmentation rule to the raw window: a previous id is
// replaced by eos when a NEARER previous id is eos (the window crossed a
// sequence boundary).
__device__ __forceinline__ void ple_mask_window(
    const PleHash& h, int* window)
{
    bool crossed = false;
    for (int p = 1; p < h.ngram; ++p) {
        if (crossed) window[p] = h.eos;
        if (window[p] == h.eos) crossed = true;
    }
}

// Decode form: one thread per lane row. Reads the lane's state, hashes the
// one new token, shifts the window.
__global__ void ple_ngram_ids_update(
    const int* __restrict__ ids,
    int* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    int* __restrict__ ngram_ids,
    int rows,
    PleHash h,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first thread owns. The token
    // it reads and the table row it lands are both there; the slot table is
    // the LANES', and a lane ordinal is not a row.
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

    int out[PLE_MAX_HEADS];
    ple_hash_row(h, window, out);
    for (int k = 0; k < h.heads; ++k) {
        ngram_ids[(long long)r_row * h.heads + k] = out[k];
    }

    for (int p = 0; p < span - 1; ++p) state[p] = state[p + 1];
    state[span - 1] = ids[r_row] + 1;
}

// Prefill form: one thread block per request; walks the request's tokens in
// order (the window is tiny, so every thread rebuilds its own token's
// window from the fire's rows and the state fills only the first `span`).
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
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;

    // **THE STAGED-GEOMETRY SEAT, ON THE LANE AXIS** (the chunked-arm wave).
    // One block per REQUEST, so the word that retires a ceiling grid's padding
    // is `win[2]` — the window's live lane count — and not `win[0]`, which is
    // the row count the decode form above reads.
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    // **AND THE LANE READS SPLIT, WHICH IS THIS WAVE'S CRUX.** `qo_indptr` is
    // the WINDOW's own rebased CSR — staged into the fixed-stride window blob
    // at an address a body may bake — so it is read at the window-local `r`.
    // `slot_ids`, the fold predicate, the commit length and the segment origin
    // are the FIRE's tables, handed over whole under a plane base
    // (`Run::recurrent_absolute`) because `lane_offset` is not a function of a
    // body key; those are read at `r + win[3]`.
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    // And the ROW axis: `ids` and `ngram_ids` are the fire's own planes,
    // handed as BASES under an armed seat while the CSR above counts from the
    // window's zero, so `win[1]` bridges the two. The state slab is addressed
    // by the slot's VALUE and moves for neither.
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = (int)qo_indptr[r] + row0;
    int Nr = (int)qo_indptr[r + 1] - (int)qo_indptr[r];

    // The segment this launch owns (the 2R split) — the causal conv's own
    // trimming, read for the same reason: state may only advance over the
    // committed prefix, and the tail launch re-covers the rest.
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
        // The new window: the last `span` ids of (state ++ segment).
        int next[PLE_MAX_NGRAM];
        for (int p = 0; p < span; ++p) {
            const int src_t = Nr - span + p;
            next[p] = src_t >= 0 ? ids[t0 + src_t] + 1 : state[p + Nr];
        }
        for (int p = 0; p < span; ++p) state[p] = next[p];
    }
}

} // namespace pie::attn
