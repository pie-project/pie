#pragma once

// Hashed n-gram ids (Engram / per-layer n-gram embeddings): for each token,
// the window of it and its last `ngram − 1` predecessors is hashed per order
// (2..ngram) into `heads_per_ngram` table rows each,
//
//   mixed = ⊕_{p < order} window[p] · mults[p]        (u64, wrapping)
//   id    = mixed mod primes[head] + offsets[head]
//
// A predecessor past an eos sees eos (the window does not reach across a
// document boundary). Predecessors before the fire's rows come from the
// slot's recurrent state: the last `ngram − 1` ids, oldest first, each
// stored + 1 so that 0 reads as "none" (eos).

#include "prelude/device.hip"

namespace pie::attn {

constexpr int kPleMaxNgram = 4;
constexpr int kPleMaxHeads = 32;

// Mirrors `attn::ple::PleHash` field for field; passed by value.
struct PleHash {
    u64 mults[kPleMaxNgram];
    u64 primes[kPleMaxHeads];
    u64 offsets[kPleMaxHeads];
    int ngram;
    int heads;
    int heads_per_ngram;
    int eos;
};

// Hash `window` (window[0] the token, window[p] its p-th predecessor) into
// the token's row of `heads` ids.
__device__ __forceinline__ void ple_hash_row(const PleHash& h, const int* window, int* __restrict__ out) {
    for (int order = 2; order <= h.ngram; ++order) {
        u64 mixed = static_cast<u64>(window[0]) * h.mults[0];
        for (int p = 1; p < order; ++p) mixed ^= static_cast<u64>(window[p]) * h.mults[p];
        const int base = (order - 2) * h.heads_per_ngram;
        for (int k = 0; k < h.heads_per_ngram; ++k) {
            const int head = base + k;
            out[head] = static_cast<int>(mixed % h.primes[head] + h.offsets[head]);
        }
    }
}

// Every predecessor past the first eos reads eos, and the ids are mapped
// through `map` (Engram's tokenizer compression) after the masking, so the
// state and the eos barrier stay in the tokenizer's id space.
__device__ __forceinline__ void ple_prepare(const PleHash& h, const i64* __restrict__ map, int* window) {
    bool crossed = false;
    for (int p = 1; p < h.ngram; ++p) {
        if (crossed) window[p] = h.eos;
        if (window[p] == h.eos) crossed = true;
    }
    if (map != nullptr)
        for (int p = 0; p < h.ngram; ++p) window[p] = static_cast<int>(map[window[p]]);
}

// A predecessor from the state, cell 0 the oldest.
__device__ __forceinline__ int ple_cell(const PleHash& h, int cell) { return cell == 0 ? h.eos : cell - 1; }

// A thread per (token, head), each token from its own slot (decode: a
// request is one row): a block holds whole rows, `blockDim.x / heads` of
// them, so each row's state moves once its heads have all read it.
__global__ void ple_ngram_ids_update(const int* __restrict__ ids, int* __restrict__ state_base,
                                     const int* __restrict__ slot_ids, long long slot_stride_elems,
                                     int* __restrict__ ngram_ids, int rows, PleHash h,
                                     const i64* __restrict__ map, const u32* __restrict__ win) {
    const Window window_{win};
    const int per_block = static_cast<int>(blockDim.x) / h.heads;
    const int local = static_cast<int>(threadIdx.x) / h.heads, head = static_cast<int>(threadIdx.x) % h.heads;
    const int r = static_cast<int>(blockIdx.x) * per_block + local;
    const bool live = local < per_block && r < rows && window_.live(r);
    const long long r_row = live ? window_.plane(r) : 0;
    const int slot = live ? slot_ids[r] : -1;
    int* state = state_base + static_cast<long long>(max(slot, 0)) * slot_stride_elems;
    const int span = h.ngram - 1;
    int token = 0;
    if (slot >= 0) {
        int window[kPleMaxNgram];
        token = ids[r_row];
        window[0] = token;
        for (int p = 1; p <= span; ++p) window[p] = ple_cell(h, state[span - p]);
        ple_prepare(h, map, window);
        // The head's order (heads are (ngram − 1)·heads_per_ngram, the
        // launcher checks), and its hash of the window's first `order` ids.
        const int order = 2 + head / h.heads_per_ngram;
        u64 mixed = static_cast<u64>(window[0]) * h.mults[0];
        for (int p = 1; p < order; ++p) mixed ^= static_cast<u64>(window[p]) * h.mults[p];
        ngram_ids[r_row * h.heads + head] = static_cast<int>(mixed % h.primes[head] + h.offsets[head]);
    }
    // Every head of the row has read the state before it moves.
    __syncthreads();
    if (slot >= 0 && head == 0) {
        for (int p = 0; p < span - 1; ++p) state[p] = state[p + 1];
        state[span - 1] = token + 1;
    }
}

// One block per request over its (begin-offset, commit-capped) span of rows,
// a thread per token; then one thread folds the span's last ids into the
// state when the request commits it.
__global__ void ple_ngram_ids_chunked(const int* __restrict__ ids, int* __restrict__ state_base,
                                      const int* __restrict__ slot_ids, const u32* __restrict__ qo_indptr,
                                      long long slot_stride_elems, int* __restrict__ ngram_ids,
                                      bool write_state, const u8* __restrict__ write_state_mask,
                                      const int* __restrict__ commit_len, const int* __restrict__ begin_at,
                                      PleHash h, const i64* __restrict__ map,
                                      const u32* __restrict__ win) {
    const int r = static_cast<int>(blockIdx.x);
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]) + row0;
    int n = static_cast<int>(qo_indptr[r + 1] - qo_indptr[r]);
    if (begin_at != nullptr) {
        const int b = min(begin_at[rl], n);
        if (b > 0) {
            t0 += b;
            n -= b;
        }
    }
    if (commit_len != nullptr) n = min(n, commit_len[rl]);
    if (n <= 0) return;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    int* state = state_base + static_cast<long long>(slot) * slot_stride_elems;
    const int span = h.ngram - 1;

    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int window[kPleMaxNgram];
        window[0] = ids[t0 + t];
        for (int p = 1; p <= span; ++p) window[p] = t - p >= 0 ? ids[t0 + t - p] : ple_cell(h, state[span - (p - t)]);
        ple_prepare(h, map, window);
        ple_hash_row(h, window, ngram_ids + static_cast<long long>(t0 + t) * h.heads);
    }

    // Every thread has read the state it needs before it is overwritten.
    __syncthreads();
    if (threadIdx.x == 0 && write_state && (write_state_mask == nullptr || write_state_mask[rl] != 0)) {
        int next[kPleMaxNgram];
        for (int p = 0; p < span; ++p) {
            const int src = n - span + p;
            next[p] = src >= 0 ? ids[t0 + src] + 1 : state[p + n];
        }
        for (int p = 0; p < span; ++p) state[p] = next[p];
    }
}

}  // namespace pie::attn
