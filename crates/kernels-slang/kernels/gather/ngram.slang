#include "prim/params.slang"
#include "prim/u64.slang"

// Per-layer embedding n-gram ids: each token hashes the window of itself and
// the `ngram - 1` ids before it (from the batch, or from the lane's recurrent
// state, which holds `id + 1` with 0 for "none yet") into one table row per
// head: for each order, `xor_p window[p] * mults[p]` modulo the head's prime
// plus its offset, in 64-bit arithmetic. A window is cut at the first
// end-of-sequence id. The hash plane is `mults | primes | offsets`, u64s as
// (lo, hi) words. `PIE_UPDATE`: one decoded token a lane; `PIE_CHUNKED`: a
// request's run of tokens a lane (an i32 CSR). Both roll the state forward.

#define MAX_NGRAM 4
#define MAX_HEADS 32

[[vk::binding(0)]] StructuredBuffer<int> ids;

#if defined(PIE_UPDATE)
[[vk::binding(1)]] RWStructuredBuffer<int> state;
[[vk::binding(2)]] StructuredBuffer<uint> slots;
[[vk::binding(3)]] StructuredBuffer<uint> hash;
[[vk::binding(4)]] RWStructuredBuffer<int> ngram_ids;
#define PIE_PARAMS_AT 5
#elif defined(PIE_CHUNKED)
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] RWStructuredBuffer<int> state;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] StructuredBuffer<uint> hash;
[[vk::binding(5)]] RWStructuredBuffer<int> ngram_ids;
#define PIE_PARAMS_AT 6
#else
#error "gather/ngram.slang: no kernel selected"
#endif

struct Params {
    int ngram;
    int heads;
    int heads_per_ngram;
    int eos;
    int lanes;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

U64 hash_at(uint i) {
    return u64_of(hash[2u * i], hash[2u * i + 1u]);
}

int cell(int held) {
    return held == 0 ? params.eos : held - 1;
}

// Every id after the first end-of-sequence in the window becomes one.
void mask_window(inout int window[MAX_NGRAM]) {
    bool crossed = false;
    for (int p = 1; p < params.ngram; p++) {
        if (crossed) {
            window[p] = params.eos;
        }
        if (window[p] == params.eos) {
            crossed = true;
        }
    }
}

// Hash `window` into row `row` of the ids.
void hash_row(int window[MAX_NGRAM], uint row) {
    for (int order = 2; order <= params.ngram; order++) {
        U64 mixed = u64_mul(u64_from_i32(window[0]), hash_at(0u));
        for (int p = 1; p < order; p++) {
            mixed = u64_xor(mixed, u64_mul(u64_from_i32(window[p]), hash_at(uint(p))));
        }
        int base = (order - 2) * params.heads_per_ngram;
        for (int k = 0; k < params.heads_per_ngram; k++) {
            int head = base + k;
            U64 prime = hash_at(uint(params.ngram + head));
            U64 offset = hash_at(uint(params.ngram + params.heads + head));
            ngram_ids[row * uint(params.heads) + uint(head)] = int(u64_add(u64_mod(mixed, prime), offset).lo);
        }
    }
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint r = gid.x;
    if (r >= uint(params.lanes)) {
        return;
    }
    int span = params.ngram - 1;
#if defined(PIE_UPDATE)
    uint slab = slots[r] * uint(span);
    int window[MAX_NGRAM] = { 0, 0, 0, 0 };
    int fresh = ids[r];
    window[0] = fresh;
    for (int p = 1; p <= span; p++) {
        window[p] = cell(state[slab + uint(span - p)]);
    }
    mask_window(window);
    hash_row(window, r);
    for (int p = 0; p + 1 < span; p++) {
        state[slab + uint(p)] = state[slab + uint(p + 1)];
    }
    state[slab + uint(span - 1)] = fresh + 1;
#else
    int begin = indptr[r];
    int rows = indptr[r + 1u] - begin;
    if (rows <= 0) {
        return;
    }
    uint slab = slots[uint(begin)] * uint(span);
    for (int t = 0; t < rows; t++) {
        int window[MAX_NGRAM] = { 0, 0, 0, 0 };
        window[0] = ids[uint(begin + t)];
        for (int p = 1; p <= span; p++) {
            window[p] = t >= p ? ids[uint(begin + t - p)] : cell(state[slab + uint(span - (p - t))]);
        }
        mask_window(window);
        hash_row(window, uint(begin + t));
    }
    int next[MAX_NGRAM] = { 0, 0, 0, 0 };
    for (int p = 0; p < span; p++) {
        int src = rows - span + p;
        next[p] = src >= 0 ? ids[uint(begin + src)] + 1 : state[slab + uint(p + rows)];
    }
    for (int p = 0; p < span; p++) {
        state[slab + uint(p)] = next[p];
    }
#endif
}

// pie:instantiate gather_ngram_update PIE_UPDATE=1 PIE_GROUP_X=64
// pie:instantiate gather_ngram_chunked PIE_CHUNKED=1 PIE_GROUP_X=64
