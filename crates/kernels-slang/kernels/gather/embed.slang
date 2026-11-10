#include "prim/bf16.slang"
#include "prim/params.slang"

// Embedding lookups into bf16 rows, a word (two columns) a lane and a landed
// row a grid row: a table row by token id (an out-of-range id reads row 0),
// table rows side by side (`PIE_CONCAT`, an out-of-range id lands zeros), an
// affine-quantized bank's rows (`PIE_AFFINE`, zeros out of range), and a
// weighted sum of taps (`PIE_WEIGHTED`). `PIE_<KERNEL>` picks the kernel.

#if defined(PIE_EMBED) || defined(PIE_CONCAT)

[[vk::binding(0)]] StructuredBuffer<int> ids;
[[vk::binding(1)]] StructuredBuffer<uint> table;
[[vk::binding(2)]] RWStructuredBuffer<uint> y;

struct Params {
    int hidden;
    int vocab;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint words = uint(max(params.hidden, 0)) / 2u;
    uint c = gid.x;
    uint n = gid.y;
    if (c >= words) {
        return;
    }
    int raw = ids[n];
    bool inside = raw >= 0 && raw < params.vocab;
#if defined(PIE_CONCAT)
    y[n * words + c] = inside ? table[uint(raw) * words + c] : 0u;
#else
    y[n * words + c] = table[(inside ? uint(raw) : 0u) * words + c];
#endif
}

#elif defined(PIE_AFFINE)

#include "prim/affine.slang"

[[vk::binding(0)]] StructuredBuffer<uint> w;
[[vk::binding(1)]] StructuredBuffer<uint> scales;
[[vk::binding(2)]] StructuredBuffer<uint> biases;
[[vk::binding(3)]] StructuredBuffer<int> ids;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;

struct Params {
    int hidden;
    int vocab;
};
PIE_PARAMS(5, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint hidden = uint(max(params.hidden, 0));
    uint k = 2u * gid.x;
    uint m = gid.y;
    if (k >= hidden) {
        return;
    }
    int id = ids[m];
    uint at = (m * hidden + k) >> 1;
    if (id < 0 || id >= params.vocab) {
        y[at] = 0u;
        return;
    }
    uint row = uint(id);
    uint word = w[affine_word_of(row, hidden, k)];
    uint g = affine_scale_of(row, hidden, k);
    float s = bf16_at(scales, g);
    float b = bf16_at(biases, g);
    uint c = affine_code_of(k);
    y[at] = pack_bf16(affine_value(word, c, s, b), affine_value(word, c + 1u, s, b));
}

#elif defined(PIE_TABLED)

#include "prim/pool.slang"

// A bf16 table streamed by its rows (the weight cache, SPEC "Weight
// cache"): row `id` lies in block `id / block_rows`, which `blocks` names
// the pool unit of (its slot) or `NOT_HELD`. A held block's rows lie in
// its slot, `atom_words` a row apart from word `origin` (a token id's rows
// of every table sharing the slot side by side); a row of a block not
// held was staged for this fire, `staged[n]` its row of `stage`. Each
// row read counts one use of its block in `hits`.

#define NOT_HELD 0xffffffffu

[[vk::binding(0)]] StructuredBuffer<int> ids;
[[vk::binding(1)]] StructuredBuffer<uint> blocks;
[[vk::binding(2)]] StructuredBuffer<uint> staged;
[[vk::binding(3)]] StructuredBuffer<uint> stage;
[[vk::binding(4)]] RWStructuredBuffer<Atomic<uint>> hits;
[[vk::binding(5)]] RWStructuredBuffer<uint> y;
PIE_POOL_SEGMENTS(StructuredBuffer, uint, 6)

struct Params {
    int hidden;
    int vocab;
    uint block_rows;
    uint seg_words;
    uint unit_words;
    uint origin;
    uint atom_words;
};
PIE_PARAMS(10, Params, params);
PIE_POOL_READ(uint, params.seg_words)

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint words = uint(max(params.hidden, 0)) / 2u;
    uint c = gid.x;
    uint n = gid.y;
    if (c >= words) {
        return;
    }
    int raw = ids[n];
    uint id = (raw >= 0 && raw < params.vocab) ? uint(raw) : 0u;
    uint block = id / params.block_rows;
    if (c == 0u) {
        hits[block].add(1u);
    }
    uint unit = blocks[block];
    if (unit != NOT_HELD) {
        y[n * words + c] = pie_pool_get(
            unit * params.unit_words + params.origin + (id % params.block_rows) * params.atom_words + c);
    } else {
        uint at = staged[n];
        y[n * words + c] = at != NOT_HELD ? stage[at * words + c] : 0u;
    }
}

#elif defined(PIE_WEIGHTED)

[[vk::binding(0)]] StructuredBuffer<int> ids;
[[vk::binding(1)]] StructuredBuffer<float> weights;
[[vk::binding(2)]] StructuredBuffer<uint> table;
[[vk::binding(3)]] RWStructuredBuffer<uint> y;

struct Params {
    int hidden;
    int vocab;
    int taps;
    int rows;
};
PIE_PARAMS(4, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint hidden = uint(max(params.hidden, 0));
    uint c = 2u * gid.x;
    uint n = gid.y;
    if (c >= hidden || n >= uint(max(params.rows, 0))) {
        return;
    }
    uint taps = uint(max(params.taps, 0));
    float2 acc = float2(0.0, 0.0);
    for (uint t = 0u; t < taps; t++) {
        int raw = ids[n * taps + t];
        uint at = (raw >= 0 && raw < params.vocab) ? uint(raw) : 0u;
        uint word = table[(at * hidden + c) >> 1];
        acc += weights[n * taps + t] * float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
    }
    y[(n * hidden + c) >> 1] = pack_bf16(acc.x, acc.y);
}

#else
#error "gather/embed.slang: no kernel selected"
#endif

// pie:instantiate gather_embed_bf16 PIE_EMBED=1 PIE_GROUP_X=256
// pie:instantiate gather_embed_concat_bf16 PIE_CONCAT=1 PIE_GROUP_X=256
// pie:instantiate gather_embed_tabled_bf16 PIE_TABLED=1 PIE_GROUP_X=256
// pie:instantiate gather_embed_weighted_bf16 PIE_WEIGHTED=1 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_32_b_2 PIE_AFFINE=1 PIE_GROUP=32 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_32_b_4 PIE_AFFINE=1 PIE_GROUP=32 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_32_b_8 PIE_AFFINE=1 PIE_GROUP=32 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_64_b_2 PIE_AFFINE=1 PIE_GROUP=64 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_64_b_4 PIE_AFFINE=1 PIE_GROUP=64 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_64_b_8 PIE_AFFINE=1 PIE_GROUP=64 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_128_b_2 PIE_AFFINE=1 PIE_GROUP=128 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_128_b_4 PIE_AFFINE=1 PIE_GROUP=128 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate gather_embed_affine_bf16_gs_128_b_8 PIE_AFFINE=1 PIE_GROUP=128 PIE_BITS=8 PIE_GROUP_X=256
