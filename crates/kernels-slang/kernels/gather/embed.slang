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
// pie:instantiate gather_concat_bf16 PIE_CONCAT=1 PIE_GROUP_X=256
// pie:instantiate gather_weighted_bf16 PIE_WEIGHTED=1 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_32_b_2 PIE_AFFINE=1 PIE_GROUP=32 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_32_b_4 PIE_AFFINE=1 PIE_GROUP=32 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_32_b_8 PIE_AFFINE=1 PIE_GROUP=32 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_64_b_2 PIE_AFFINE=1 PIE_GROUP=64 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_64_b_4 PIE_AFFINE=1 PIE_GROUP=64 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_64_b_8 PIE_AFFINE=1 PIE_GROUP=64 PIE_BITS=8 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_128_b_2 PIE_AFFINE=1 PIE_GROUP=128 PIE_BITS=2 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_128_b_4 PIE_AFFINE=1 PIE_GROUP=128 PIE_BITS=4 PIE_GROUP_X=256
// pie:instantiate gather_affine_bf16_gs_128_b_8 PIE_AFFINE=1 PIE_GROUP=128 PIE_BITS=8 PIE_GROUP_X=256
