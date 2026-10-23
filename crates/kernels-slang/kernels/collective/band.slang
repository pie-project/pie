#include "prim/pair.slang"
#include "prim/params.slang"

// The collectives over a band: one buffer every rank of a communicator
// sees, a `slot` of elements a rank (even, so a slot starts on a word). A
// rank publishes its operand into its own slot; once every rank has, each
// folds the band. A pair of elements a lane (`prim/pair.slang`), sums in
// f32 in rank order.
//
// `PIE_PUBLISH`  band[rank * slot + i] = x[i]
// `PIE_REDUCE`   y[i] = sum_r band[r * slot + i]            (all-reduce)
// `PIE_GATHER`   y[r * n + i] = band[r * slot + i]           (all-gather)
// `PIE_SCATTER`  y[i] = sum_r band[r * slot + rank * n + i]  (reduce-scatter)

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> in_;
[[vk::binding(1)]] RWStructuredBuffer<PIE_ACT> out_;

struct Params {
    uint n;
    uint world;
    uint rank;
    uint slot;
};
PIE_PARAMS(2, Params, params);

// Elements `i` and (when `both`) `i + 1` of the band, `i` of either parity:
// a reduce-scatter's shard of an odd count starts mid-word.
float2 read_pair(uint i, bool both) {
#ifdef PIE_ACT_F32
    return pair_at(in_, i, both);
#else
    if ((i & 1u) == 0u) {
        return pair_at(in_, i, both);
    }
    return float2(act_at(in_, i), both ? act_at(in_, i + 1u) : 0.0);
#endif
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
#if defined(PIE_PUBLISH)
    pair_put(out_, params.rank * params.slot + i, both, pair_at(in_, i, both));
#elif defined(PIE_GATHER)
    for (uint r = 0u; r < params.world; ++r) {
        pair_put(out_, r * params.n + i, both, pair_at(in_, r * params.slot + i, both));
    }
#elif defined(PIE_SCATTER)
    float2 acc = float2(0.0, 0.0);
    for (uint r = 0u; r < params.world; ++r) {
        acc += read_pair(r * params.slot + params.rank * params.n + i, both);
    }
    pair_put(out_, i, both, acc);
#else
    float2 acc = float2(0.0, 0.0);
    for (uint r = 0u; r < params.world; ++r) {
        acc += pair_at(in_, r * params.slot + i, both);
    }
    pair_put(out_, i, both, acc);
#endif
}

// pie:instantiate collective_band_publish_bf16 PIE_GROUP_X=256 PIE_PUBLISH=1
// pie:instantiate collective_band_publish_f32 PIE_GROUP_X=256 PIE_PUBLISH=1 PIE_ACT_F32=1
// pie:instantiate collective_band_reduce_bf16 PIE_GROUP_X=256
// pie:instantiate collective_band_reduce_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate collective_band_gather_bf16 PIE_GROUP_X=256 PIE_GATHER=1
// pie:instantiate collective_band_gather_f32 PIE_GROUP_X=256 PIE_GATHER=1 PIE_ACT_F32=1
// pie:instantiate collective_band_scatter_bf16 PIE_GROUP_X=256 PIE_SCATTER=1
// pie:instantiate collective_band_scatter_f32 PIE_GROUP_X=256 PIE_SCATTER=1 PIE_ACT_F32=1
