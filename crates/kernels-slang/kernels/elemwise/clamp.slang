#include "prim/pair.slang"
#include "prim/params.slang"

// `x = min(max(x, lo), hi)` in place over a whole tensor, a pair of elements
// a lane. The bounds are stated, rounded as the activation would hold them,
// or with `PIE_LEARNED` read from one-element tensors of its element.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;
#ifdef PIE_LEARNED
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> lo_t;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> hi_t;

struct Params {
    uint n;
};
PIE_PARAMS(3, Params, params);

float2 bounds() {
    return float2(pair_at(lo_t, 0u, false).x, pair_at(hi_t, 0u, false).x);
}
#else
struct Params {
    float lo;
    float hi;
    uint n;
};
PIE_PARAMS(1, Params, params);

float2 bounds() {
    return float2(act_round(params.lo), act_round(params.hi));
}
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    float2 b = bounds();
    pair_put(x, i, both, min(max(pair_at(x, i, both), b.x), b.y));
}

// pie:instantiate elemwise_clamp_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_clamp_learned_bf16 PIE_GROUP_X=256 PIE_LEARNED=1
