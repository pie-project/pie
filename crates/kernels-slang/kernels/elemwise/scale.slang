#include "prim/pair.slang"
#include "prim/params.slang"

// `x *= s` in place over a whole tensor, a pair of elements a lane. The
// scalar is stated, rounded as the activation would hold it, or with
// `PIE_READ` read from a one-element tensor of the activation's element.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;
#ifdef PIE_READ
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> s;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);

float scalar() {
    return pair_at(s, 0u, false).x;
}
#else
struct Params {
    float s;
    uint n;
};
PIE_PARAMS(1, Params, params);

float scalar() {
    return act_round(params.s);
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
    pair_put(x, i, both, pair_at(x, i, both) * scalar());
}

// pie:instantiate elemwise_scale_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_scale_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_scale_read_bf16 PIE_GROUP_X=256 PIE_READ=1
