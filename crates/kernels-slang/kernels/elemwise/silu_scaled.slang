#include "prim/pair.slang"
#include "prim/params.slang"

// `x = silu(x * s)` in place over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;

struct Params {
    float s;
    uint n;
};
PIE_PARAMS(1, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    float2 v = pair_at(x, i, both) * params.s;
    pair_put(x, i, both, v / (1.0 + exp(-v)));
}

// pie:instantiate elemwise_silu_scaled_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_silu_scaled_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
