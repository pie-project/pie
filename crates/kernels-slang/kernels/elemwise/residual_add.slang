#include "prim/pair.slang"
#include "prim/params.slang"

// `y += x` over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] RWStructuredBuffer<PIE_ACT> y;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(y, i, both, pair_at(y, i, both) + pair_at(x, i, both));
}

// pie:instantiate elemwise_residual_add_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_residual_add_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
