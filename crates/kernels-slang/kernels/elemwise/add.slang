#include "prim/pair.slang"
#include "prim/params.slang"

// `z = x + y` over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> y;
[[vk::binding(2)]] RWStructuredBuffer<PIE_ACT> z;

struct Params {
    uint n;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(z, i, both, pair_at(x, i, both) + pair_at(y, i, both));
}

// pie:instantiate elemwise_add_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_add_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
