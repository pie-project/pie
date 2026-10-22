#include "prim/pair.slang"
#include "prim/params.slang"

// f32 values rounded into bf16 ones, a pair a lane.

[[vk::binding(0)]] StructuredBuffer<float> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> o;

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
    pair_put(o, i, both, float2(x[i], both ? x[i + 1u] : 0.0));
}

// pie:instantiate elemwise_cast_f32_to_bf16 PIE_GROUP_X=256
