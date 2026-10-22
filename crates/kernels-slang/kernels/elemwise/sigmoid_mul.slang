#include "prim/pair.slang"
#include "prim/params.slang"

// `x *= sigmoid(gate)` in place, the gate a tensor of `x`'s own shape, a
// pair of elements a lane.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> gate;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);

float2 sigmoid(float2 g) {
    float2 e = 1.0 / (1.0 + exp(-abs(g)));
    return float2(g.x < 0.0 ? 1.0 - e.x : e.x, g.y < 0.0 ? 1.0 - e.y : e.y);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(x, i, both, pair_at(x, i, both) * sigmoid(pair_at(gate, i, both)));
}

// pie:instantiate elemwise_sigmoid_mul_bf16 PIE_GROUP_X=256
