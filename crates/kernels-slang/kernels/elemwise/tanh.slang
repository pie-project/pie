#include "prim/pair.slang"
#include "prim/params.slang"

// `x = tanh(x)` in place over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;

struct Params {
    uint n;
};
PIE_PARAMS(1, Params, params);

// `1 - 2 / (e^2|v| + 1)` with the sign put back: the exponential may
// overflow to infinity and the quotient still lands on 1.
float2 stable_tanh(float2 v) {
    float2 t = 1.0 - 2.0 / (exp(2.0 * abs(v)) + 1.0);
    return float2(v.x < 0.0 ? -t.x : t.x, v.y < 0.0 ? -t.y : t.y);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(x, i, both, stable_tanh(pair_at(x, i, both)));
}

// pie:instantiate elemwise_tanh_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_tanh_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
