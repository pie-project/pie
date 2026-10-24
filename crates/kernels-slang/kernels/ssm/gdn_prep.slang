#include "prim/bf16.slang"
#include "prim/params.slang"

// The gated delta rule's per-head gates from the fused `[b | a]` projection,
// a value head a lane and a row a grid row: `g = -exp(a_log) *
// softplus(a + dt_bias)` and `beta = sigmoid(b)`, landed as the f32 row
// `[g | beta]` beside the projection it is derived from.

[[vk::binding(0)]] StructuredBuffer<uint> ba;
[[vk::binding(1)]] StructuredBuffer<float> a_log;
[[vk::binding(2)]] StructuredBuffer<uint> dt_bias;
[[vk::binding(3)]] RWStructuredBuffer<float> gates;

struct Params {
    int v_heads;
};
PIE_PARAMS(4, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint h = gid.x;
    uint t = gid.y;
    uint vh = uint(params.v_heads);
    if (h >= vh) {
        return;
    }
    uint row = t * 2u * vh;
    float bv = bf16_at(ba, row + h);
    float av = bf16_at(ba, row + vh + h);
    float z = av + bf16_at(dt_bias, h);
    float sp = z > 20.0 ? z : log(1.0 + exp(z));
    gates[row + h] = -exp(a_log[h]) * sp;
    gates[row + vh + h] = 1.0 / (1.0 + exp(-bv));
}

// pie:instantiate ssm_gdn_prep_bf16 PIE_GROUP_X=256
