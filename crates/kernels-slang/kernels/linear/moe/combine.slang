#include "prim/pair.slang"
#include "prim/params.slang"

// Mixture-of-experts combines on bf16, a pair of elements a lane over the
// token rectangle, an odd tail half there. `PIE_WEIGHTED`: each row the
// route-weighted sum of its `top_k` routed rows. `PIE_BIAS`: each row plus
// its routes' weighted expert biases (a route below 0 adds none).
// `PIE_SHARED`: the routed row plus the shared expert's, gated by the
// sigmoid of the row's gate.

[[vk::binding(0)]] StructuredBuffer<uint> a;
#if defined(PIE_WEIGHTED)
[[vk::binding(1)]] StructuredBuffer<float> expert_weights;
[[vk::binding(2)]] RWStructuredBuffer<uint> y;
#define PIE_PARAMS_AT 3
#elif defined(PIE_BIAS)
[[vk::binding(1)]] StructuredBuffer<uint> bias;
[[vk::binding(2)]] StructuredBuffer<int> expert_ids;
[[vk::binding(3)]] StructuredBuffer<float> expert_weights;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;
#define PIE_PARAMS_AT 5
#elif defined(PIE_SHARED)
[[vk::binding(1)]] StructuredBuffer<uint> shared_;
[[vk::binding(2)]] StructuredBuffer<uint> gate;
[[vk::binding(3)]] RWStructuredBuffer<uint> y;
#define PIE_PARAMS_AT 4
#else
#error "linear/moe/combine.slang: no kernel selected"
#endif

struct Params {
    int width;
    int rows;
#if !defined(PIE_SHARED)
    int experts_per_token;
#endif
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

// Element `i` of the token rectangle, combined.
float combined(uint i) {
    uint width = uint(params.width);
    uint row = i / width;
#if defined(PIE_SHARED)
    float g = 1.0 / (1.0 + exp(-bf16_at(gate, row)));
    return bf16_at(a, i) + g * bf16_at(shared_, i);
#else
    uint c = i - row * width;
    uint top_k = uint(params.experts_per_token);
    uint base = row * top_k;
#if defined(PIE_WEIGHTED)
    float acc = 0.0;
    for (uint e = 0u; e < top_k; e++) {
        acc += expert_weights[base + e] * bf16_at(a, (base + e) * width + c);
    }
#else
    float acc = bf16_at(a, i);
    for (uint e = 0u; e < top_k; e++) {
        int expert = expert_ids[base + e];
        if (expert >= 0) {
            acc += expert_weights[base + e] * bf16_at(bias, uint(expert) * width + c);
        }
    }
#endif
    return acc;
#endif
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint n = uint(params.width) * uint(params.rows);
    uint i = 2u * gid.x;
    if (i >= n) {
        return;
    }
    bool both = i + 1u < n;
    pair_put(y, i, both, float2(combined(i), both ? combined(i + 1u) : 0.0));
}

// pie:instantiate linear_moe_combine_weighted_sum_bf16 PIE_WEIGHTED=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_combine_bias_sum_bf16 PIE_BIAS=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_combine_sigmoid_gate_add_bf16 PIE_SHARED=1 PIE_GROUP_X=256
