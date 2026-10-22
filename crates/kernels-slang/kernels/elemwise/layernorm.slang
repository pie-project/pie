#include "prim/pair.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// Layer norm over whole rows, one workgroup a row and a pair of elements a
// lane: `y = (x - mean) / sqrt(var + eps) * w + b`, or with `PIE_UNSCALED`
// no weight and no bias. The weight and the bias ride the row's element.

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
#ifdef PIE_UNSCALED
[[vk::binding(1)]] RWStructuredBuffer<PIE_ACT> y;
#define PIE_PARAMS_AT 2
#else
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> w;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> b;
[[vk::binding(3)]] RWStructuredBuffer<PIE_ACT> y;
#define PIE_PARAMS_AT 4
#endif

struct Params {
    float eps;
    int axis_size;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = uint(max(params.axis_size, 0));
    uint base = group.x * n;
    uint lid = local.x;
    uint stride = 2u * uint(PIE_GROUP_X);

    float acc = 0.0;
    for (uint c = 2u * lid; c < n; c += stride) {
        float2 v = pair_at(x, base + c, c + 1u < n);
        acc += v.x + v.y;
    }
    float mean = pie_workgroup_sum(acc, lid) / float(n);

    float spread = 0.0;
    for (uint c = 2u * lid; c < n; c += stride) {
        bool both = c + 1u < n;
        float2 d = pair_at(x, base + c, both) - mean;
        spread += d.x * d.x + (both ? d.y * d.y : 0.0);
    }
    float inv = rsqrt(pie_workgroup_sum(spread, lid) / float(n) + params.eps);

    for (uint c = 2u * lid; c < n; c += stride) {
        bool both = c + 1u < n;
        float2 v = (pair_at(x, base + c, both) - mean) * inv;
#ifndef PIE_UNSCALED
        v = v * pair_at(w, c, both) + pair_at(b, c, both);
#endif
        pair_put(y, base + c, both, v);
    }
}

// pie:instantiate elemwise_layernorm_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_layernorm_unscaled_bf16 PIE_GROUP_X=256 PIE_UNSCALED=1
// pie:instantiate elemwise_layernorm_unscaled_f32 PIE_GROUP_X=256 PIE_UNSCALED=1 PIE_ACT_F32=1
