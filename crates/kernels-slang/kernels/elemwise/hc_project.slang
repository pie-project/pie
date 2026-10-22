#include "prim/params.slang"
#include "prim/reduce.slang"

// The hyper-connection mix projection in f32, one workgroup a (row, mix):
// `mixes[n, o] = normed[n] . hc_fn[o]`, the dot reduced through workgroup
// memory.

[[vk::binding(0)]] StructuredBuffer<float> normed;
[[vk::binding(1)]] StructuredBuffer<float> hc_fn;
[[vk::binding(2)]] RWStructuredBuffer<float> mixes;

struct Params {
    int fan_in;
    int mix_hc;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint mix_hc = uint(max(params.mix_hc, 1));
    uint fan = uint(max(params.fan_in, 0));
    uint o = group.x % mix_hc;
    uint n = group.x / mix_hc;
    uint lid = local.x;
    float acc = 0.0;
    for (uint d = lid; d < fan; d += uint(PIE_GROUP_X)) {
        acc += normed[n * fan + d] * hc_fn[o * fan + d];
    }
    float total = pie_workgroup_sum(acc, lid);
    if (lid == 0u) {
        mixes[n * mix_hc + o] = total;
    }
}

// pie:instantiate elemwise_hc_project_f32 PIE_GROUP_X=256
