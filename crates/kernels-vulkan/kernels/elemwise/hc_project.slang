#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

[[vk::binding(0)]] StructuredBuffer<float> normed;
[[vk::binding(1)]] StructuredBuffer<float> hc_fn;
[[vk::binding(2)]] RWStructuredBuffer<float> mixes;

struct Push { int fan_in; int mix_hc; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint mix_hc = uint(pc.mix_hc);
    uint fan = uint(pc.fan_in);
    uint o = group.x % mix_hc;
    uint n = group.x / mix_hc;
    float acc = 0.0;
    for (uint d = local.x; d < fan; d += uint(PIE_GROUP_X)) {
        acc += normed[n * fan + d] * hc_fn[o * fan + d];
    }
    float total = pie_workgroup_sum(acc);
    if (local.x == 0) {
        mixes[n * mix_hc + o] = total;
    }
}

// pie:instantiate hc_project PIE_GROUP_X=256
