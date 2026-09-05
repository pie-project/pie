#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif
#define HC_MAX_MULT 8

[[vk::binding(0)]] StructuredBuffer<float> mixes;
[[vk::binding(1)]] StructuredBuffer<float> scale;
[[vk::binding(2)]] StructuredBuffer<float> base;
PIE_BUFFER_RO(3, residual);
PIE_BUFFER_RW(4, out_);

struct Push { int m; int h; float hc_eps; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float gates[HC_MAX_MULT];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = group.x;
    uint tid = local.x;
    uint M = uint(pc.m);
    uint H = uint(pc.h);
    if (tid < M) {
        float logit = mixes[n * M + tid] * scale[0] + base[tid];
        gates[tid] = 1.0 / (1.0 + exp(-logit)) + pc.hc_eps;
    }
    GroupMemoryBarrierWithGroupSync();
    uint res = n * M * H;
    for (uint h = tid; h < H; h += uint(PIE_GROUP_X)) {
        float acc = 0.0;
        for (uint i = 0; i < M; i++) {
            acc += gates[i] * PIE_LOAD(residual[res + i * H + h]);
        }
        out_[n * H + h] = PIE_STORE(acc);
    }
}

// pie:instantiate hc_collapse_bf16 PIE_GROUP_X=256
