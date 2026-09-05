#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif
#define HC_MAX_MULT 8

[[vk::binding(0)]] StructuredBuffer<float> mixes;
[[vk::binding(1)]] StructuredBuffer<float> scale;
[[vk::binding(2)]] StructuredBuffer<float> base;
PIE_BUFFER_RO(3, residual);
[[vk::binding(4)]] RWStructuredBuffer<float> post_mix;
[[vk::binding(5)]] RWStructuredBuffer<float> comb_mix;
PIE_BUFFER_RW(6, layer_input);

struct Push {
    int m;
    int h;
    float hc_eps;
    float alpha;
    int sinkhorn;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pre[HC_MAX_MULT];
groupshared float post[HC_MAX_MULT];
groupshared float comb[HC_MAX_MULT * HC_MAX_MULT];

float sigmoid(float v) {
    return 1.0 / (1.0 + exp(-v));
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = group.x;
    uint tid = local.x;
    uint M = uint(pc.m);
    uint H = uint(pc.h);
    uint mix_hc = 2u * M + M * M;
    uint row = n * mix_hc;

    if (tid < M) {
        pre[tid] = sigmoid(mixes[row + tid] * scale[0] + base[tid]) + pc.hc_eps;
        float p = sigmoid(mixes[row + M + tid] * scale[1] + base[M + tid]) * pc.alpha;
        post[tid] = p;
        post_mix[n * M + tid] = p;
    }
    if (tid < M * M) {
        comb[tid] = mixes[row + 2u * M + tid] * scale[2] + base[2u * M + tid];
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < M) {
        float max_v = -1e30;
        for (uint j = 0; j < M; j++) {
            max_v = max(max_v, comb[tid * M + j]);
        }
        float sum = 0.0;
        for (uint j = 0; j < M; j++) {
            float e = exp(comb[tid * M + j] - max_v);
            comb[tid * M + j] = e;
            sum += e;
        }
        for (uint j = 0; j < M; j++) {
            comb[tid * M + j] = comb[tid * M + j] / sum + pc.hc_eps;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < M) {
        float col = 0.0;
        for (uint i = 0; i < M; i++) {
            col += comb[i * M + tid];
        }
        col += pc.hc_eps;
        for (uint i = 0; i < M; i++) {
            comb[i * M + tid] = comb[i * M + tid] / col;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (int iter = 0; iter < pc.sinkhorn - 1; iter++) {
        if (tid < M) {
            float rs = 0.0;
            for (uint j = 0; j < M; j++) {
                rs += comb[tid * M + j];
            }
            rs += pc.hc_eps;
            for (uint j = 0; j < M; j++) {
                comb[tid * M + j] = comb[tid * M + j] / rs;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < M) {
            float col = 0.0;
            for (uint i = 0; i < M; i++) {
                col += comb[i * M + tid];
            }
            col += pc.hc_eps;
            for (uint i = 0; i < M; i++) {
                comb[i * M + tid] = comb[i * M + tid] / col;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < M * M) {
        comb_mix[n * M * M + tid] = comb[tid];
    }
    GroupMemoryBarrierWithGroupSync();

    uint res = n * M * H;
    uint out = n * H;
    for (uint h = tid; h < H; h += uint(PIE_GROUP_X)) {
        float acc = 0.0;
        for (uint i = 0; i < M; i++) {
            acc += pre[i] * PIE_LOAD(residual[res + i * H + h]);
        }
        layer_input[out + h] = PIE_STORE(acc);
    }
}

// pie:instantiate hc_gates_bf16 PIE_GROUP_X=256
