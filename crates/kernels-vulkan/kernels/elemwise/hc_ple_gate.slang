#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RO(0, key);
PIE_BUFFER_RO(1, query);
PIE_BUFFER_RO(2, value);
PIE_BUFFER_RW(3, y);

struct Push { int m; int h; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint M = uint(pc.m);
    uint H = uint(pc.h);
    uint n = group.x / M;
    uint s = group.x - n * M;
    uint stream = (n * M + s) * H;
    float acc = 0.0;
    for (uint i = local.x; i < H; i += uint(PIE_GROUP_X)) {
        acc += PIE_LOAD(key[stream + i]) * PIE_LOAD(query[stream + i]);
    }
    float dot = pie_workgroup_sum(acc) * rsqrt(float(H));
    float damped = sqrt(max(abs(dot), 1e-6));
    damped = dot > 0.0 ? damped : (dot < 0.0 ? -damped : 0.0);
    float gate = 1.0 / (1.0 + exp(-damped));
    for (uint i = local.x; i < H; i += uint(PIE_GROUP_X)) {
        y[stream + i] = PIE_STORE(gate * PIE_LOAD(value[n * H + i]));
    }
}

// pie:instantiate ple_gate_bf16 PIE_GROUP_X=256
