#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RO(0, ba);
[[vk::binding(1)]] StructuredBuffer<float> a_log;
PIE_BUFFER_RO(2, dt_bias);
[[vk::binding(3)]] RWStructuredBuffer<float> gates;

struct Push {
    int v_heads;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint h = gid.x;
    uint t = gid.y;
    uint vh = uint(pc.v_heads);
    if (h >= vh) {
        return;
    }
    uint row = t * 2u * vh;
    float bv = PIE_LOAD(ba[row + h]);
    float av = PIE_LOAD(ba[row + vh + h]);
    float z = av + PIE_LOAD(dt_bias[h]);
    float sp = z > 20.0 ? z : log(1.0 + exp(z));
    gates[row + h] = -exp(a_log[h]) * sp;
    gates[row + vh + h] = 1.0 / (1.0 + exp(-bv));
}

// pie:instantiate gdn_ba_gates_bf16 PIE_GROUP_X=256
