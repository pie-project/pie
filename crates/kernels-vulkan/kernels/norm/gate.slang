#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RW(0, x);
PIE_BUFFER_RO(1, gate);

struct Push {
    int width;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width = uint(pc.width);
    if (gid.x >= width) {
        return;
    }
    uint i = gid.y * width + gid.x;
    float g = PIE_LOAD(gate[i]);
    float e = 1.0 / (1.0 + exp(-abs(g)));
    float sig = g < 0.0 ? 1.0 - e : e;
    x[i] = PIE_STORE(PIE_LOAD(x[i]) * sig);
}

// pie:instantiate gate_sigmoid_mul_bf16 PIE_GROUP_X=256
