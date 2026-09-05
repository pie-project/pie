#include "common/bf16.slang"

PIE_BUFFER_RW(0, out_);
PIE_BUFFER_RO(1, bias);

struct Push { int width; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    if (c >= uint(pc.width)) {
        return;
    }
    uint i = gid.y * uint(pc.width) + c;
    out_[i] = PIE_STORE(PIE_LOAD(out_[i]) + PIE_LOAD(bias[c]));
}

// pie:instantiate add_bias_bf16
