#include "common/bf16.slang"

PIE_BUFFER_RW(0, out_);

struct Push {
    int n;
    float value;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= uint(pc.n)) {
        return;
    }
    out_[i] = PIE_STORE(pc.value);
}

// pie:instantiate probe_fill_bf16 PIE_GROUP_X=256
