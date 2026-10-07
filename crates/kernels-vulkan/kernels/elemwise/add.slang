#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, y);
PIE_BUFFER_RW(2, z);

struct Push { uint n; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) {
        return;
    }
    z[i] = PIE_STORE(PIE_LOAD(x[i]) + PIE_LOAD(y[i]));
}

// pie:instantiate add_bf16
