#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, residual);
PIE_BUFFER_RW(2, out_);

struct Push { uint n; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) {
        return;
    }
    out_[i] = PIE_STORE(PIE_LOAD(x[i]) + PIE_LOAD(residual[i]));
}

// pie:instantiate residual_add_bf16
