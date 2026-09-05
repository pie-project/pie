#include "common/bf16.slang"

PIE_BUFFER_RW(0, out_);
PIE_BUFFER_RO(1, bias);
PIE_BUFFER_RO(2, scale);

struct Push { int width; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint n = gid.y;
    if (c >= uint(pc.width) || n >= uint(pc.rows)) {
        return;
    }
    uint i = n * uint(pc.width) + c;
    out_[i] = PIE_STORE((PIE_LOAD(out_[i]) - PIE_LOAD(bias[c])) * PIE_LOAD(scale[c]));
}

// pie:instantiate standardize_bf16
