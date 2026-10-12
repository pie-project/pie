#include "common/bf16.slang"

// o = bf16(x), one element per invocation.

[[vk::binding(0)]] StructuredBuffer<float> x;
PIE_BUFFER_RW(1, o);

struct Push { uint n; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) return;
    o[i] = PIE_STORE(x[i]);
}

// pie:instantiate cast_f32_to_bf16
