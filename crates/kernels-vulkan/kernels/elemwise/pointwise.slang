#include "common/bf16.slang"

#ifdef PIE_CAST
[[vk::binding(0)]] StructuredBuffer<float> x;
#else
PIE_BUFFER_RO(0, x);
#endif
PIE_BUFFER_RW(1, o);

struct Push { uint n; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) return;
#ifdef PIE_CAST
    o[i] = PIE_STORE(x[i]);
#else
    o[i] = PIE_STORE(tanh(PIE_LOAD(x[i])));
#endif
}

// pie:instantiate tanh_bf16
// pie:instantiate tanh_f32 PIE_ACT_F32=1
// pie:instantiate cast_f32_to_bf16 PIE_CAST=1
