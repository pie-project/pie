#include "dit/types.slang"

// One activation or sum per element: o = silu(x), tanh(x), or x + y.

[[vk::binding(0)]] StructuredBuffer<PIE_XT> x;
#ifdef PIE_OP_ADD
[[vk::binding(1)]] StructuredBuffer<PIE_XT> y;
[[vk::binding(2)]] RWStructuredBuffer<PIE_XT> o;
#else
[[vk::binding(1)]] RWStructuredBuffer<PIE_XT> o;
#endif

struct Push { uint n; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= pc.n) return;
    float v = PIE_XL(x[i]);
#if defined(PIE_OP_SILU)
    v = v / (1.0 + exp(-v));
#elif defined(PIE_OP_TANH)
    v = tanh(v);
#else
    v = v + PIE_XL(y[i]);
#endif
    o[i] = PIE_XS(v);
}

// pie:instantiate silu_bf16 PIE_OP_SILU=1
// pie:instantiate silu_f32 PIE_OP_SILU=1 PIE_X_F32=1
// pie:instantiate tanh_bf16 PIE_OP_TANH=1
// pie:instantiate tanh_f32 PIE_OP_TANH=1 PIE_X_F32=1
// pie:instantiate add_bf16 PIE_OP_ADD=1
// pie:instantiate add_f32 PIE_OP_ADD=1 PIE_X_F32=1
