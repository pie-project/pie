#include "common/bf16.slang"

PIE_BUFFER_RW(0, out_);
#ifdef PIE_BIAS_BF16
[[vk::binding(1)]] StructuredBuffer<uint16_t> bias;
#define PIE_BIAS(i) bf16_to_f32(bias[i])
#else
PIE_BUFFER_RO(1, bias);
#define PIE_BIAS(i) PIE_LOAD(bias[i])
#endif

struct Push { int width; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    if (c >= uint(pc.width)) {
        return;
    }
    uint i = gid.y * uint(pc.width) + c;
    out_[i] = PIE_STORE(PIE_LOAD(out_[i]) + PIE_BIAS(c));
}

// pie:instantiate add_bias_bf16
// pie:instantiate add_bias_f32 PIE_ACT_F32=1
// pie:instantiate add_bias_f32_bf16 PIE_ACT_F32=1 PIE_BIAS_BF16=1
