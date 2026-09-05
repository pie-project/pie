#include "common/bf16.slang"

#define PIE_GROUP 64

PIE_BUFFER_RO(0, o_in);
PIE_BUFFER_RW(1, out_);
[[vk::binding(2)]] StructuredBuffer<float> lse;
PIE_BUFFER_RO(3, sinks);

struct Push { int head_dim; int heads; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

bool pie_finite(float v) {
    return (asuint(v) & 0x7f800000u) != 0x7f800000u;
}

[numthreads(PIE_GROUP, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    static const float kLn2 = 0.69314718055994530942;
    uint d = gid.x;
    uint h = gid.y;
    uint t = gid.z;
    if (d >= uint(pc.head_dim)) return;
    float r = 1.0;
    float lse_val = lse[t * uint(pc.heads) + h];
    if (pie_finite(lse_val)) {
        float diff = lse_val * kLn2 - PIE_LOAD(sinks[h]);
        r = 1.0 / (1.0 + exp(-diff));
    }
    uint i = (t * uint(pc.heads) + h) * uint(pc.head_dim) + d;
    out_[i] = PIE_STORE(PIE_LOAD(o_in[i]) * r);
}

// pie:instantiate attn_sink_rescale_bfloat16
