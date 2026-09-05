#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 64
#endif

PIE_BUFFER_RW(0, x);
[[vk::binding(1)]] StructuredBuffer<int> position;

#if defined(PIE_LAST)

struct Push {
    float base;
    int head_dim;
    int interleaved;
    float sign;
    float factor;
    float low;
    float high;
    int pair_half;
    int heads;
};
#else
struct Push {
    float scale;
    float base;
    int head_dim;
    int pair_half;
    int heads;
};
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    uint h = gid.y;
    uint row = gid.z;
    uint pair_half = uint(pc.pair_half);
    if (i >= pair_half) {
        return;
    }
    uint head_dim = uint(pc.head_dim);
    uint row_base = row * uint(pc.heads) * head_dim;
#if defined(PIE_LAST)
    uint rotary = 2u * pair_half;
    uint offset = head_dim - rotary;
    float d = 2.0 * float(i) / float(rotary);
    float inv_freq = exp2(-d * pc.base);
    if (pc.factor > 1.0) {
        float span = max(pc.high - pc.low, 0.001);
        float ramp = clamp((float(i) - pc.low) / span, 0.0, 1.0);
        float smooth = 1.0 - ramp;
        inv_freq = inv_freq / pc.factor * (1.0 - smooth) + inv_freq * smooth;
    }
    float tl = pc.sign * float(position[row]) * inv_freq;
    float cl = cos(tl);
    float sl = sin(tl);
    uint head_base = row_base + h * head_dim + offset;
    uint j1 = pc.interleaved != 0 ? head_base + 2u * i : head_base + i;
    uint j2 = pc.interleaved != 0 ? j1 + 1u : j1 + pair_half;
    float xa = PIE_LOAD(x[j1]);
    float xb = PIE_LOAD(x[j2]);
    x[j1] = PIE_STORE(xa * cl - xb * sl);
    x[j2] = PIE_STORE(xa * sl + xb * cl);
    return;
#elif defined(PIE_PROP)
    float d = 2.0 * float(i) / float(head_dim);
    uint i2_off = head_dim / 2u;
#else
    float d = float(i) / float(pair_half);
    uint i2_off = pair_half;
#endif
#if !defined(PIE_LAST)
    float theta = pc.scale * float(position[row]) * exp2(-d * pc.base);
    float c = cos(theta);
    float s = sin(theta);
    uint i1 = row_base + h * head_dim + i;
    uint i2 = i1 + i2_off;
    float x1 = PIE_LOAD(x[i1]);
    float x2 = PIE_LOAD(x[i2]);
    x[i1] = PIE_STORE(x1 * c - x2 * s);
    x[i2] = PIE_STORE(x1 * s + x2 * c);
#endif
}

// pie:instantiate neox_mb_bf16 PIE_GROUP_X=64
// pie:instantiate neox_prop_mb_bf16 PIE_GROUP_X=64 PIE_PROP=1
// pie:instantiate neox_last_mb_bf16 PIE_GROUP_X=64 PIE_LAST=1
