#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 64
#endif

PIE_BUFFER_RW(0, x);
[[vk::binding(1)]] StructuredBuffer<int> position;

struct Push {
    float base;
    int head_dim;
    float factor;
    float low_dim;
    float high_dim;
    float mscale;
    int interleaved;
    int pair_half;
    int heads;
};
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
    float d = 2.0 * float(i) / float(head_dim);
    float base_freq = exp2(-d * pc.base);
    float denom = pc.high_dim == pc.low_dim ? pc.high_dim + 1e-3 - pc.low_dim
                                            : pc.high_dim - pc.low_dim;
    float ramp = clamp((float(i) - pc.low_dim) / denom, 0.0, 1.0);
    float freq = base_freq * ((1.0 - ramp) + ramp / pc.factor);
    float theta = float(position[row]) * freq;
    float c = cos(theta) * pc.mscale;
    float s = sin(theta) * pc.mscale;
    uint row_base = (row * uint(pc.heads) + h) * head_dim;
    uint i1 = pc.interleaved != 0 ? row_base + 2u * i : row_base + i;
    uint i2 = pc.interleaved != 0 ? i1 + 1u : i1 + pair_half;
    float x1 = PIE_LOAD(x[i1]);
    float x2 = PIE_LOAD(x[i2]);
    x[i1] = PIE_STORE(x1 * c - x2 * s);
    x[i2] = PIE_STORE(x1 * s + x2 * c);
}

// pie:instantiate neox_yarn_mb_bf16 PIE_GROUP_X=64
