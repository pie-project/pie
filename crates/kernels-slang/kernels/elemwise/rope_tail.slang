#include "prim/params.slang"
#include "prim/rotate.slang"

// Rotary embedding of each head's last `2 * pair_half` columns, in place: by
// halves or, `interleaved`, by adjacent pairs; `sign` -1 turns back. With
// `factor > 1` a YaRN ramp blends each frequency between itself and itself
// over `factor`, from pair `low` to pair `high`. A lane turns two pairs; the
// grid is (pair lanes, heads, rows).

[[vk::binding(0)]] RWStructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> position;

struct Params {
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
PIE_PARAMS(2, Params, params);

float inv_freq(uint i, uint rotary) {
    float f = exp2(-2.0 * float(i) / float(max(rotary, 1u)) * params.base);
    if (params.factor > 1.0) {
        float ramp = clamp((float(i) - params.low) / max(params.high - params.low, 0.001), 0.0, 1.0);
        float lag = 1.0 - ramp;
        f = f / params.factor * (1.0 - lag) + f * lag;
    }
    return f;
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint pair_half = uint(max(params.pair_half, 0));
    uint i = 2u * gid.x;
    if (i >= pair_half) {
        return;
    }
    uint head_dim = uint(max(params.head_dim, 1));
    uint rotary = 2u * pair_half;
    uint head = (gid.z * uint(max(params.heads, 0)) + gid.y) * head_dim + (head_dim - rotary);
    float pos = params.sign * float(position[gid.z]);
    float4 cs = turns(pos * inv_freq(i, rotary), pos * inv_freq(i + 1u, rotary));
    if (params.interleaved != 0) {
        rotate_adjacent(x, head + 2u * i, cs);
    } else {
        rotate_halves(x, head + i, head + i + pair_half, cs);
    }
}

// pie:instantiate elemwise_rope_tail_bf16 PIE_GROUP_X=64
