#include "prim/bf16.slang"
#include "prim/params.slang"

// Multimodal rotary (M-RoPE) in place, one rotation pair a lane: each pair
// turns by one of a (t, h, w) position triple, picked per pair by the
// section layout. Interleaved, axes cycle t, h, w over the first `3 * s1`
// and `3 * s2` pairs; blocked (`PIE_BLOCKED`), the sections tile one ladder
// in order; split (`PIE_SPLIT`), each section is its own halves-rotation
// over its own columns. Pairs may sit a bf16 half apart from a neighbour's,
// so a lane writes its two values through atomics on their words.

[[vk::binding(0)]] RWStructuredBuffer<Atomic<uint>> x;
[[vk::binding(1)]] StructuredBuffer<int> positions;

struct Params {
    float base;
    int head_dim;
    int s0;
    int s1;
    int s2;
    int heads;
    int pairs;
};
PIE_PARAMS(2, Params, params);

float load_at(uint i) {
    uint word = x[i >> 1].load();
    return bf16_to_f32((i & 1u) != 0u ? (word >> 16) : (word & 0xffffu));
}

void store_at(uint i, float v) {
    uint shift = (i & 1u) * 16u;
    x[i >> 1].and(~(0xffffu << shift));
    x[i >> 1].or(f32_to_bf16(v) << shift);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int i = int(gid.x);
    if (i >= params.pairs) {
        return;
    }
    uint m = gid.z;
    int3 pos = int3(positions[3u * m], positions[3u * m + 1u], positions[3u * m + 2u]);
    uint row = (m * uint(params.heads) + gid.y) * uint(params.head_dim);
    int axis_pos;
    float inv_freq;
    uint i1;
    uint i2;
#if defined(PIE_BLOCKED)
    int within = i;
    axis_pos = pos.x;
    if (i >= params.s0 + params.s1) {
        axis_pos = pos.z;
        within = i - params.s0 - params.s1;
    } else if (i >= params.s0) {
        axis_pos = pos.y;
        within = i - params.s0;
    }
    inv_freq = exp2(-(2.0 * float(within) / float(params.s0 + params.s1 + params.s2)) * params.base);
    i1 = row + uint(i);
    i2 = i1 + uint(params.head_dim / 2);
#elif defined(PIE_SPLIT)
    int within = i;
    int before = 0;
    int width = params.s0;
    axis_pos = pos.x;
    if (i >= params.s0 + params.s1) {
        axis_pos = pos.z;
        within = i - params.s0 - params.s1;
        before = params.s0 + params.s1;
        width = params.s2;
    } else if (i >= params.s0) {
        axis_pos = pos.y;
        within = i - params.s0;
        before = params.s0;
        width = params.s1;
    }
    if (width <= 0) {
        return;
    }
    inv_freq = exp2(-(float(within) / float(width)) * params.base);
    i1 = row + uint(2 * before + within);
    i2 = i1 + uint(width);
#else
    int r = i % 3;
    axis_pos = pos.x;
    if (r == 1 && i < 3 * params.s1) {
        axis_pos = pos.y;
    } else if (r == 2 && i < 3 * params.s2) {
        axis_pos = pos.z;
    }
    inv_freq = exp2(-(2.0 * float(i) / float(2 * params.pairs)) * params.base);
    i1 = row + uint(i);
    i2 = i1 + uint(params.pairs);
#endif
    float theta = float(axis_pos) * inv_freq;
    float c = cos(theta);
    float s = sin(theta);
    float x1 = load_at(i1);
    float x2 = load_at(i2);
    store_at(i1, x1 * c - x2 * s);
    store_at(i2, x1 * s + x2 * c);
}

// pie:instantiate elemwise_rope_mrope_interleaved_bf16 PIE_GROUP_X=64
// pie:instantiate elemwise_rope_mrope_blocked_bf16 PIE_GROUP_X=64 PIE_BLOCKED=1
// pie:instantiate elemwise_rope_mrope_split_bf16 PIE_GROUP_X=64 PIE_SPLIT=1
