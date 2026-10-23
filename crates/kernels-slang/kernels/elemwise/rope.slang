#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/rotate.slang"

// Rotary embeddings, in place on rows: neox halves at a span, the rotated
// tail of each head, whole-head YaRN, multimodal (M-RoPE) sections, and up
// to four position axes. `PIE_<KERNEL>` picks the kernel.

#if defined(PIE_NEOX)

// Rotary embedding by halves, in place: pair `i` of each head is
// `(i, i + span)` turned by `position * theta^(-i / span)` (`base` is
// `log2(theta)`); `span` is the rotated width's half for a plain rotary, the
// head's half for a proportional one. A lane turns pairs `2g` and `2g + 1`;
// the grid is (pair lanes, heads, rows).

[[vk::binding(0)]] RWStructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> position;

struct Params {
    float base;
    int head_dim;
    int span;
    int pair_half;
    int heads;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= uint(max(params.pair_half, 0))) {
        return;
    }
    uint head_dim = uint(max(params.head_dim, 1));
    uint span = uint(max(params.span, 1));
    float pos = float(position[gid.z]);
    float t0 = pos * exp2(-float(i) / float(span) * params.base);
    float t1 = pos * exp2(-float(i + 1u) / float(span) * params.base);
    uint i1 = (gid.z * uint(max(params.heads, 0)) + gid.y) * head_dim + i;
    rotate_halves(x, i1, i1 + span, turns(t0, t1));
}

#elif defined(PIE_TAIL)

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

#elif defined(PIE_YARN)

// YaRN rotary over whole heads, in place: each pair's frequency ramps from
// itself (below `low_dim`) to itself over `factor` (above `high_dim`), and
// the turn is scaled by `mscale`; by halves or, `interleaved`, by adjacent
// pairs. A lane turns two pairs; the grid is (pair lanes, heads, rows).

[[vk::binding(0)]] RWStructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> position;

struct Params {
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
PIE_PARAMS(2, Params, params);

float theta_at(uint i, uint head_dim, float pos) {
    float base_freq = exp2(-2.0 * float(i) / float(head_dim) * params.base);
    float denom = params.high_dim == params.low_dim ? 1e-3 : params.high_dim - params.low_dim;
    float ramp = clamp((float(i) - params.low_dim) / denom, 0.0, 1.0);
    return pos * base_freq * ((1.0 - ramp) + ramp / params.factor);
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
    float pos = float(position[gid.z]);
    uint head = (gid.z * uint(max(params.heads, 0)) + gid.y) * head_dim;
    float4 cs = turns(theta_at(i, head_dim, pos), theta_at(i + 1u, head_dim, pos)) * params.mscale;
    if (params.interleaved != 0) {
        rotate_adjacent(x, head + 2u * i, cs);
    } else {
        rotate_halves(x, head + i, head + i + pair_half, cs);
    }
}

#elif defined(PIE_MROPE)

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

#elif defined(PIE_AXES)

// Rotary over up to four position axes, in place, one rotation pair a lane:
// each axis owns `dims[a]` channels of the rotated prefix and turns them by
// its own position and `thetas[a]`. The pair is adjacent channels
// (interleaved, `form` 0), halves of the prefix (neox, 1), halves of the
// axis's own block (split, 2), or, the split ladder (3), axes cycling over
// the rotated channels of every head after a centred pad. Positions are f32,
// one row of `axes` a token. A pair may sit a bf16 half from a neighbour's,
// so bf16 halves are written through atomics on their words.

#ifdef PIE_ACT_F32
[[vk::binding(0)]] RWStructuredBuffer<float> x;

float x_at(uint i) {
    return x[i];
}

void x_put(uint i, float v) {
    x[i] = v;
}
#else
[[vk::binding(0)]] RWStructuredBuffer<Atomic<uint>> x;

float x_at(uint i) {
    uint word = x[i >> 1].load();
    return bf16_to_f32((i & 1u) != 0u ? (word >> 16) : (word & 0xffffu));
}

void x_put(uint i, float v) {
    uint shift = (i & 1u) * 16u;
    x[i >> 1].and(~(0xffffu << shift));
    x[i >> 1].or(f32_to_bf16(v) << shift);
}
#endif
[[vk::binding(1)]] StructuredBuffer<float> positions;

struct Params {
    uint4 dims;
    float4 thetas;
    uint axes;
    uint rotary_dim;
    uint head_dim;
    uint heads;
    int form;
};
PIE_PARAMS(2, Params, params);

float theta_pow(float base, float e) {
    return exp(e * log(base));
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint angles = params.rotary_dim / 2u;
    uint idx = gid.x;
    if (idx >= params.heads * angles) {
        return;
    }
    uint row = gid.y;
    uint pos = row * params.axes;
    uint head = idx / angles;
    uint angle = idx % angles;
    uint xh = (row * params.heads + head) * params.head_dim;
    uint lo;
    uint hi;
    float cos_v = 1.0;
    float sin_v = 0.0;

    if (params.form == 3) {
        uint span = 0u;
        for (uint a = 0u; a < params.axes; a++) {
            span += params.dims[a];
        }
        uint pad = (params.heads * params.rotary_dim - span) / 2u;
        if (idx >= pad) {
            uint slot = idx - pad;
            uint axis = slot % params.axes;
            uint f = slot / params.axes;
            uint ladder = params.dims[axis] / 2u;
            float exponent = ladder > 1u ? float(f) / float(ladder - 1u) : 0.0;
            float a = positions[pos + axis] * theta_pow(params.thetas[axis], exponent);
            sin_v = sin(a);
            cos_v = cos(a);
        }
        lo = angle;
        hi = angle + angles;
    } else {
        uint axis = 0u;
        uint first_angle = 0u;
        uint first_channel = 0u;
        while (axis < params.axes && angle >= first_angle + params.dims[axis] / 2u) {
            first_angle += params.dims[axis] / 2u;
            first_channel += params.dims[axis];
            axis++;
        }
        if (axis >= params.axes) {
            return;
        }
        uint within = angle - first_angle;
        float freq = theta_pow(params.thetas[axis], -2.0 * float(within) / float(params.dims[axis]));
        float a = positions[pos + axis] * freq;
        sin_v = sin(a);
        cos_v = cos(a);
        if (params.form == 0) {
            lo = first_channel + 2u * within;
            hi = lo + 1u;
        } else if (params.form == 1) {
            lo = angle;
            hi = angle + angles;
        } else {
            lo = first_channel + within;
            hi = first_channel + params.dims[axis] / 2u + within;
        }
    }
    float va = x_at(xh + lo);
    float vb = x_at(xh + hi);
    x_put(xh + lo, va * cos_v - vb * sin_v);
    x_put(xh + hi, vb * cos_v + va * sin_v);
}

#else
#error "elemwise/rope.slang: no kernel selected"
#endif

// pie:instantiate elemwise_rope_neox_bf16 PIE_NEOX=1 PIE_GROUP_X=64
// pie:instantiate elemwise_rope_tail_bf16 PIE_TAIL=1 PIE_GROUP_X=64
// pie:instantiate elemwise_rope_yarn_bf16 PIE_YARN=1 PIE_GROUP_X=64
// pie:instantiate elemwise_rope_mrope_interleaved_bf16 PIE_MROPE=1 PIE_GROUP_X=64
// pie:instantiate elemwise_rope_mrope_blocked_bf16 PIE_MROPE=1 PIE_GROUP_X=64 PIE_BLOCKED=1
// pie:instantiate elemwise_rope_mrope_split_bf16 PIE_MROPE=1 PIE_GROUP_X=64 PIE_SPLIT=1
// pie:instantiate elemwise_rope_axes_bf16 PIE_AXES=1 PIE_GROUP_X=256
// pie:instantiate elemwise_rope_axes_f32 PIE_AXES=1 PIE_GROUP_X=256 PIE_ACT_F32=1
