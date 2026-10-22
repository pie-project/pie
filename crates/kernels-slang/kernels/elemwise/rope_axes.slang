#include "prim/bf16.slang"
#include "prim/params.slang"

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

// pie:instantiate elemwise_rope_axes_bf16 PIE_GROUP_X=256
// pie:instantiate elemwise_rope_axes_f32 PIE_GROUP_X=256 PIE_ACT_F32=1
