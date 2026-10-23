#include "prim/params.slang"

// Sinusoidal timestep embeddings in f32, a frequency a lane and a row a grid
// row: `[sin(t f_i) | cos(t f_i)]` (or cos first, `flip_sin_cos`), `f_i =
// max_period^(-i / half)`, the angle scaled by `scale`; an odd width ends in
// a zero.

[[vk::binding(0)]] StructuredBuffer<float> t;
[[vk::binding(1)]] RWStructuredBuffer<float> o;

struct Params {
    uint dim;
    float max_period;
    uint flip_sin_cos;
    float scale;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint half_dim = params.dim / 2u;
    uint i = gid.x;
    uint row = gid.y;
    if (i >= half_dim) {
        return;
    }
    float freq = exp(-log(params.max_period) * float(i) / float(half_dim));
    float angle = params.scale * (t[row] * freq);
    // Fold the angle into one turn: GPU sin and cos are accurate only near zero.
    const float TURN = 6.283185307179586;
    angle = angle - TURN * round(angle / TURN);
    float s = sin(angle);
    float c = cos(angle);
    uint base = row * params.dim;
    o[base + i] = params.flip_sin_cos != 0u ? c : s;
    o[base + i + half_dim] = params.flip_sin_cos != 0u ? s : c;
    if ((params.dim & 1u) != 0u && i == 0u) {
        o[base + params.dim - 1u] = 0.0;
    }
}

// pie:instantiate elemwise_posenc_sinusoid_f32 PIE_GROUP_X=256
