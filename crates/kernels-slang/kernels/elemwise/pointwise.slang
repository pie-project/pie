#include "prim/pair.slang"
#include "prim/params.slang"

// Elementwise kernels with no algorithm of their own, a pair of elements a
// lane (`prim/pair.slang`): sums, a bias, scalars, activations, the f32 to
// bf16 cast, clamps and the sigmoid gate. `PIE_<KERNEL>` picks the kernel.

#if defined(PIE_ADD)

// `z = x + y` over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> y;
[[vk::binding(2)]] RWStructuredBuffer<PIE_ACT> z;

struct Params {
    uint n;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(z, i, both, pair_at(x, i, both) + pair_at(y, i, both));
}

#elif defined(PIE_RESIDUAL_ADD)

// `y += x` over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] RWStructuredBuffer<PIE_ACT> y;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(y, i, both, pair_at(y, i, both) + pair_at(x, i, both));
}

#elif defined(PIE_ADD_BIAS)

// `out[r, c] += bias[c]`, a pair of columns a lane and a row a grid row. The
// bias rides the row's element, or is bf16 under f32 rows (`PIE_BIAS_BF16`).

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> out_;
#ifdef PIE_BIAS_BF16
[[vk::binding(1)]] StructuredBuffer<uint> bias;

float2 bias_pair(uint c, bool both) {
    return float2(bf16_at(bias, c), both ? bf16_at(bias, c + 1u) : 0.0);
}
#else
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> bias;

float2 bias_pair(uint c, bool both) {
    return pair_at(bias, c, both);
}
#endif

struct Params {
    int width;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width = uint(max(params.width, 0));
    uint c = 2u * gid.x;
    if (c >= width) {
        return;
    }
    bool both = c + 1u < width;
    uint i = gid.y * width + c;
    pair_put(out_, i, both, pair_at(out_, i, both) + bias_pair(c, both));
}

#elif defined(PIE_SCALE)

// `x *= s` in place over a whole tensor, a pair of elements a lane. The
// scalar is stated, rounded as the activation would hold it, or with
// `PIE_READ` read from a one-element tensor of the activation's element.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;
#ifdef PIE_READ
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> s;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);

float scalar() {
    return pair_at(s, 0u, false).x;
}
#else
struct Params {
    float s;
    uint n;
};
PIE_PARAMS(1, Params, params);

float scalar() {
    return act_round(params.s);
}
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(x, i, both, pair_at(x, i, both) * scalar());
}

#elif defined(PIE_SILU_SCALED)

// `x = silu(x * s)` in place over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;

struct Params {
    float s;
    uint n;
};
PIE_PARAMS(1, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    float2 v = pair_at(x, i, both) * params.s;
    pair_put(x, i, both, v / (1.0 + exp(-v)));
}

#elif defined(PIE_TANH)

// `x = tanh(x)` in place over a whole tensor, a pair of elements a lane.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;

struct Params {
    uint n;
};
PIE_PARAMS(1, Params, params);

// `1 - 2 / (e^2|v| + 1)` with the sign put back: the exponential may
// overflow to infinity and the quotient still lands on 1.
float2 stable_tanh(float2 v) {
    float2 t = 1.0 - 2.0 / (exp(2.0 * abs(v)) + 1.0);
    return float2(v.x < 0.0 ? -t.x : t.x, v.y < 0.0 ? -t.y : t.y);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(x, i, both, stable_tanh(pair_at(x, i, both)));
}

#elif defined(PIE_CAST)

// f32 values rounded into bf16 ones, a pair a lane.

[[vk::binding(0)]] StructuredBuffer<float> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> o;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(o, i, both, float2(x[i], both ? x[i + 1u] : 0.0));
}

#elif defined(PIE_CLAMP)

// `x = min(max(x, lo), hi)` in place over a whole tensor, a pair of elements
// a lane. The bounds are stated, rounded as the activation would hold them,
// or with `PIE_LEARNED` read from one-element tensors of its element.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;
#ifdef PIE_LEARNED
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> lo_t;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> hi_t;

struct Params {
    uint n;
};
PIE_PARAMS(3, Params, params);

float2 bounds() {
    return float2(pair_at(lo_t, 0u, false).x, pair_at(hi_t, 0u, false).x);
}
#else
struct Params {
    float lo;
    float hi;
    uint n;
};
PIE_PARAMS(1, Params, params);

float2 bounds() {
    return float2(act_round(params.lo), act_round(params.hi));
}
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    float2 b = bounds();
    pair_put(x, i, both, min(max(pair_at(x, i, both), b.x), b.y));
}

#elif defined(PIE_SIGMOID_MUL)

// `x *= sigmoid(gate)` in place, the gate a tensor of `x`'s own shape, a
// pair of elements a lane.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> x;
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> gate;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);

float2 sigmoid(float2 g) {
    float2 e = 1.0 / (1.0 + exp(-abs(g)));
    return float2(g.x < 0.0 ? 1.0 - e.x : e.x, g.y < 0.0 ? 1.0 - e.y : e.y);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
    pair_put(x, i, both, pair_at(x, i, both) * sigmoid(pair_at(gate, i, both)));
}

#else
#error "elemwise/pointwise.slang: no kernel selected"
#endif

// pie:instantiate elemwise_pointwise_add_bf16 PIE_ADD=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_add_f32 PIE_ADD=1 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_pointwise_residual_add_bf16 PIE_RESIDUAL_ADD=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_residual_add_f32 PIE_RESIDUAL_ADD=1 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_pointwise_add_bias_bf16 PIE_ADD_BIAS=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_add_bias_f32 PIE_ADD_BIAS=1 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_pointwise_add_bias_f32_bf16 PIE_ADD_BIAS=1 PIE_GROUP_X=256 PIE_ACT_F32=1 PIE_BIAS_BF16=1
// pie:instantiate elemwise_pointwise_scale_bf16 PIE_SCALE=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_scale_f32 PIE_SCALE=1 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_pointwise_scale_read_bf16 PIE_SCALE=1 PIE_GROUP_X=256 PIE_READ=1
// pie:instantiate elemwise_pointwise_silu_scaled_bf16 PIE_SILU_SCALED=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_silu_scaled_f32 PIE_SILU_SCALED=1 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_pointwise_tanh_bf16 PIE_TANH=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_tanh_f32 PIE_TANH=1 PIE_GROUP_X=256 PIE_ACT_F32=1
// pie:instantiate elemwise_pointwise_cast_f32_to_bf16 PIE_CAST=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_clamp_bf16 PIE_CLAMP=1 PIE_GROUP_X=256
// pie:instantiate elemwise_pointwise_clamp_learned_bf16 PIE_CLAMP=1 PIE_GROUP_X=256 PIE_LEARNED=1
// pie:instantiate elemwise_pointwise_sigmoid_mul_bf16 PIE_SIGMOID_MUL=1 PIE_GROUP_X=256
