#include "prim/pair.slang"
#include "prim/params.slang"

// Gated linear units on bf16, a pair of columns a lane. `PIE_PACKED`: each
// row is `[gate | up]`, `intermediate` wide each half, landing an
// `intermediate`-wide row; the grid is (words, rows). Otherwise the gate and
// the up are planes of one shape walked flat (`PIE_GELU`: no up at all), an
// odd tail half there. The activation: SwiGLU, GeGLU (tanh), clamped SwiGLU
// (`PIE_CLAMP`), GPT-OSS's clamped SwiGLU with alpha and `up + 1`
// (`PIE_GPTOSS`), SiTU's soft-capped gate and up (`PIE_SITU`).

float tanh_(float v) {
    float t = 1.0 - 2.0 / (exp(2.0 * abs(v)) + 1.0);
    return v < 0.0 ? -t : t;
}

float gelu_tanh(float v) {
    return 0.5 * v * (1.0 + tanh_(0.7978845608028654 * (v + 0.044715 * v * v * v)));
}

#if defined(PIE_PACKED)

[[vk::binding(0)]] StructuredBuffer<uint> packed;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    uint intermediate;
#if defined(PIE_SITU)
    float beta;
    float up_cap;
#elif defined(PIE_GPTOSS)
    float limit;
    float alpha;
#elif defined(PIE_CLAMP)
    float limit;
#endif
};
PIE_PARAMS(2, Params, params);

float unit(float g, float u) {
#if defined(PIE_SITU)
    float sg = params.beta * tanh_(g / params.beta) / (1.0 + exp(-g));
    if (params.up_cap > 0.0) {
        u = params.up_cap * tanh_(u / params.up_cap);
    }
    return sg * u;
#elif defined(PIE_GEGLU)
    return gelu_tanh(g) * u;
#elif defined(PIE_GPTOSS)
    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    return (g / (1.0 + exp(-params.alpha * g))) * (u + 1.0);
#elif defined(PIE_CLAMP)
    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    return (g / (1.0 + exp(-g))) * u;
#else
    return (g / (1.0 + exp(-g))) * u;
#endif
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint words = params.intermediate / 2u;
    uint i = gid.x;
    if (i >= words) {
        return;
    }
    uint row = gid.y * words;
    uint g = packed[2u * row + i];
    uint u = packed[2u * row + words + i];
    y[row + i] = pack_bf16(
        unit(bf16_to_f32(g & 0xffffu), bf16_to_f32(u & 0xffffu)),
        unit(bf16_to_f32(g >> 16), bf16_to_f32(u >> 16)));
}

#else

#if defined(PIE_GELU)
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    uint n;
};
PIE_PARAMS(2, Params, params);
#else
[[vk::binding(0)]] StructuredBuffer<uint> gate;
[[vk::binding(1)]] StructuredBuffer<uint> up;
[[vk::binding(2)]] RWStructuredBuffer<uint> y;

struct Params {
#if defined(PIE_SITU)
    float beta;
    float up_cap;
#elif defined(PIE_GPTOSS)
    float limit;
    float alpha;
#elif defined(PIE_CLAMP)
    float limit;
#endif
    uint n;
};
PIE_PARAMS(3, Params, params);
#endif

float2 unit(float2 g, float2 u) {
#if defined(PIE_GELU)
    return float2(gelu_tanh(g.x), gelu_tanh(g.y));
#elif defined(PIE_SITU)
    float2 sg = params.beta * float2(tanh_(g.x / params.beta), tanh_(g.y / params.beta))
        / (1.0 + exp(-g));
    if (params.up_cap > 0.0) {
        u = params.up_cap * float2(tanh_(u.x / params.up_cap), tanh_(u.y / params.up_cap));
    }
    return sg * u;
#elif defined(PIE_GPTOSS)
    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    return (g / (1.0 + exp(-params.alpha * g))) * (u + 1.0);
#elif defined(PIE_CLAMP)
    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    return (g / (1.0 + exp(-g))) * u;
#else
    return float2(gelu_tanh(g.x), gelu_tanh(g.y)) * u;
#endif
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    if (i >= params.n) {
        return;
    }
    bool both = i + 1u < params.n;
#if defined(PIE_GELU)
    float2 g = pair_at(x, i, both);
    float2 u = float2(0.0, 0.0);
#else
    float2 g = pair_at(gate, i, both);
    float2 u = pair_at(up, i, both);
#endif
    pair_put(y, i, both, unit(g, u));
}

#endif

// pie:instantiate linear_glu_swiglu_bf16 PIE_PACKED=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_geglu_tanh_packed_bf16 PIE_PACKED=1 PIE_GEGLU=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_swiglu_clamp_bf16 PIE_PACKED=1 PIE_CLAMP=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_swiglu_clamp_alpha_bf16 PIE_PACKED=1 PIE_GPTOSS=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_situ_bf16 PIE_PACKED=1 PIE_SITU=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_geglu_tanh_bf16 PIE_GROUP_X=256
// pie:instantiate linear_glu_swiglu_clamp_split_bf16 PIE_CLAMP=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_swiglu_clamp_alpha_split_bf16 PIE_GPTOSS=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_situ_split_bf16 PIE_SITU=1 PIE_GROUP_X=256
// pie:instantiate linear_glu_gelu_tanh_bf16 PIE_GELU=1 PIE_GROUP_X=256
