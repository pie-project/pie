#include "prim/pair.slang"
#include "prim/params.slang"

// Per-row modulation, a pair of columns a lane and a row a grid row, each
// row reading the modulation vectors of its own row of `m` or, with a lane
// map, of row `lane_of_row[row]`. `PIE_MODULATE`: `o = x * (1 + scale) +
// shift` (form 0), `x * (1 + scale)` (1) or `tanh(gate) * x` (2), the
// vectors side by side in `m`. `PIE_GATED`: `r += g * y` in place. The
// modulation is bf16, or f32 with `PIE_MOD_F32`; rows are `PIE_ACT`.

#ifdef PIE_MOD_F32
[[vk::binding(1)]] StructuredBuffer<float> m;

float2 mod_at(uint i, bool both) {
    return float2(m[i], both ? m[i + 1u] : 0.0);
}
#else
[[vk::binding(1)]] StructuredBuffer<uint> m;

float2 mod_at(uint i, bool both) {
    return float2(bf16_at(m, i), both ? bf16_at(m, i + 1u) : 0.0);
}
#endif

#if defined(PIE_MODULATE)

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
[[vk::binding(2)]] StructuredBuffer<int> lane_of_row;
[[vk::binding(3)]] RWStructuredBuffer<PIE_ACT> o;

struct Params {
    uint width;
    uint m_width;
    int form;
    uint has_lanes;
};
PIE_PARAMS(4, Params, params);

float2 stable_tanh(float2 v) {
    float2 t = 1.0 - 2.0 / (exp(2.0 * abs(v)) + 1.0);
    return float2(v.x < 0.0 ? -t.x : t.x, v.y < 0.0 ? -t.y : t.y);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = 2u * gid.x;
    uint row = gid.y;
    if (c >= params.width) {
        return;
    }
    bool both = c + 1u < params.width;
    uint at = row * params.width + c;
    uint lane = params.has_lanes != 0u ? uint(lane_of_row[row]) : row;
    uint mat = lane * params.m_width + c;
    float2 xv = pair_at(x, at, both);
    float2 v;
    if (params.form == 0) {
        v = xv * (1.0 + mod_at(mat, both)) + mod_at(mat + params.width, both);
    } else if (params.form == 1) {
        v = xv * (1.0 + mod_at(mat, both));
    } else {
        v = stable_tanh(mod_at(mat, both)) * xv;
    }
    pair_put(o, at, both, v);
}

#elif defined(PIE_GATED)

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> r;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> y;
[[vk::binding(3)]] StructuredBuffer<int> lane_of_row;

struct Params {
    uint width;
    uint has_lanes;
};
PIE_PARAMS(4, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = 2u * gid.x;
    uint row = gid.y;
    if (c >= params.width) {
        return;
    }
    bool both = c + 1u < params.width;
    uint at = row * params.width + c;
    uint lane = params.has_lanes != 0u ? uint(lane_of_row[row]) : row;
    float2 v = mod_at(lane * params.width + c, both) * pair_at(y, at, both) + pair_at(r, at, both);
    pair_put(r, at, both, v);
}

#else
#error "elemwise/modulate.slang: no kernel selected"
#endif

// pie:instantiate elemwise_modulate_bf16 PIE_MODULATE=1 PIE_GROUP_X=256
// pie:instantiate elemwise_modulate_bf16_f32 PIE_MODULATE=1 PIE_GROUP_X=256 PIE_MOD_F32=1
// pie:instantiate elemwise_modulate_f32 PIE_MODULATE=1 PIE_GROUP_X=256 PIE_ACT_F32=1 PIE_MOD_F32=1
// pie:instantiate elemwise_modulate_gated_residual_add_bf16 PIE_GATED=1 PIE_GROUP_X=256
// pie:instantiate elemwise_modulate_gated_residual_add_bf16_f32 PIE_GATED=1 PIE_GROUP_X=256 PIE_MOD_F32=1
// pie:instantiate elemwise_modulate_gated_residual_add_f32 PIE_GATED=1 PIE_GROUP_X=256 PIE_ACT_F32=1 PIE_MOD_F32=1
