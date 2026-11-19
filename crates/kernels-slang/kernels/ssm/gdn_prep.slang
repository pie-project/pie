#include "prim/bf16.slang"
#include "prim/params.slang"
#if defined(PIE_REPLAY)
#include "prim/pool.slang"
#endif

// The gated delta rule's per-head gates from the fused `[b | a]` projection,
// a value head a lane and a row a grid row: `g = -exp(a_log) *
// softplus(a + dt_bias)` and `beta = sigmoid(b)`, landed as the f32 row
// `[g | beta]` beside the projection it is derived from.
//
// `PIE_REPLAY` is the decode step of a lane that carries rows
// (`ssm/conv1d.slang` says how `lag` and the rs-buffer rows are laid): row
// `r` is lane `lane0 + r`'s own row, whose projection also lands in its
// rs-buffer row (when it names one), and the gates of its carried rows,
// read from the rs buffer, land in `carried`, lane `l`'s row `j` at row
// `l * carry + j`.

[[vk::binding(0)]] StructuredBuffer<uint> ba;
[[vk::binding(1)]] StructuredBuffer<float> a_log;
[[vk::binding(2)]] StructuredBuffer<uint> dt_bias;
[[vk::binding(3)]] RWStructuredBuffer<float> gates;
#if defined(PIE_REPLAY)
[[vk::binding(4)]] StructuredBuffer<int> lag;
[[vk::binding(5)]] RWStructuredBuffer<float> carried;
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 6)
#define PARAMS_AT 10
#else
#define PARAMS_AT 4
#endif

struct Params {
    int v_heads;
#if defined(PIE_REPLAY)
    uint seg_words;
    uint lane0;
    uint carry;
    uint page_tokens;
    uint page_words;
    uint plane_words;
    uint row_words;
#endif
};
PIE_PARAMS(PARAMS_AT, Params, params);
#if defined(PIE_REPLAY)
PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)
#endif

float2 gate(float bv, float av, uint h) {
    float z = av + bf16_at(dt_bias, h);
    float sp = z > 20.0 ? z : log(1.0 + exp(z));
    return float2(-exp(a_log[h]) * sp, 1.0 / (1.0 + exp(-bv)));
}

#if defined(PIE_REPLAY)
// Element `i` of rs-buffer row `unit`'s projection.
float rs_at(int unit, uint i) {
    uint u = uint(unit);
    uint page = u / params.page_tokens;
    uint token = u % params.page_tokens;
    uint word = pie_pool_get(page * params.page_words + params.plane_words + token * params.row_words + (i >> 1));
    return bf16_to_f32((i & 1u) != 0u ? (word >> 16) : (word & 0xffffu));
}
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint h = gid.x;
    uint t = gid.y;
    uint vh = uint(params.v_heads);
    if (h >= vh) {
        return;
    }
    uint row = t * 2u * vh;
    float2 own = gate(bf16_at(ba, row + h), bf16_at(ba, row + vh + h), h);
    gates[row + h] = own.x;
    gates[row + vh + h] = own.y;
#if defined(PIE_REPLAY)
    uint lane = params.lane0 + t;
    uint entry = lane * (3u + params.carry);
    int carries = lag[entry];
    int at = lag[entry + 2u];
    // The projection row is `vh` words: a lane lands one.
    if (at >= 0) {
        uint u = uint(at);
        uint page = u / params.page_tokens;
        uint token = u % params.page_tokens;
        pie_pool_set(page * params.page_words + params.plane_words + token * params.row_words + h, ba[t * vh + h]);
    }
    for (int s = 0; s < carries; s++) {
        int unit = lag[entry + 3u + uint(s)];
        float2 held = gate(rs_at(unit, h), rs_at(unit, vh + h), h);
        uint into = (lane * params.carry + uint(s)) * 2u * vh;
        carried[into + h] = held.x;
        carried[into + vh + h] = held.y;
    }
#endif
}

// pie:instantiate ssm_gdn_prep_bf16 PIE_GROUP_X=256
// pie:instantiate ssm_gdn_prep_replay_bf16 PIE_GROUP_X=256 PIE_REPLAY=1
