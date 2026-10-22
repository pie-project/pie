#include "prim/params.slang"
#include "prim/rotate.slang"

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

// pie:instantiate elemwise_rope_yarn_bf16 PIE_GROUP_X=64
