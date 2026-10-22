#include "prim/params.slang"
#include "prim/rotate.slang"

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

// pie:instantiate elemwise_rope_bf16 PIE_GROUP_X=64
