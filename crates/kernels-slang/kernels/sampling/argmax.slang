#include "prim/bf16.slang"
#include "prim/params.slang"

// Each row's argmax, one workgroup a row (the grid's `y`), landed as an i32
// at column `column` of a `depth`-wide plane: the first index of the
// largest value, NaNs skipped (a row of NaNs lands 0). The reduction runs
// through workgroup memory, no subgroup operation. Rows are bf16 two a
// word, or f32 with `PIE_F32`.

#ifdef PIE_F32
[[vk::binding(0)]] StructuredBuffer<float> x;

float x_at(uint i) {
    return x[i];
}
#else
[[vk::binding(0)]] StructuredBuffer<uint> x;

float x_at(uint i) {
    return bf16_at(x, i);
}
#endif
[[vk::binding(1)]] RWStructuredBuffer<int> y;

struct Params {
    int width;
    int depth;
    int column;
};
PIE_PARAMS(2, Params, params);

#define NONE 0xffffffffu

groupshared float best_v[PIE_GROUP_X];
groupshared uint best_i[PIE_GROUP_X];

// Whether candidate (`v`, `i`) beats (`bv`, `bi`): any index beats none,
// then the larger value, then the earlier index.
bool beats(float v, uint i, float bv, uint bi) {
    return i != NONE && (bi == NONE || v > bv || (v == bv && i < bi));
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.y;
    uint lid = local.x;
    uint width = uint(max(params.width, 0));
    float bv = -3.402823466e38;
    uint bi = NONE;
    for (uint c = lid; c < width; c += uint(PIE_GROUP_X)) {
        float v = x_at(row * width + c);
        if (v == v && beats(v, c, bv, bi)) {
            bv = v;
            bi = c;
        }
    }
    best_v[lid] = bv;
    best_i[lid] = bi;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = uint(PIE_GROUP_X) >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride && beats(best_v[lid + stride], best_i[lid + stride], best_v[lid], best_i[lid])) {
            best_v[lid] = best_v[lid + stride];
            best_i[lid] = best_i[lid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0u) {
        uint top = best_i[0];
        y[row * uint(max(params.depth, 1)) + uint(max(params.column, 0))] = int(top == NONE ? 0u : top);
    }
}

// pie:instantiate sampling_argmax_bf16 PIE_GROUP_X=256
// pie:instantiate sampling_argmax_f32 PIE_GROUP_X=256 PIE_F32=1
