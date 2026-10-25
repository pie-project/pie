#include "prim/clip.slang"
#include "prim/params.slang"

// Positions from the clip and group tables, a thread a row. `PIE_CELLS`:
// each token row's cell, `width` f32: a row of a clip of `tgrid` its
// patch's (t, h, w) there, any other row positions[row] in column `axis`,
// zeros elsewhere. `PIE_GROUP_ROWS`:
// each row's count of the rows `csr` packs in its lane's group, one f32.

#if defined(PIE_CELLS)
[[vk::binding(0)]] StructuredBuffer<int> tgrid;
[[vk::binding(1)]] StructuredBuffer<int> positions;
[[vk::binding(2)]] RWStructuredBuffer<float> y;

struct Params {
    uint clips;
    uint axis;
    uint width;
    uint rows;
    // The row `tgrid` addresses `y`'s row 0 at: a window of a segmented
    // fire starts `y` at its own rows while the fire's grid holds fire-wide
    // ones.
    uint base;
};
PIE_PARAMS(3, Params, params);
#else
[[vk::binding(0)]] StructuredBuffer<int> csr;
[[vk::binding(1)]] StructuredBuffer<int> groups;
[[vk::binding(2)]] StructuredBuffer<int> lanes;
[[vk::binding(3)]] RWStructuredBuffer<float> y;

struct Params {
    uint lane_count;
    uint group_count;
    uint rows;
};
PIE_PARAMS(4, Params, params);
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint row = gid.x;
    if (row >= params.rows) {
        return;
    }
#if defined(PIE_CELLS)
    uint at = params.width * row;
    for (uint c = 0u; c < params.width; ++c) {
        y[at + c] = 0.0;
    }
    Clip g;
    int fire_row = int(row) + int(params.base);
    if (clip_of(tgrid, int(params.clips), fire_row, g) < 0) {
        y[at + params.axis] = float(positions[row]);
        return;
    }
    Voxel v = unravel(g, fire_row - g.off);
    y[at] = float(v.t);
    y[at + 1u] = float(v.h);
    y[at + 2u] = float(v.w);
#else
    int l = lanes[row];
    int g = (l >= 0 && uint(l) < params.lane_count) ? groups[l] : -1;
    y[row] = (g >= 0 && uint(g) < params.group_count) ? float(csr[g + 1] - csr[g]) : 0.0;
#endif
}

// pie:instantiate conv_cells_rows PIE_CELLS=1
// pie:instantiate conv_cells_group_rows PIE_GROUP_ROWS=1
