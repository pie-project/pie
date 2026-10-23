#include "prim/bf16.slang"
#include "prim/params.slang"

// Image input. `PIE_PIXELS`: raw RGB patch bytes (HWC, four a word) to
// normalized bf16 rows, channel-major (order 0) or pixel-major (order 1),
// the frame repeated `width / (3 patch^2)` times; a word a lane.
// `PIE_GRID_TAPS`: position-table taps per patch, bilinear (kind 0) over its
// image's grid stretched onto a `side x side` table, or axes (kind 1), one
// unit tap on the column and one on the row offset by `side`; a patch a
// lane.

#if defined(PIE_PIXELS)

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    int side_px;
    int width;
    int order;
    float mean0;
    float mean1;
    float mean2;
    float sd0;
    float sd1;
    float sd2;
    int rows;
};
PIE_PARAMS(2, Params, params);

float pixel(uint n, uint c_out) {
    uint pixels = uint(params.side_px * params.side_px);
    uint in_width = 3u * pixels;
    uint temporal = uint(params.width) / in_width;
    uint ch;
    uint at;
    if (params.order == 0) {
        ch = c_out / (temporal * pixels);
        at = (c_out % pixels) * 3u + ch;
    } else {
        at = c_out % in_width;
        ch = at % 3u;
    }
    float mean = ch == 0u ? params.mean0 : (ch == 1u ? params.mean1 : params.mean2);
    float sd = ch == 0u ? params.sd0 : (ch == 1u ? params.sd1 : params.sd2);
    uint byte = n * in_width + at;
    float v = float((x[byte >> 2] >> ((byte & 3u) * 8u)) & 0xffu) / 255.0;
    return (v - mean) / sd;
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = 2u * gid.x;
    uint n = gid.y;
    if (c >= uint(params.width) || n >= uint(params.rows)) {
        return;
    }
    y[(n * uint(params.width) + c) >> 1] = pack_bf16(pixel(n, c), pixel(n, c + 1u));
}

#elif defined(PIE_GRID_TAPS)

[[vk::binding(0)]] StructuredBuffer<int> positions;
[[vk::binding(1)]] StructuredBuffer<int> grids;
[[vk::binding(2)]] StructuredBuffer<int> segments;
[[vk::binding(3)]] RWStructuredBuffer<int> ids;
[[vk::binding(4)]] RWStructuredBuffer<float> weights;

struct Params {
    int kind;
    int side;
    int images;
    int rows;
};
PIE_PARAMS(5, Params, params);

// The two table rows (or columns) an index of a `size`-long grid axis falls
// between, stretched onto the table's side, and their linear weights.
void axis_taps(int index, int size, out int2 taps, out float2 w) {
    int side = params.side;
    float src = float(index) * float(side - 1) / float(max(size - 1, 1));
    float fl = floor(src);
    taps = int2(clamp(int(fl), 0, side - 1), clamp(int(fl) + 1, 0, side - 1));
    w = float2(max(1.0 - abs(src - fl), 0.0), max(1.0 - abs(src - fl - 1.0), 0.0));
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int n = int(gid.x);
    if (n >= params.rows) {
        return;
    }
    int side = params.side;
    int row = positions[n * 3 + 1];
    int col = positions[n * 3 + 2];
    if (params.kind == 1) {
        ids[n * 2] = min(col, side - 1);
        ids[n * 2 + 1] = side + min(row, side - 1);
        weights[n * 2] = 1.0;
        weights[n * 2 + 1] = 1.0;
        return;
    }
    // The image this patch belongs to: the last segment start at or before it.
    int lo = 0;
    int hi = params.images;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (segments[mid] <= n) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    int2 ht;
    float2 hw;
    int2 wt;
    float2 ww;
    axis_taps(row, grids[lo * 3 + 1], ht, hw);
    axis_taps(col, grids[lo * 3 + 2], wt, ww);
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            ids[n * 4 + a * 2 + b] = ht[a] * side + wt[b];
            weights[n * 4 + a * 2 + b] = hw[a] * ww[b];
        }
    }
}

#else
#error "gather/pixels.slang: no kernel selected"
#endif

// pie:instantiate gather_pixels_bf16 PIE_PIXELS=1 PIE_GROUP_X=256
// pie:instantiate gather_pixels_grid_taps PIE_GRID_TAPS=1 PIE_GROUP_X=256
