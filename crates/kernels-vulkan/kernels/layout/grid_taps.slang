// Position-table taps per patch: bilinear (kind 0) over its image's grid
// stretched onto a side x side table, or axes (kind 1), one unit tap on the
// column and one on the row offset by `side`.
[[vk::binding(0)]] StructuredBuffer<int> positions;
[[vk::binding(1)]] StructuredBuffer<int> grids;
[[vk::binding(2)]] StructuredBuffer<int> segments;
[[vk::binding(3)]] RWStructuredBuffer<int> ids;
[[vk::binding(4)]] RWStructuredBuffer<float> weights;
struct Push { int kind; int side; int images; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

void axis_taps(int index, int size, int side, out int taps[2], out float w[2]) {
    float src = float(index) * float(side - 1) / float(max(size - 1, 1));
    float fl = floor(src);
    for (int t = 0; t < 2; t++) {
        taps[t] = clamp(int(fl) + t, 0, side - 1);
        w[t] = max(1.0 - abs(src - fl - float(t)), 0.0);
    }
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int n = int(gid.x);
    if (n >= pc.rows) return;
    int side = pc.side;
    int row = positions[n * 3 + 1];
    int col = positions[n * 3 + 2];
    if (pc.kind == 1) {
        ids[n * 2] = min(col, side - 1);
        ids[n * 2 + 1] = side + min(row, side - 1);
        weights[n * 2] = 1.0;
        weights[n * 2 + 1] = 1.0;
        return;
    }
    int lo = 0;
    int hi = pc.images;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (segments[mid] <= n) lo = mid; else hi = mid;
    }
    int h_taps[2], w_taps[2];
    float h_w[2], w_w[2];
    axis_taps(row, grids[lo * 3 + 1], side, h_taps, h_w);
    axis_taps(col, grids[lo * 3 + 2], side, w_taps, w_w);
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            ids[n * 4 + a * 2 + b] = h_taps[a] * side + w_taps[b];
            weights[n * 4 + a * 2 + b] = h_w[a] * w_w[b];
        }
    }
}

// pie:instantiate grid_taps
