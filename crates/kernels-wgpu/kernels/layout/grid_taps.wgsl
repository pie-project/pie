// Position-table taps per patch: bilinear (kind 0) over its image's grid
// stretched onto a side x side table, or axes (kind 1), one unit tap on the
// column and one on the row offset by `side`.
@group(0) @binding(0) var<storage, read> positions: array<i32>;
@group(0) @binding(1) var<storage, read> grids: array<i32>;
@group(0) @binding(2) var<storage, read> segments: array<i32>;
@group(0) @binding(3) var<storage, read_write> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
struct Params {
    kind: i32,
    side: i32,
    images: i32,
    rows: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

struct AxisTaps {
    taps: vec2<i32>,
    w: vec2<f32>,
}

fn axis_taps(index: i32, size: i32) -> AxisTaps {
    let side = params.side;
    let src = f32(index) * f32(side - 1) / f32(max(size - 1, 1));
    let fl = floor(src);
    var r: AxisTaps;
    for (var t = 0; t < 2; t = t + 1) {
        r.taps[t] = clamp(i32(fl) + t, 0, side - 1);
        r.w[t] = max(1.0 - abs(src - fl - f32(t)), 0.0);
    }
    return r;
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(gid.x);
    if (n >= params.rows) {
        return;
    }
    let side = params.side;
    let row = positions[n * 3 + 1];
    let col = positions[n * 3 + 2];
    if (params.kind == 1) {
        ids[n * 2] = min(col, side - 1);
        ids[n * 2 + 1] = side + min(row, side - 1);
        weights[n * 2] = 1.0;
        weights[n * 2 + 1] = 1.0;
        return;
    }
    var lo = 0;
    var hi = params.images;
    while (hi - lo > 1) {
        let mid = (lo + hi) / 2;
        if (segments[mid] <= n) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let h = axis_taps(row, grids[lo * 3 + 1]);
    let w = axis_taps(col, grids[lo * 3 + 2]);
    for (var a = 0; a < 2; a = a + 1) {
        for (var b = 0; b < 2; b = b + 1) {
            ids[n * 4 + a * 2 + b] = h.taps[a] * side + w.taps[b];
            weights[n * 4 + a * 2 + b] = h.w[a] * w.w[b];
        }
    }
}

// pie:instantiate grid_taps PIE_GROUP_X=256
