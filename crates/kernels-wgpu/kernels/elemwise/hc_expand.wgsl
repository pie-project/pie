@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct Params {
    n_rows: i32,
    m: i32,
    h: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let words = u32(max(params.h, 0)) / 2u;
    let w = gid.x;
    let n = gid.y;
    if (w >= words || n >= u32(max(params.n_rows, 0))) {
        return;
    }
    let m = u32(max(params.m, 0));
    let v = x[n * words + w];
    let base = n * m * words + w;
    for (var s = 0u; s < m; s = s + 1u) {
        out_[base + s * words] = v;
    }
}

// pie:instantiate hc_expand_bf16
