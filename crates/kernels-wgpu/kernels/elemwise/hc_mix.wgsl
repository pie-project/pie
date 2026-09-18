//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read> gates: array<u32>;
@group(0) @binding(1) var<storage, read> normed: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<u32>;

struct Params {
    n_rows: i32,
    m: i32,
    h: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

fn sigmoid(v: f32) -> f32 {
    return 1.0 / (1.0 + exp(-v));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = u32(max(params.m, 0));
    let H = u32(max(params.h, 0));
    let h = 2u * gid.x;
    let n = gid.y;
    if (h >= H || n >= u32(max(params.n_rows, 0))) {
        return;
    }
    let base = n * M * H + h;
    var lo = 0.0;
    var hi = 0.0;
    for (var s = 0u; s < M; s = s + 1u) {
        let g = gates[(base + s * H) >> 1u];
        let v = normed[(base + s * H) >> 1u];
        lo = lo + pie_bf16_to_f32(v & 0xffffu) * sigmoid(pie_bf16_to_f32(g & 0xffffu));
        hi = hi + pie_bf16_to_f32(v >> 16u) * sigmoid(pie_bf16_to_f32(g >> 16u));
    }
    let inv = 1.0 / f32(max(M, 1u));
    y[(n * H + h) >> 1u] = pie_pack_bf16(lo * inv, hi * inv);
}

// pie:instantiate hc_mix_bf16
