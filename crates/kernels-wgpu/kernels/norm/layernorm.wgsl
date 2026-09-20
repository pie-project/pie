//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;

struct Params {
    eps: f32,
    axis_size: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let n = params.axis_size;
    let base = group.x * n;
    var acc = 0.0;
    for (var i = local.x; i < n; i = i + u32(PIE_GROUP_X)) {
        acc = acc + pie_bf16_at(x[(base + i) >> 1u], base + i);
    }
    let mean = pie_workgroup_sum(local.x, u32(PIE_GROUP_X), acc) / f32(max(n, 1u));
    var spread = 0.0;
    for (var i = local.x; i < n; i = i + u32(PIE_GROUP_X)) {
        let c = pie_bf16_at(x[(base + i) >> 1u], base + i) - mean;
        spread = spread + c * c;
    }
    let inv = inverseSqrt(pie_workgroup_sum(local.x, u32(PIE_GROUP_X), spread) / f32(max(n, 1u)) + params.eps);
    for (var i = 2u * local.x; i < n; i = i + 2u * u32(PIE_GROUP_X)) {
        let xw = x[(base + i) >> 1u];
        let ww = w[i >> 1u];
        let bw = b[i >> 1u];
        let lo = (pie_bf16_to_f32(xw & 0xffffu) - mean) * inv * pie_bf16_to_f32(ww & 0xffffu)
            + pie_bf16_to_f32(bw & 0xffffu);
        let hi = (pie_bf16_to_f32(xw >> 16u) - mean) * inv * pie_bf16_to_f32(ww >> 16u)
            + pie_bf16_to_f32(bw >> 16u);
        out_[(base + i) >> 1u] = pie_pack_bf16(lo, hi);
    }
}

// pie:instantiate layernorm_bf16 PIE_GROUP_X=256
