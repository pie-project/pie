//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<f32>;

struct Params {
    dim: i32,
    eps: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let dim = u32(max(params.dim, 0));
    let base = group.x * dim;
    var acc = 0.0;
    for (var d = local.x; d < dim; d = d + u32(PIE_GROUP_X)) {
        let v = pie_bf16_at(x[(base + d) >> 1u], base + d);
        acc = acc + v * v;
    }
    let inv = pie_inv_rms(local.x, u32(PIE_GROUP_X), acc, dim, params.eps);
    for (var d = local.x; d < dim; d = d + u32(PIE_GROUP_X)) {
        out_[base + d] = pie_bf16_at(x[(base + d) >> 1u], base + d) * inv;
    }
}

// pie:instantiate hc_rmsnorm_f32_bf16 PIE_GROUP_X=256
