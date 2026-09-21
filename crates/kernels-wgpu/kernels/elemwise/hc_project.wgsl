//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read> normed: array<f32>;
@group(0) @binding(1) var<storage, read> hc_fn: array<f32>;
@group(0) @binding(2) var<storage, read_write> mixes: array<f32>;

struct Params {
    fan_in: i32,
    mix_hc: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let mix_hc = u32(max(params.mix_hc, 1));
    let fan = u32(max(params.fan_in, 0));
    let o = group.x % mix_hc;
    let n = group.x / mix_hc;
    var acc = 0.0;
    for (var d = local.x; d < fan; d = d + u32(PIE_GROUP_X)) {
        acc = acc + normed[n * fan + d] * hc_fn[o * fan + d];
    }
    let total = pie_workgroup_sum(local.x, u32(PIE_GROUP_X), acc);
    if (local.x == 0u) {
        mixes[n * mix_hc + o] = total;
    }
}

// pie:instantiate hc_project PIE_GROUP_X=256
