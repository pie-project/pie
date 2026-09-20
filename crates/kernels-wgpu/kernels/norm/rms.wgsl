//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

struct Params {
    eps: f32,
    axis_size: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
//#if defined(PIE_GROUPED)

    groups: u32,
//#endif
}
@group(0) @binding(3) var<uniform> params: Params;

var<private> pie_plane: u32 = 0u;

fn gain_at(i: u32) -> f32 {
    let j = pie_plane + params.w_stride * i;
    let wv = pie_bf16_at(w[j >> 1u], j);
    return params.gain * select(wv, 1.0 + wv, params.plus_one != 0u);
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let axis_size = u32(max(params.axis_size, 0));
    let base = group.x * axis_size;
//#if defined(PIE_GROUPED)
    pie_plane = (group.x % max(params.groups, 1u)) * axis_size * params.w_stride;
//#endif
    let span = u32(PIE_GROUP_X) * u32(N_READS);
    let lid = local.x;

    var acc = 0.0;
    for (var start = lid * u32(N_READS); start < axis_size; start = start + span) {
        for (var i = 0u; i < u32(N_READS); i = i + 1u) {
            let at = start + i;
            if (at < axis_size) {
                let xi = pie_bf16_at(x[(base + at) >> 1u], base + at);
                acc = acc + xi * xi;
            }
        }
    }
    let inv = pie_inv_rms(lid, u32(PIE_GROUP_X), acc, axis_size, params.eps);

    for (var start = lid * u32(N_READS); start < axis_size; start = start + span) {
        for (var i = 0u; i < u32(N_READS); i = i + 2u) {
            let at = start + i;
            if (at < axis_size) {
                let e = base + at;
                let word = x[e >> 1u];
                let lo = gain_at(at) * (pie_bf16_to_f32(word & 0xffffu) * inv);
                var hi = 0.0;
                if (at + 1u < axis_size) {
                    hi = gain_at(at + 1u) * (pie_bf16_to_f32(word >> 16u) * inv);
                }
                out_[e >> 1u] = pie_pack_bf16(lo, hi);
            }
        }
    }
}

// pie:instantiate rms_single_row_bf16 PIE_GROUP_X=256 N_READS=4
// pie:instantiate rms_grouped_row_bf16 PIE_GROUP_X=256 N_READS=4 PIE_GROUPED=1
