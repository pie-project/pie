//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct Params {
    eps: f32,
    axis_size: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let axis_size = u32(max(params.axis_size, 0));
    let base = group.x * axis_size;
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
                let lo = pie_bf16_to_f32(word & 0xffffu) * inv;
                var hi = 0.0;
                if (at + 1u < axis_size) {
                    hi = pie_bf16_to_f32(word >> 16u) * inv;
                }
                out_[e >> 1u] = pie_pack_bf16(lo, hi);
            }
        }
    }
}

// pie:instantiate vnorm_single_row_bf16 PIE_GROUP_X=256 N_READS=4
