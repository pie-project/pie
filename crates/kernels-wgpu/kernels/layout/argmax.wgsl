//#include "common/bf16.inc.wgsl"

//#if defined(PIE_F32)
@group(0) @binding(0) var<storage, read> x: array<f32>;
//#else
@group(0) @binding(0) var<storage, read> x: array<u32>;
//#endif
@group(0) @binding(1) var<storage, read_write> y: array<i32>;

struct Params {
    width: i32,
    depth: i32,
    column: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sh_v: array<f32, PIE_GROUP_X>;
var<workgroup> sh_i: array<u32, PIE_GROUP_X>;

fn load_x(i: u32) -> f32 {
//#if defined(PIE_F32)
    return x[i];
//#else
    return pie_bf16_at(x[i >> 1u], i);
//#endif
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.y;
    let lid = local.x;
    let width = u32(max(params.width, 0));
    var best = -3.402823466e38;
    var best_i = 0xffffffffu;
    for (var c = lid; c < width; c = c + u32(PIE_GROUP_X)) {
        let v = load_x(row * width + c);

        if (v == v && (best_i == 0xffffffffu || v > best || (v == best && c < best_i))) {
            best = v;
            best_i = c;
        }
    }
    sh_v[lid] = best;
    sh_i[lid] = best_i;
    workgroupBarrier();
    for (var stride = u32(PIE_GROUP_X) >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let ov = sh_v[lid + stride];
            let oi = sh_i[lid + stride];
            if (oi != 0xffffffffu && (sh_i[lid] == 0xffffffffu || ov > sh_v[lid] || (ov == sh_v[lid] && oi < sh_i[lid]))) {
                sh_v[lid] = ov;
                sh_i[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        var top = sh_i[0];
        if (top == 0xffffffffu) {
            top = 0u;
        }
        y[row * u32(max(params.depth, 1)) + u32(max(params.column, 0))] = i32(top);
    }
}

// pie:instantiate argmax_rows_bf16 PIE_GROUP_X=256
// pie:instantiate argmax_rows_f32 PIE_GROUP_X=256 PIE_F32=1
