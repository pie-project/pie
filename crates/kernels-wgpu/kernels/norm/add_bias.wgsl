//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@group(0) @binding(1) var<storage, read> bias: array<u32>;

struct Params {
    width: i32,
    n: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let n = u32(max(params.n, 0));
    let width = u32(max(params.width, 1));
    if (w >= (n + 1u) / 2u) {
        return;
    }
    let e0 = 2u * w;
    let c0 = e0 % width;
    let c1 = (e0 + 1u) % width;
    let a = out_[w];
    let lo = pie_bf16_to_f32(a & 0xffffu) + pie_bf16_at(bias[c0 >> 1u], c0);
    let hi = pie_bf16_to_f32(a >> 16u) + pie_bf16_at(bias[c1 >> 1u], c1);
    if (e0 + 1u < n) {
        out_[w] = pie_pack_bf16(lo, hi);
    } else {
        out_[w] = pie_bf16_into(a, 0u, lo);
    }
}

// pie:instantiate add_bias_bf16
