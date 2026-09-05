//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct Params {
    n: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let n = params.n;
    if (w >= (n + 1u) / 2u) {
        return;
    }
    let a = x[w];
    let b = out_[w];
    let lo = pie_bf16_to_f32(a & 0xffffu) + pie_bf16_to_f32(b & 0xffffu);
    let hi = pie_bf16_to_f32(a >> 16u) + pie_bf16_to_f32(b >> 16u);
    if (2u * w + 1u < n) {
        out_[w] = pie_pack_bf16(lo, hi);
    } else {
        out_[w] = pie_bf16_into(out_[w], 0u, lo);
    }
}

// pie:instantiate residual_add_bf16
