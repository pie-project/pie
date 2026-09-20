//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@group(0) @binding(1) var<storage, read> bias: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<u32>;

struct Params {
    width: i32,
    rows: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

fn one(e: u32, v: f32) -> f32 {
    let c = e % u32(max(params.width, 1));
    return (v - pie_bf16_at(bias[c >> 1u], c)) * pie_bf16_at(scale[c >> 1u], c);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u32(max(params.width, 0)) * u32(max(params.rows, 0));
    let w = gid.x;
    let e = 2u * w;
    if (e >= total) {
        return;
    }
    let word = out_[w];
    let lo = one(e, pie_bf16_to_f32(word & 0xffffu));
    if (e + 1u < total) {
        out_[w] = pie_pack_bf16(lo, one(e + 1u, pie_bf16_to_f32(word >> 16u)));
    } else {
        out_[w] = pie_bf16_into(word, 0u, lo);
    }
}

// pie:instantiate standardize_bf16
