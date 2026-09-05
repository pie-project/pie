//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> o_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
@group(0) @binding(2) var<storage, read> lse: array<f32>;
@group(0) @binding(3) var<storage, read> sinks: array<u32>;

struct Params {
    head_dim: i32,
    heads: i32,
}
@group(0) @binding(4) var<uniform> params: Params;

const PIE_LN2: f32 = 0.69314718055994530942;

fn pie_finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_dim = u32(params.head_dim);
    let d = gid.x * 2u;
    let h = gid.y;
    let t = gid.z;
    if (d >= head_dim) {
        return;
    }
    let th = t * u32(params.heads) + h;
    var r = 1.0;
    let lse_val = lse[th];
    if (pie_finite(lse_val)) {
        let diff = lse_val * PIE_LN2 - pie_bf16_at(sinks[h >> 1u], h);
        r = 1.0 / (1.0 + exp(-diff));
    }
    let w = (th * head_dim + d) >> 1u;
    let word = o_in[w];
    out_[w] = pie_pack_bf16(pie_bf16_to_f32(word & 0xffffu) * r, pie_bf16_to_f32(word >> 16u) * r);
}

// pie:instantiate attn_sink_rescale_bfloat16 PIE_GROUP_X=64
