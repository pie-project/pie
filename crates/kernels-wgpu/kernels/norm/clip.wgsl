//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
//#if defined(PIE_LEARNED)
@group(0) @binding(1) var<storage, read> lo_p: array<u32>;
@group(0) @binding(2) var<storage, read> hi_p: array<u32>;
struct Params {
    n: u32,
}
@group(0) @binding(3) var<uniform> params: Params;
//#else
struct Params {
    lo: f32,
    hi: f32,
    n: u32,
}
@group(0) @binding(1) var<uniform> params: Params;
//#endif

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let n = params.n;
    if (w >= (n + 1u) / 2u) {
        return;
    }
//#if defined(PIE_LEARNED)
    let lo = pie_bf16_at(lo_p[0], 0u);
    let hi = pie_bf16_at(hi_p[0], 0u);
//#else
    let lo = pie_bf16_to_f32(pie_f32_to_bf16(params.lo));
    let hi = pie_bf16_to_f32(pie_f32_to_bf16(params.hi));
//#endif
    let word = out_[w];
    let a = min(max(pie_bf16_at(word, 0u), lo), hi);
    let b = min(max(pie_bf16_at(word, 1u), lo), hi);
    if (2u * w + 1u < n) {
        out_[w] = pie_pack_bf16(a, b);
    } else {
        out_[w] = pie_bf16_into(word, 0u, a);
    }
}

// pie:instantiate clamp_bf16 PIE_GROUP_X=256
// pie:instantiate clamp_learned_bf16 PIE_GROUP_X=256 PIE_LEARNED=1
