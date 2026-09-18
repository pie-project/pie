//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
//#if defined(PIE_STATED) || defined(PIE_SILU)
struct Params {
    scalar: f32,
    n: u32,
}
@group(0) @binding(1) var<uniform> params: Params;
//#else
@group(0) @binding(1) var<storage, read> scalar: array<u32>;
struct Params {
    n: u32,
}
@group(0) @binding(2) var<uniform> params: Params;
//#endif

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let n = params.n;
    if (w >= (n + 1u) / 2u) {
        return;
    }
    let a = out_[w];
//#if defined(PIE_SILU)
    let v0 = pie_bf16_to_f32(a & 0xffffu) * params.scalar;
    let v1 = pie_bf16_to_f32(a >> 16u) * params.scalar;
    let lo = v0 / (1.0 + exp(-v0));
    let hi = v1 / (1.0 + exp(-v1));
//#else
//#if defined(PIE_STATED)
    let s = pie_bf16_to_f32(pie_f32_to_bf16(params.scalar));
//#else
    let s = pie_bf16_to_f32(scalar[0] & 0xffffu);
//#endif
    let lo = pie_bf16_to_f32(a & 0xffffu) * s;
    let hi = pie_bf16_to_f32(a >> 16u) * s;
//#endif
    if (2u * w + 1u < n) {
        out_[w] = pie_pack_bf16(lo, hi);
    } else {
        out_[w] = pie_bf16_into(out_[w], 0u, lo);
    }
}

// pie:instantiate layer_scalar_mul_bf16
// pie:instantiate layer_scalar_mul_stated_bf16 PIE_STATED=1
// pie:instantiate silu_scaled_bf16 PIE_SILU=1
