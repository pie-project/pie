//#include "common/bf16.inc.wgsl"

//#if defined(PIE_GELU)
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
struct Params {
    n: u32,
}
@group(0) @binding(2) var<uniform> params: Params;
//#elif defined(PIE_SWIGLU_CLAMP)
@group(0) @binding(0) var<storage, read> gate: array<u32>;
@group(0) @binding(1) var<storage, read> up: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
struct Params {
    limit: f32,
    n: u32,
}
@group(0) @binding(3) var<uniform> params: Params;
//#else
@group(0) @binding(0) var<storage, read> gate: array<u32>;
@group(0) @binding(1) var<storage, read> up: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
struct Params {
    n: u32,
}
@group(0) @binding(3) var<uniform> params: Params;
//#endif

fn gelu_tanh(v: f32) -> f32 {
    let k = 0.7978845608028654;
    let inner = k * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + tanh(inner));
}

fn one(g: f32, u: f32) -> f32 {
//#if defined(PIE_GELU)
    return gelu_tanh(g);
//#elif defined(PIE_SWIGLU_CLAMP)
    let gc = min(g, params.limit);
    let uc = clamp(u, -params.limit, params.limit);
    return (gc / (1.0 + exp(-gc))) * uc;
//#else
    return gelu_tanh(g) * u;
//#endif
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let wi = gid.x;
    let n = params.n;
    if (wi >= (n + 1u) / 2u) {
        return;
    }
//#if defined(PIE_GELU)
    let g = x[wi];
    let u = 0u;
//#else
    let g = gate[wi];
    let u = up[wi];
//#endif
    let lo = one(pie_bf16_to_f32(g & 0xffffu), pie_bf16_to_f32(u & 0xffffu));
    let hi = one(pie_bf16_to_f32(g >> 16u), pie_bf16_to_f32(u >> 16u));
    if (2u * wi + 1u < n) {
        out_[wi] = pie_pack_bf16(lo, hi);
    } else {
        out_[wi] = pie_bf16_into(out_[wi], 0u, lo);
    }
}

// pie:instantiate geglu_tanh_bf16
// pie:instantiate swiglu_clamp_bf16 PIE_SWIGLU_CLAMP=1
// pie:instantiate mlp_gelu_tanh_bf16 PIE_GELU=1
