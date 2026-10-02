//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read> position: array<i32>;

struct Params {
    base: f32,
    head_dim: i32,
    factor: f32,
    low_dim: f32,
    high_dim: f32,
    mscale: f32,
    interleaved: i32,
    pair_half: i32,
    heads: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn theta_at(i: u32, head_dim: u32, pos: f32) -> f32 {
    let d = 2.0 * f32(i) / f32(head_dim);
    let base_freq = exp2(-d * params.base);
    var denom = params.high_dim - params.low_dim;
    if (params.high_dim == params.low_dim) {
        denom = 1e-3;
    }
    let ramp = clamp((f32(i) - params.low_dim) / denom, 0.0, 1.0);
    let freq = base_freq * ((1.0 - ramp) + ramp / params.factor);
    return pos * freq;
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_half = u32(max(params.pair_half, 0));
    let i = 2u * gid.x;
    if (i >= pair_half) {
        return;
    }
    let h = gid.y;
    let row = gid.z;
    let head_dim = u32(max(params.head_dim, 1));
    let pos = f32(position[row]);
    let row_base = (row * u32(max(params.heads, 0)) + h) * head_dim;
    let t0 = theta_at(i, head_dim, pos);
    let t1 = theta_at(i + 1u, head_dim, pos);
    let c0 = cos(t0) * params.mscale;
    let s0 = sin(t0) * params.mscale;
    let c1 = cos(t1) * params.mscale;
    let s1 = sin(t1) * params.mscale;
    if (params.interleaved != 0) {
        let w0 = row_base + 2u * i;
        let a = x[w0 >> 1u];
        let b = x[(w0 + 2u) >> 1u];
        let a0 = pie_bf16_to_f32(a & 0xffffu);
        let a1 = pie_bf16_to_f32(a >> 16u);
        let b0 = pie_bf16_to_f32(b & 0xffffu);
        let b1 = pie_bf16_to_f32(b >> 16u);
        x[w0 >> 1u] = pie_pack_bf16(a0 * c0 - a1 * s0, a0 * s0 + a1 * c0);
        x[(w0 + 2u) >> 1u] = pie_pack_bf16(b0 * c1 - b1 * s1, b0 * s1 + b1 * c1);
    } else {
        let i1 = row_base + i;
        let i2 = i1 + pair_half;
        let a = x[i1 >> 1u];
        let b = x[i2 >> 1u];
        let x1_0 = pie_bf16_to_f32(a & 0xffffu);
        let x1_1 = pie_bf16_to_f32(a >> 16u);
        let x2_0 = pie_bf16_to_f32(b & 0xffffu);
        let x2_1 = pie_bf16_to_f32(b >> 16u);
        x[i1 >> 1u] = pie_pack_bf16(x1_0 * c0 - x2_0 * s0, x1_1 * c1 - x2_1 * s1);
        x[i2 >> 1u] = pie_pack_bf16(x1_0 * s0 + x2_0 * c0, x1_1 * s1 + x2_1 * c1);
    }
}

// pie:instantiate neox_yarn_mb_bf16 PIE_GROUP_X=64
