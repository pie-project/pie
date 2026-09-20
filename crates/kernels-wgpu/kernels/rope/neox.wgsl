//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read> position: array<i32>;

//#if defined(PIE_LAST)
struct Params {
    base: f32,
    head_dim: i32,
    interleaved: i32,
    sign: f32,
    factor: f32,
    low: f32,
    high: f32,
    pair_half: i32,
    heads: i32,
}
//#else
struct Params {
    scale: f32,
    base: f32,
    head_dim: i32,
    pair_half: i32,
    heads: i32,
}
//#endif
@group(0) @binding(2) var<uniform> params: Params;

fn rotate_word(i1: u32, i2: u32, theta0: f32, theta1: f32) {
    let a = x[i1 >> 1u];
    let b = x[i2 >> 1u];
    let x1_0 = pie_bf16_to_f32(a & 0xffffu);
    let x1_1 = pie_bf16_to_f32(a >> 16u);
    let x2_0 = pie_bf16_to_f32(b & 0xffffu);
    let x2_1 = pie_bf16_to_f32(b >> 16u);
    let c0 = cos(theta0);
    let s0 = sin(theta0);
    let c1 = cos(theta1);
    let s1 = sin(theta1);
    x[i1 >> 1u] = pie_pack_bf16(x1_0 * c0 - x2_0 * s0, x1_1 * c1 - x2_1 * s1);
    x[i2 >> 1u] = pie_pack_bf16(x1_0 * s0 + x2_0 * c0, x1_1 * s1 + x2_1 * c1);
}

//#if defined(PIE_LAST)

fn tail_inv_freq(i: u32, rotary: u32) -> f32 {
    let d = 2.0 * f32(i) / f32(max(rotary, 1u));
    var inv_freq = exp2(-d * params.base);
    if (params.factor > 1.0) {
        let span = max(params.high - params.low, 0.001);
        let ramp = clamp((f32(i) - params.low) / span, 0.0, 1.0);
        let lag = 1.0 - ramp;
        inv_freq = inv_freq / params.factor * (1.0 - lag) + inv_freq * lag;
    }
    return inv_freq;
}
//#endif

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
    let row_base = row * u32(max(params.heads, 0)) * head_dim;
//#if defined(PIE_LAST)
    let rotary = 2u * pair_half;
    let head_base = row_base + h * head_dim + (head_dim - rotary);
    let pos = params.sign * f32(position[row]);
    let theta0 = pos * tail_inv_freq(i, rotary);
    let theta1 = pos * tail_inv_freq(i + 1u, rotary);
    if (params.interleaved != 0) {
        let j0 = (head_base + 2u * i) >> 1u;
        let a = x[j0];
        let b = x[j0 + 1u];
        let c0 = cos(theta0);
        let s0 = sin(theta0);
        let c1 = cos(theta1);
        let s1 = sin(theta1);
        let a1 = pie_bf16_to_f32(a & 0xffffu);
        let a2 = pie_bf16_to_f32(a >> 16u);
        let b1 = pie_bf16_to_f32(b & 0xffffu);
        let b2 = pie_bf16_to_f32(b >> 16u);
        x[j0] = pie_pack_bf16(a1 * c0 - a2 * s0, a1 * s0 + a2 * c0);
        x[j0 + 1u] = pie_pack_bf16(b1 * c1 - b2 * s1, b1 * s1 + b2 * c1);
    } else {
        let i1 = head_base + i;
        rotate_word(i1, i1 + pair_half, theta0, theta1);
    }
//#else
//#if defined(PIE_PROP)
    let d0 = 2.0 * f32(i) / f32(head_dim);
    let d1 = 2.0 * f32(i + 1u) / f32(head_dim);
    let i2_off = head_dim / 2u;
//#else
    let d0 = f32(i) / f32(pair_half);
    let d1 = f32(i + 1u) / f32(pair_half);
    let i2_off = pair_half;
//#endif
    let pos = params.scale * f32(position[row]);
    let theta0 = pos * exp2(-d0 * params.base);
    let theta1 = pos * exp2(-d1 * params.base);
    let i1 = row_base + h * head_dim + i;
    rotate_word(i1, i1 + i2_off, theta0, theta1);
//#endif
}

// pie:instantiate neox_mb_bf16 PIE_GROUP_X=64
// pie:instantiate neox_prop_mb_bf16 PIE_GROUP_X=64 PIE_PROP=1
// pie:instantiate neox_last_mb_bf16 PIE_GROUP_X=64 PIE_LAST=1
