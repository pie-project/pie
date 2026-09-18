//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> positions: array<i32>;
struct Params {
    base_: f32,
    head_dim: i32,
    s0: i32,
    s1: i32,
    s2: i32,
    heads: i32,
    pairs: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn load_at(i: u32) -> f32 {
    return pie_bf16_at(atomicLoad(&x[i >> 1u]), i);
}

fn store_at(i: u32, v: f32) {
    let bits = pie_f32_to_bf16(v);
    let shift = (i & 1u) * 16u;
    atomicAnd(&x[i >> 1u], ~(0xffffu << shift));
    atomicOr(&x[i >> 1u], bits << shift);
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= params.pairs) {
        return;
    }
    let h = gid.y;
    let m = gid.z;
    let half_hd = params.head_dim / 2;
    let pos_t = positions[3u * m + 0u];
    let pos_h = positions[3u * m + 1u];
    let pos_w = positions[3u * m + 2u];
    let row = (m * u32(params.heads) + h) * u32(params.head_dim);
    var inv_freq = 0.0;
    var axis_pos = 0;
    var i1 = 0u;
    var i2 = 0u;
//#if defined(PIE_BLOCKED)
    let total = params.s0 + params.s1 + params.s2;
    var within = 0;
    if (i < params.s0) {
        axis_pos = pos_t;
        within = i;
    } else if (i < params.s0 + params.s1) {
        axis_pos = pos_h;
        within = i - params.s0;
    } else {
        axis_pos = pos_w;
        within = i - params.s0 - params.s1;
    }
    inv_freq = exp2(-(2.0 * f32(within) / f32(total)) * params.base_);
    i1 = row + u32(i);
    i2 = i1 + u32(half_hd);
//#elif defined(PIE_SPLIT)
    var within = 0;
    var before = 0;
    var width = 0;
    if (i < params.s0) {
        axis_pos = pos_t;
        within = i;
        before = 0;
        width = params.s0;
    } else if (i < params.s0 + params.s1) {
        axis_pos = pos_h;
        within = i - params.s0;
        before = params.s0;
        width = params.s1;
    } else {
        axis_pos = pos_w;
        within = i - params.s0 - params.s1;
        before = params.s0 + params.s1;
        width = params.s2;
    }
    if (width <= 0) {
        return;
    }
    inv_freq = exp2(-(f32(within) / f32(width)) * params.base_);
    i1 = row + u32(2 * before + within);
    i2 = i1 + u32(width);
//#else
    let r = i % 3;
    if (r == 1 && i < 3 * params.s1) {
        axis_pos = pos_h;
    } else if (r == 2 && i < 3 * params.s2) {
        axis_pos = pos_w;
    } else {
        axis_pos = pos_t;
    }
    inv_freq = exp2(-(2.0 * f32(i) / f32(params.head_dim)) * params.base_);
    i1 = row + u32(i);
    i2 = i1 + u32(half_hd);
//#endif
    let theta = f32(axis_pos) * inv_freq;
    let c = cos(theta);
    let s = sin(theta);
    let x1 = load_at(i1);
    let x2 = load_at(i2);
    store_at(i1, x1 * c - x2 * s);
    store_at(i2, x1 * s + x2 * c);
}

// pie:instantiate rope_mrope_interleaved_bf16 PIE_GROUP_X=64
// pie:instantiate rope_mrope_blocked_bf16 PIE_BLOCKED=1 PIE_GROUP_X=64
// pie:instantiate rope_mrope_split_bf16 PIE_SPLIT=1 PIE_GROUP_X=64
