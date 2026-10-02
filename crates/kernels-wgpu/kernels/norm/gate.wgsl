//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read> gate: array<u32>;

struct Params {
    width: i32,
    rows: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn gated(word: u32, i: u32) -> f32 {
    let g = pie_bf16_at(gate[i >> 1u], i);
    let e = 1.0 / (1.0 + exp(-abs(g)));
    let sig = select(e, 1.0 - e, g < 0.0);
    return pie_bf16_at(word, i) * sig;
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(params.width) * u32(params.rows);
    let lo = gid.x * 2u;
    if (lo >= n) {
        return;
    }
    let word = x[gid.x];
    let v0 = gated(word, lo);
    if (lo + 1u < n) {
        x[gid.x] = pie_pack_bf16(v0, gated(word, lo + 1u));
    } else {
        x[gid.x] = pie_bf16_into(word, lo, v0);
    }
}

// pie:instantiate gate_sigmoid_mul_bf16 PIE_GROUP_X=256
