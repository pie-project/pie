//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read> ba: array<u32>;
@group(0) @binding(1) var<storage, read> a_log: array<f32>;
@group(0) @binding(2) var<storage, read> dt_bias: array<u32>;
@group(0) @binding(3) var<storage, read_write> gates: array<f32>;

struct Params {
    v_heads: i32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    let t = gid.y;
    let vh = u32(params.v_heads);
    if (h >= vh) {
        return;
    }
    let row = t * 2u * vh;
    let bv = pie_bf16_at(ba[(row + h) >> 1u], row + h);
    let av = pie_bf16_at(ba[(row + vh + h) >> 1u], row + vh + h);
    let z = av + pie_bf16_at(dt_bias[h >> 1u], h);
    var sp = z;
    if (z <= 20.0) {
        sp = log(1.0 + exp(z));
    }
    gates[row + h] = -exp(a_log[h]) * sp;
    gates[row + vh + h] = 1.0 / (1.0 + exp(-bv));
}

// pie:instantiate gdn_ba_gates_bf16 PIE_GROUP_X=256
