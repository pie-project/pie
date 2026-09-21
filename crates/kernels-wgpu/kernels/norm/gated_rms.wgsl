//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> z: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_: array<atomic<u32>>;

struct Params {
    eps: f32,
    vd: i32,
}
@group(0) @binding(4) var<uniform> params: Params;

fn store_out(i: u32, v: f32) {
    let at = i >> 1u;
    let b = pie_f32_to_bf16(v);
    if ((i & 1u) == 1u) {
        atomicAnd(&out_[at], 0x0000ffffu);
        atomicOr(&out_[at], b << 16u);
    } else {
        atomicAnd(&out_[at], 0xffff0000u);
        atomicOr(&out_[at], b);
    }
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let vd = u32(params.vd);
    let base = group.x * vd;
    let lid = local.x;
    var acc = 0.0;
    for (var i = lid; i < vd; i = i + u32(PIE_GROUP_X)) {
        let xi = x[base + i];
        acc = acc + xi * xi;
    }
    let inv = pie_inv_rms(lid, u32(PIE_GROUP_X), acc, vd, params.eps);
    for (var i = lid; i < vd; i = i + u32(PIE_GROUP_X)) {
        let zr = pie_bf16_at(z[(base + i) >> 1u], base + i);
        let e = 1.0 / (1.0 + exp(-abs(zr)));
        let sig = select(e, 1.0 - e, zr < 0.0);
//#if defined(PIE_SIGMOID)
        let gate = sig;
//#else
        let gate = zr * sig;
//#endif
        store_out(base + i, (x[base + i] * inv) * w[i] * gate);
    }
}

// pie:instantiate gated_rms_f32_bf16 PIE_GROUP_X=256
// pie:instantiate gated_rms_sigmoid_f32_bf16 PIE_GROUP_X=256 PIE_SIGMOID=1
