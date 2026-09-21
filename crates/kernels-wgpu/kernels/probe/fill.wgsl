//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> out_: array<u32>;

struct Params {
    n: i32,
    value: f32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(max(params.n, 0));
    let words = (n + 1u) / 2u;
    let w = gid.x;
    if (w >= words || w >= arrayLength(&out_)) {
        return;
    }

    let lo = w * 2u;
    if (lo + 1u < n) {
        out_[w] = pie_pack_bf16(params.value, params.value);
    } else {
        out_[w] = pie_bf16_into(out_[w], lo, params.value);
    }
}

// pie:instantiate probe_fill_bf16 PIE_GROUP_X=256
