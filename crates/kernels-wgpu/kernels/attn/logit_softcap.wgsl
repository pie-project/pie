//#include "common/bf16.inc.wgsl"
//#include "common/math.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct Params {
    cap: f32,
    n: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn softcap(x: f32) -> f32 {
    return params.cap * pie_tanh(x / params.cap);
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let n = u32(max(params.n, 0));

    let w = (group.y * groups.x + group.x) * u32(PIE_GROUP_X) + local.x;
    let lo = w * 2u;
    if (lo >= n) {
        return;
    }
    let word = logits[w];
    let a = softcap(pie_bf16_to_f32(word & 0xffffu));
    if (lo + 1u < n) {
        out_[w] = pie_pack_bf16(a, softcap(pie_bf16_to_f32(word >> 16u)));
    } else {
        out_[w] = (word & 0xffff0000u) | pie_f32_to_bf16(a);
    }
}

// pie:instantiate logit_softcap_bfloat16 PIE_GROUP_X=256
