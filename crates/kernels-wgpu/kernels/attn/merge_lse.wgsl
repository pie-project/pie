//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read> o1: array<u32>;
@group(0) @binding(1) var<storage, read> lse1: array<f32>;
@group(0) @binding(2) var<storage, read> o2: array<u32>;
@group(0) @binding(3) var<storage, read> lse2: array<f32>;
@group(0) @binding(4) var<storage, read_write> o_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> lse_out: array<f32>;
struct Params {
    head_dim: i32,
    heads: i32,
}
@group(0) @binding(6) var<uniform> params: Params;

fn finite_(x: f32) -> bool {
    return abs(x) < 3.0e38;
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x * 2u;
    if (d >= u32(params.head_dim)) {
        return;
    }
    let h = gid.y;
    let t = gid.z;
    let col = t * u32(params.heads) + h;
    let i = (col * u32(params.head_dim) + d) >> 1u;
    let l1 = lse1[col];
    let l2 = lse2[col];
    if (!finite_(l2)) {
        o_out[i] = o1[i];
        if (d == 0u) {
            lse_out[col] = l1;
        }
        return;
    }
    if (!finite_(l1)) {
        o_out[i] = o2[i];
        if (d == 0u) {
            lse_out[col] = l2;
        }
        return;
    }
    let merged_max = max(l1, l2);
    let w1 = exp2(l1 - merged_max);
    let w2 = exp2(l2 - merged_max);
    let total = w1 + w2;
    let a = o1[i];
    let b = o2[i];
    let lo = (pie_bf16_to_f32(a & 0xffffu) * w1 + pie_bf16_to_f32(b & 0xffffu) * w2) / total;
    let hi = (pie_bf16_to_f32(a >> 16u) * w1 + pie_bf16_to_f32(b >> 16u) * w2) / total;
    o_out[i] = pie_pack_bf16(lo, hi);
    if (d == 0u) {
        lse_out[col] = merged_max + log2(total);
    }
}

// pie:instantiate merge_lse_combine_bf16 PIE_GROUP_X=256
