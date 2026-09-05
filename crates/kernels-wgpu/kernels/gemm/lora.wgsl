//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

const PIE_MAX_RANK = 128u;

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> bank_a: array<u32>;
@group(0) @binding(2) var<storage, read> bank_b: array<u32>;
@group(0) @binding(3) var<storage, read> routes: array<i32>;
@group(0) @binding(4) var<storage, read_write> y: array<u32>;

struct Params {
    in_width: i32,
    out_width: i32,
    rank: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> waist: array<f32, PIE_MAX_RANK>;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.y;
    let adapter = routes[row];

    if (adapter < 0) {
        return;
    }
    let lid = local.x;
    let in_width = u32(max(params.in_width, 0));
    let out_width = u32(max(params.out_width, 0));
    let r = min(u32(max(params.rank, 0)), PIE_MAX_RANK);
    let down = u32(adapter) * r * in_width;
    let up = u32(adapter) * out_width * r;
    let a = row * in_width;
    let out = row * out_width;

    for (var i = 0u; i < r; i = i + 1u) {
        let wrow = down + i * in_width;
        var acc = 0.0;
        for (var c = lid; c < in_width; c = c + u32(PIE_GROUP_X)) {
            acc = acc + pie_bf16_at(bank_a[(wrow + c) >> 1u], wrow + c) * pie_bf16_at(x[(a + c) >> 1u], a + c);
        }
        let total = pie_workgroup_sum(lid, u32(PIE_GROUP_X), acc);
        if (lid == 0u) {
            waist[i] = total;
        }
    }
    workgroupBarrier();

    for (var n = 2u * lid; n < out_width; n = n + 2u * u32(PIE_GROUP_X)) {
        let brow0 = up + n * r;
        let brow1 = brow0 + r;
        var acc0 = 0.0;
        var acc1 = 0.0;
        for (var i = 0u; i < r; i = i + 1u) {
            acc0 = acc0 + pie_bf16_at(bank_b[(brow0 + i) >> 1u], brow0 + i) * waist[i];
            acc1 = acc1 + pie_bf16_at(bank_b[(brow1 + i) >> 1u], brow1 + i) * waist[i];
        }
        let at = (out + n) >> 1u;
        let old = y[at];
        y[at] = pie_pack_bf16(
            pie_bf16_to_f32(old & 0xffffu) + acc0,
            pie_bf16_to_f32(old >> 16u) + acc1,
        );
    }
}

// pie:instantiate lora_correct_bf16 PIE_GROUP_X=256
