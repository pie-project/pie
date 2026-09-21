//#include "common/bf16.inc.wgsl"

const HC_MAX_MULT = 8u;

@group(0) @binding(0) var<storage, read> o: array<u32>;
@group(0) @binding(1) var<storage, read> gates: array<u32>;
@group(0) @binding(2) var<storage, read_write> hyper: array<u32>;

struct Params {
    n_rows: i32,
    m: i32,
    h: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = u32(max(params.m, 0));
    let H = u32(max(params.h, 0));
    let h = 2u * gid.x;
    let n = gid.y;
    if (M > HC_MAX_MULT || h >= H || n >= u32(max(params.n_rows, 0))) {
        return;
    }
    let ow = o[(n * H + h) >> 1u];
    let o_lo = pie_bf16_to_f32(ow & 0xffffu);
    let o_hi = pie_bf16_to_f32(ow >> 16u);
    let base = n * M * H + h;
    for (var s = 0u; s < M; s = s + 1u) {
        let logit = pie_bf16_at(gates[(n * M + s) >> 1u], n * M + s) / f32(M);
        let g = 2.0 / (1.0 + exp(-logit));
        let at = (base + s * H) >> 1u;
        let word = hyper[at];
        hyper[at] = pie_pack_bf16(
            pie_bf16_to_f32(word & 0xffffu) + g * o_lo,
            pie_bf16_to_f32(word >> 16u) + g * o_hi,
        );
    }
}

// pie:instantiate hc_inject_bf16
