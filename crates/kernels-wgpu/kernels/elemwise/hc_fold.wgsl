//#include "common/bf16.inc.wgsl"

const HC_MAX_MULT = 8u;

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> residual: array<u32>;
@group(0) @binding(2) var<storage, read> post_mix: array<f32>;
@group(0) @binding(3) var<storage, read> comb_mix: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_: array<u32>;

struct Params {
    n_rows: i32,
    m: i32,
    h: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = u32(max(params.m, 0));
    let H = u32(max(params.h, 0));
    let h = 2u * gid.x;
    let n = gid.y;
    if (M > HC_MAX_MULT || h >= H || n >= u32(max(params.n_rows, 0))) {
        return;
    }
    let comb = n * M * M;
    let post = n * M;
    let xw = x[(n * H + h) >> 1u];
    let x_lo = pie_bf16_to_f32(xw & 0xffffu);
    let x_hi = pie_bf16_to_f32(xw >> 16u);
    let res = n * M * H + h;
    var r_lo: array<f32, HC_MAX_MULT>;
    var r_hi: array<f32, HC_MAX_MULT>;
    for (var i = 0u; i < M; i = i + 1u) {
        let word = residual[(res + i * H) >> 1u];
        r_lo[i] = pie_bf16_to_f32(word & 0xffffu);
        r_hi[i] = pie_bf16_to_f32(word >> 16u);
    }
    for (var j = 0u; j < M; j = j + 1u) {
        var lo = post_mix[post + j] * x_lo;
        var hi = post_mix[post + j] * x_hi;
        for (var i = 0u; i < M; i = i + 1u) {
            let c = comb_mix[comb + i * M + j];
            lo = lo + c * r_lo[i];
            hi = hi + c * r_hi[i];
        }
        out_[(res + j * H) >> 1u] = pie_pack_bf16(lo, hi);
    }
}

// pie:instantiate hc_fold_bf16
