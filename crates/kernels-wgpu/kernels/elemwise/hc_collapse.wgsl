//#include "common/bf16.inc.wgsl"

const HC_MAX_MULT = 8u;

@group(0) @binding(0) var<storage, read> mixes: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_: array<u32>;

struct Params {
    m: i32,
    h: i32,
    hc_eps: f32,
}
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> gates: array<f32, HC_MAX_MULT>;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let n = group.x;
    let tid = local.x;
    let M = u32(max(params.m, 0));
    let H = u32(max(params.h, 0));
    if (tid < M) {
        let logit = mixes[n * M + tid] * scale[0] + base[tid];
        gates[tid] = 1.0 / (1.0 + exp(-logit)) + params.hc_eps;
    }
    workgroupBarrier();
    let res = n * M * H;
    for (var h = 2u * tid; h < H; h = h + 2u * u32(PIE_GROUP_X)) {
        var lo = 0.0;
        var hi = 0.0;
        for (var i = 0u; i < M; i = i + 1u) {
            let word = residual[(res + i * H + h) >> 1u];
            lo = lo + gates[i] * pie_bf16_to_f32(word & 0xffffu);
            hi = hi + gates[i] * pie_bf16_to_f32(word >> 16u);
        }
        out_[(n * H + h) >> 1u] = pie_pack_bf16(lo, hi);
    }
}

// pie:instantiate hc_collapse_bf16 PIE_GROUP_X=256
