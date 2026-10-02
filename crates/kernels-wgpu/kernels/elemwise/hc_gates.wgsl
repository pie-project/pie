//#include "common/bf16.inc.wgsl"

const HC_MAX_MULT = 8u;

@group(0) @binding(0) var<storage, read> mixes: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<u32>;
@group(0) @binding(4) var<storage, read_write> post_mix: array<f32>;
@group(0) @binding(5) var<storage, read_write> comb_mix: array<f32>;
@group(0) @binding(6) var<storage, read_write> layer_input: array<u32>;

struct Params {
    m: i32,
    h: i32,
    hc_eps: f32,
    alpha: f32,
    sinkhorn: i32,
}
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> pre: array<f32, HC_MAX_MULT>;
var<workgroup> post: array<f32, HC_MAX_MULT>;
var<workgroup> comb: array<f32, HC_MAX_MULT * HC_MAX_MULT>;

fn sigmoid(v: f32) -> f32 {
    return 1.0 / (1.0 + exp(-v));
}

fn load_res(i: u32) -> f32 {
    return pie_bf16_at(residual[i >> 1u], i);
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let n = group.x;
    let tid = local.x;
    let M = u32(max(params.m, 0));
    let H = u32(max(params.h, 0));
    let mix_hc = 2u * M + M * M;
    let row = n * mix_hc;

    if (tid < M) {
        pre[tid] = sigmoid(mixes[row + tid] * scale[0] + base[tid]) + params.hc_eps;
        let p = sigmoid(mixes[row + M + tid] * scale[1] + base[M + tid]) * params.alpha;
        post[tid] = p;
        post_mix[n * M + tid] = p;
    }
    if (tid < M * M) {
        comb[tid] = mixes[row + 2u * M + tid] * scale[2] + base[2u * M + tid];
    }
    workgroupBarrier();

    if (tid < M) {
        var max_v = -1e30;
        for (var j = 0u; j < M; j = j + 1u) {
            max_v = max(max_v, comb[tid * M + j]);
        }
        var sum = 0.0;
        for (var j = 0u; j < M; j = j + 1u) {
            let e = exp(comb[tid * M + j] - max_v);
            comb[tid * M + j] = e;
            sum = sum + e;
        }
        for (var j = 0u; j < M; j = j + 1u) {
            comb[tid * M + j] = comb[tid * M + j] / sum + params.hc_eps;
        }
    }
    workgroupBarrier();

    if (tid < M) {
        var col = 0.0;
        for (var i = 0u; i < M; i = i + 1u) {
            col = col + comb[i * M + tid];
        }
        col = col + params.hc_eps;
        for (var i = 0u; i < M; i = i + 1u) {
            comb[i * M + tid] = comb[i * M + tid] / col;
        }
    }
    workgroupBarrier();

    for (var iter = 0; iter < params.sinkhorn - 1; iter = iter + 1) {
        if (tid < M) {
            var rs = 0.0;
            for (var j = 0u; j < M; j = j + 1u) {
                rs = rs + comb[tid * M + j];
            }
            rs = rs + params.hc_eps;
            for (var j = 0u; j < M; j = j + 1u) {
                comb[tid * M + j] = comb[tid * M + j] / rs;
            }
        }
        workgroupBarrier();
        if (tid < M) {
            var col = 0.0;
            for (var i = 0u; i < M; i = i + 1u) {
                col = col + comb[i * M + tid];
            }
            col = col + params.hc_eps;
            for (var i = 0u; i < M; i = i + 1u) {
                comb[i * M + tid] = comb[i * M + tid] / col;
            }
        }
        workgroupBarrier();
    }

    if (tid < M * M) {
        comb_mix[n * M * M + tid] = comb[tid];
    }
    workgroupBarrier();

    let res = n * M * H;
    let out = n * H;
    for (var h = 2u * tid; h < H; h = h + 2u * u32(PIE_GROUP_X)) {
        var lo = 0.0;
        var hi = 0.0;
        for (var i = 0u; i < M; i = i + 1u) {
            let word = residual[(res + i * H + h) >> 1u];
            lo = lo + pre[i] * pie_bf16_to_f32(word & 0xffffu);
            hi = hi + pre[i] * pie_bf16_to_f32(word >> 16u);
        }
        layer_input[(out + h) >> 1u] = pie_pack_bf16(lo, hi);
    }
}

// pie:instantiate hc_gates_bf16 PIE_GROUP_X=256
