//#include "common/bf16.inc.wgsl"

const PIE_STRIPS = 4u;
const PIE_LANES = 32u;
const PIE_THREADS = PIE_STRIPS * PIE_LANES;
const PIE_VPT = PIE_HEAD_DIM_MAX / PIE_LANES;
const PIE_PPT = PIE_VPT / 2;

@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read> k: array<u32>;
@group(0) @binding(2) var<storage, read> v: array<u32>;
@group(0) @binding(3) var<storage, read_write> o: array<u32>;
@group(0) @binding(4) var<storage, read> segments: array<i32>;
struct Params {
    num_segments: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    sm_scale: f32,
}
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> pie_dense_q: array<f32, PIE_HEAD_DIM_MAX>;
var<workgroup> pie_dense_fold: array<array<f32, PIE_LANES>, PIE_STRIPS>;
var<workgroup> pie_dense_acc: array<f32, PIE_STRIPS * PIE_HEAD_DIM_MAX>;
var<workgroup> pie_dense_m: array<f32, PIE_STRIPS>;
var<workgroup> pie_dense_l: array<f32, PIE_STRIPS>;
var<workgroup> pie_dense_begin: i32;
var<workgroup> pie_dense_end: i32;

@compute @workgroup_size(PIE_THREADS, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let head = i32(group.x);
    let row = i32(group.y);
    let tid = local.x;
    let strip = tid / PIE_LANES;
    let lane = tid % PIE_LANES;
    let head_dim = u32(params.head_dim);

    if (tid == 0u) {
        var begin = -1;
        var end = -1;
        let first = segments[0];
        let total = segments[u32(params.num_segments)];
        if (row >= first && row < total) {
            var lo = 0;
            var hi = params.num_segments - 1;
            loop {
                if (lo >= hi) {
                    break;
                }
                let mid = (lo + hi + 1) >> 1u;
                if (segments[u32(mid)] <= row) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            begin = segments[u32(lo)];
            end = segments[u32(lo + 1)];
        }
        pie_dense_begin = begin;
        pie_dense_end = end;
    }
    workgroupBarrier();
    let begin = workgroupUniformLoad(&pie_dense_begin);
    let end = workgroupUniformLoad(&pie_dense_end);

    let out_row = (u32(row) * u32(params.num_q_heads) + u32(head)) * head_dim;
    if (end <= begin) {
        for (var d = tid * 2u; d < head_dim; d = d + PIE_THREADS * 2u) {
            o[(out_row + d) >> 1u] = 0u;
        }
        return;
    }
    for (var d = tid * 2u; d < head_dim; d = d + PIE_THREADS * 2u) {
        let word = q[(out_row + d) >> 1u];
        pie_dense_q[d] = pie_bf16_to_f32(word & 0xffffu);
        pie_dense_q[d + 1u] = pie_bf16_to_f32(word >> 16u);
    }
    workgroupBarrier();

    let kv_head = head / (params.num_q_heads / params.num_kv_heads);
    var acc: array<f32, PIE_VPT>;
    for (var u = 0u; u < PIE_VPT; u = u + 1u) {
        acc[u] = 0.0;
    }
    var running_max = -3.0e38;
    var running_sum = 0.0;

    let steps = (end - begin + i32(PIE_STRIPS) - 1) / i32(PIE_STRIPS);
    for (var n = 0; n < steps; n = n + 1) {
        let j = begin + n * i32(PIE_STRIPS) + i32(strip);
        let live = j < end;
        var k_row = 0u;
        var dot = 0.0;
        if (live) {
            k_row = (u32(j) * u32(params.num_kv_heads) + u32(kv_head)) * head_dim;
            for (var u = 0u; u < PIE_PPT; u = u + 1u) {
                let d = (lane + u * PIE_LANES) * 2u;
                if (d < head_dim) {
                    let word = k[(k_row + d) >> 1u];
                    dot = dot + pie_dense_q[d] * pie_bf16_to_f32(word & 0xffffu) + pie_dense_q[d + 1u] * pie_bf16_to_f32(word >> 16u);
                }
            }
        }
        pie_dense_fold[strip][lane] = dot;
        workgroupBarrier();
        var half_ = PIE_LANES >> 1u;
        loop {
            if (half_ == 0u) {
                break;
            }
            if (lane < half_) {
                pie_dense_fold[strip][lane] = pie_dense_fold[strip][lane] + pie_dense_fold[strip][lane + half_];
            }
            workgroupBarrier();
            half_ = half_ >> 1u;
        }
        dot = pie_dense_fold[strip][0];
        workgroupBarrier();
        if (live) {
            let score = dot * params.sm_scale;
            let widened = max(running_max, score);
            let rescale = exp(running_max - widened);
            let weight = exp(score - widened);
            for (var u = 0u; u < PIE_PPT; u = u + 1u) {
                let d = (lane + u * PIE_LANES) * 2u;
                if (d < head_dim) {
                    let word = v[(k_row + d) >> 1u];
                    acc[2u * u] = acc[2u * u] * rescale + weight * pie_bf16_to_f32(word & 0xffffu);
                    acc[2u * u + 1u] = acc[2u * u + 1u] * rescale + weight * pie_bf16_to_f32(word >> 16u);
                }
            }
            running_sum = running_sum * rescale + weight;
            running_max = widened;
        }
    }

    if (lane == 0u) {
        pie_dense_m[strip] = running_max;
        pie_dense_l[strip] = running_sum;
    }
    for (var u = 0u; u < PIE_PPT; u = u + 1u) {
        let d = (lane + u * PIE_LANES) * 2u;
        if (d < head_dim) {
            pie_dense_acc[strip * PIE_HEAD_DIM_MAX + d] = acc[2u * u];
            pie_dense_acc[strip * PIE_HEAD_DIM_MAX + d + 1u] = acc[2u * u + 1u];
        }
    }
    workgroupBarrier();

    var folded_max = -3.0e38;
    for (var w = 0u; w < PIE_STRIPS; w = w + 1u) {
        folded_max = max(folded_max, pie_dense_m[w]);
    }
    var denominator = 0.0;
    for (var w = 0u; w < PIE_STRIPS; w = w + 1u) {
        denominator = denominator + pie_dense_l[w] * exp(pie_dense_m[w] - folded_max);
    }
    var inv = 0.0;
    if (denominator > 0.0) {
        inv = 1.0 / denominator;
    }
    for (var d = tid * 2u; d < head_dim; d = d + PIE_THREADS * 2u) {
        var lo = 0.0;
        var hi = 0.0;
        for (var w = 0u; w < PIE_STRIPS; w = w + 1u) {
            let scale = exp(pie_dense_m[w] - folded_max);
            lo = lo + pie_dense_acc[w * PIE_HEAD_DIM_MAX + d] * scale;
            hi = hi + pie_dense_acc[w * PIE_HEAD_DIM_MAX + d + 1u] * scale;
        }
        o[(out_row + d) >> 1u] = pie_pack_bf16(lo * inv, hi * inv);
    }
}

// pie:instantiate dense_bidirectional_bf16_d_64 PIE_HEAD_DIM_MAX=64
// pie:instantiate dense_bidirectional_bf16_d_128 PIE_HEAD_DIM_MAX=128
// pie:instantiate dense_bidirectional_bf16_d_256 PIE_HEAD_DIM_MAX=256
