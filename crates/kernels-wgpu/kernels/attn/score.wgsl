//#include "common/bf16.inc.wgsl"

const PIE_STRIPS = 8u;
const PIE_LANES = 32u;
const PIE_THREADS = PIE_STRIPS * PIE_LANES;
const PIE_VPT = PIE_HEAD_DIM_MAX / PIE_LANES;

const PIE_NEG_INF = -3.0e38;

@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read> qo_indptr: array<i32>;
@group(0) @binding(2) var<storage, read> k_pages: array<u32>;
@group(0) @binding(3) var<storage, read> kv_page_indices: array<u32>;
@group(0) @binding(4) var<storage, read> kv_page_indptr: array<u32>;
@group(0) @binding(5) var<storage, read> position_ids: array<i32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;

struct Params {
    page_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    sm_scale: f32,
    observe: i32,
    lane_offset: i32,
    plane_stride: i32,
    plane: i32,
    kv_max: i32,
}
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> pie_score_q: array<f32, PIE_HEAD_DIM_MAX>;
var<workgroup> pie_score_fold: array<array<f32, PIE_LANES>, PIE_STRIPS>;
var<workgroup> pie_score_m: array<f32, PIE_STRIPS>;
var<workgroup> pie_score_l: array<f32, PIE_STRIPS>;
var<workgroup> pie_score_rows: i32;
var<workgroup> pie_score_qohi: i32;
var<workgroup> pie_score_cap: i32;
var<workgroup> pie_score_first: i32;
var<workgroup> pie_score_limit: i32;
var<workgroup> pie_score_steps: i32;

fn pie_score_strip_sum(strip: u32, lane: u32, v: f32) -> f32 {
    pie_score_fold[strip][lane] = v;
    workgroupBarrier();
    var half_ = PIE_LANES >> 1u;
    loop {
        if (half_ == 0u) { break; }
        if (lane < half_) {
            pie_score_fold[strip][lane] = pie_score_fold[strip][lane]
                + pie_score_fold[strip][lane + half_];
        }
        workgroupBarrier();
        half_ = half_ >> 1u;
    }
    let total = pie_score_fold[strip][0];
    workgroupBarrier();
    return total;
}

fn pie_score_dot(j: i32, kv_head: u32, lane: u32, page_first: i32, head_dim: u32) -> f32 {
    let page = kv_page_indices[u32(page_first + j / params.page_size)];
    let slot = page * u32(params.page_size) + u32(j % params.page_size);
    let row_stride = u32(params.num_kv_heads) * head_dim;
    let k_row = slot * row_stride + kv_head * head_dim;
    var dot = 0.0;
    for (var u = 0u; u < PIE_VPT; u = u + 1u) {
        let d = lane + u * PIE_LANES;
        if (d < head_dim) {
            let e = k_row + d;
            dot = dot + pie_score_q[d] * pie_bf16_at(k_pages[e >> 1u], e);
        }
    }
    return dot;
}

@compute @workgroup_size(PIE_THREADS, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let request = i32(group.x);
    let head = i32(group.y);
    let tid = local.x;
    let strip = tid / PIE_LANES;
    let lane = tid % PIE_LANES;
    let head_dim = u32(params.head_dim);

    let out_base = (u32(params.lane_offset + request) * u32(params.plane_stride)
        + u32(params.plane + head)) * u32(params.kv_max);
    for (var i = tid; i < u32(params.kv_max); i = i + PIE_THREADS) {
        scores[out_base + i] = 0.0;
    }

    if (tid == 0u) {
        let page_first = i32(kv_page_indptr[u32(request)]);
        let pages = i32(kv_page_indptr[u32(request) + 1u]) - page_first;
        let qo_hi = qo_indptr[u32(request) + 1u];
        let qo_len = qo_hi - qo_indptr[u32(request)];
        pie_score_first = page_first;

        pie_score_cap = pages * params.page_size;
        pie_score_qohi = qo_hi;
        pie_score_rows = min(params.observe, qo_len);
    }

    workgroupBarrier();
    let page_first = workgroupUniformLoad(&pie_score_first);
    let capacity = workgroupUniformLoad(&pie_score_cap);
    let qo_hi = workgroupUniformLoad(&pie_score_qohi);
    let rows = workgroupUniformLoad(&pie_score_rows);

    if (capacity <= 0 || rows <= 0) {
        return;
    }

    let kv_head = u32(head / (params.num_q_heads / params.num_kv_heads));
    let inv_rows = 1.0 / f32(rows);

    for (var w = 0; w < rows; w = w + 1) {
        let q_index = qo_hi - rows + w;
        if (tid == 0u) {
            let causal = position_ids[u32(q_index)] + 1;
            let limit = min(causal, capacity);
            pie_score_limit = limit;
            pie_score_steps = (max(limit, 0) + i32(PIE_STRIPS) - 1) / i32(PIE_STRIPS);
        }
        workgroupBarrier();
        let limit = workgroupUniformLoad(&pie_score_limit);
        let steps = workgroupUniformLoad(&pie_score_steps);

        let q_row = (u32(q_index) * u32(params.num_q_heads) + u32(head)) * head_dim;
        for (var d = tid; d < head_dim; d = d + PIE_THREADS) {
            let e = q_row + d;
            pie_score_q[d] = pie_bf16_at(q[e >> 1u], e);
        }
        workgroupBarrier();

        var running_max = PIE_NEG_INF;
        var running_sum = 0.0;
        for (var n = 0; n < steps; n = n + 1) {
            let j = n * i32(PIE_STRIPS) + i32(strip);
            let live = j < limit;
            var raw = 0.0;
            if (live) {
                raw = pie_score_dot(j, kv_head, lane, page_first, head_dim);
            }
            let dot = pie_score_strip_sum(strip, lane, raw);
            if (live) {
                let score = dot * params.sm_scale;
                let widened = max(running_max, score);
                running_sum = running_sum * exp(running_max - widened) + exp(score - widened);
                running_max = widened;
            }
        }
        if (lane == 0u) {
            pie_score_m[strip] = running_max;
            pie_score_l[strip] = running_sum;
        }
        workgroupBarrier();

        var folded_max = PIE_NEG_INF;
        for (var u = 0u; u < PIE_STRIPS; u = u + 1u) {
            folded_max = max(folded_max, pie_score_m[u]);
        }
        var denominator = 0.0;
        for (var u = 0u; u < PIE_STRIPS; u = u + 1u) {
            denominator = denominator + pie_score_l[u] * exp(pie_score_m[u] - folded_max);
        }
        var inv = 0.0;
        if (denominator > 0.0) {
            inv = 1.0 / denominator;
        }

        for (var n = 0; n < steps; n = n + 1) {
            let j = n * i32(PIE_STRIPS) + i32(strip);
            let live = j < limit;
            var raw = 0.0;
            if (live) {
                raw = pie_score_dot(j, kv_head, lane, page_first, head_dim);
            }
            let dot = pie_score_strip_sum(strip, lane, raw);

            if (lane == 0u && live && j < params.kv_max) {
                let at = out_base + u32(j);
                scores[at] = scores[at]
                    + exp(dot * params.sm_scale - folded_max) * inv * inv_rows;
            }
        }

        workgroupBarrier();
    }
}

// pie:instantiate attn_score_capture_bfloat16_d_64 PIE_HEAD_DIM_MAX=64
// pie:instantiate attn_score_capture_bfloat16_d_128 PIE_HEAD_DIM_MAX=128
// pie:instantiate attn_score_capture_bfloat16_d_256 PIE_HEAD_DIM_MAX=256
