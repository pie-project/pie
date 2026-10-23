#include "prim/bf16.slang"
#include "prim/params.slang"

// Attention mass capture, one workgroup a (request, query head): each of the
// request's last `observe` query rows takes a causal softmax over the keys
// in its pages, and the mean of those rows' probabilities lands per key in
// an f32 slab row (lane `lane_offset + request`, plane `plane + head`, a row
// `kv_max` wide). Eight strips of 32 lanes walk the keys; a dot reduces
// through workgroup memory, no subgroup operation. Two passes a row: the
// first finds the softmax's maximum and sum, the second lands the mass.

#ifndef PIE_HEAD_DIM_MAX
#error "attn/score.slang is compiled at a head width bound: -DPIE_HEAD_DIM_MAX"
#endif
#define PIE_STRIPS 8
#define PIE_LANES 32
#define PIE_THREADS (PIE_STRIPS * PIE_LANES)
#define PIE_VPT (PIE_HEAD_DIM_MAX / PIE_LANES)
#define NEG_INF -3.0e38

[[vk::binding(0)]] StructuredBuffer<uint> q;
[[vk::binding(1)]] StructuredBuffer<int> qo_indptr;
[[vk::binding(2)]] StructuredBuffer<uint> k_pages;
[[vk::binding(3)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(4)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(5)]] StructuredBuffer<int> position_ids;
[[vk::binding(6)]] RWStructuredBuffer<float> scores;

struct Params {
    int page_size;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    float sm_scale;
    int observe;
    int lane_offset;
    int plane_stride;
    int plane;
    int kv_max;
};
PIE_PARAMS(7, Params, params);

groupshared float score_q[PIE_HEAD_DIM_MAX];
groupshared float score_fold[PIE_STRIPS][PIE_LANES];
groupshared float score_m[PIE_STRIPS];
groupshared float score_l[PIE_STRIPS];
groupshared int score_first;
groupshared int score_cap;
groupshared int score_qohi;
groupshared int score_rows;
groupshared int score_limit;
groupshared int score_steps;

float strip_sum(uint strip, uint lane, float v) {
    score_fold[strip][lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_ = uint(PIE_LANES) >> 1; half_ > 0u; half_ >>= 1) {
        if (lane < half_) {
            score_fold[strip][lane] += score_fold[strip][lane + half_];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float total = score_fold[strip][0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// Key `j` of the request's pages dotted with the staged query row.
float key_dot(int j, uint kv_head, uint lane, int page_first, uint head_dim) {
    uint page = kv_page_indices[uint(page_first + j / params.page_size)];
    uint slot = page * uint(params.page_size) + uint(j % params.page_size);
    uint k_row = (slot * uint(params.num_kv_heads) + kv_head) * head_dim;
    float dot = 0.0;
    for (uint u = 0u; u < uint(PIE_VPT); u++) {
        uint d = lane + u * uint(PIE_LANES);
        if (d < head_dim) {
            dot += score_q[d] * bf16_at(k_pages, k_row + d);
        }
    }
    return dot;
}

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int request = int(group.x);
    int head = int(group.y);
    uint tid = local.x;
    uint strip = tid / uint(PIE_LANES);
    uint lane = tid % uint(PIE_LANES);
    uint head_dim = uint(params.head_dim);

    uint out_base = (uint(params.lane_offset + request) * uint(params.plane_stride)
        + uint(params.plane + head)) * uint(params.kv_max);
    for (uint i = tid; i < uint(params.kv_max); i += uint(PIE_THREADS)) {
        scores[out_base + i] = 0.0;
    }

    if (tid == 0u) {
        int page_first = int(kv_page_indptr[uint(request)]);
        int pages = int(kv_page_indptr[uint(request) + 1u]) - page_first;
        int qo_hi = qo_indptr[uint(request) + 1u];
        score_first = page_first;
        score_cap = pages * params.page_size;
        score_qohi = qo_hi;
        score_rows = min(params.observe, qo_hi - qo_indptr[uint(request)]);
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: WGSL lets a barrier follow only workgroup-uniform
    // control flow, and these bound the loops that hold barriers.
    int page_first = workgroupUniformLoad(score_first);
    int capacity = workgroupUniformLoad(score_cap);
    int qo_hi = workgroupUniformLoad(score_qohi);
    int rows = workgroupUniformLoad(score_rows);
    if (capacity <= 0 || rows <= 0) {
        return;
    }

    uint kv_head = uint(head / (params.num_q_heads / params.num_kv_heads));
    float inv_rows = 1.0 / float(rows);

    for (int w = 0; w < rows; w++) {
        int q_index = qo_hi - rows + w;
        if (tid == 0u) {
            int limit = min(position_ids[uint(q_index)] + 1, capacity);
            score_limit = limit;
            score_steps = (max(limit, 0) + PIE_STRIPS - 1) / PIE_STRIPS;
        }
        uint q_row = (uint(q_index) * uint(params.num_q_heads) + uint(head)) * head_dim;
        for (uint d = tid; d < head_dim; d += uint(PIE_THREADS)) {
            score_q[d] = bf16_at(q, q_row + d);
        }
        GroupMemoryBarrierWithGroupSync();
        int limit = workgroupUniformLoad(score_limit);
        int steps = workgroupUniformLoad(score_steps);

        float running_max = NEG_INF;
        float running_sum = 0.0;
        for (int n = 0; n < steps; n++) {
            int j = n * PIE_STRIPS + int(strip);
            bool live = j < limit;
            float dot = strip_sum(strip, lane, live ? key_dot(j, kv_head, lane, page_first, head_dim) : 0.0);
            if (live) {
                float score = dot * params.sm_scale;
                float widened = max(running_max, score);
                running_sum = running_sum * exp(running_max - widened) + exp(score - widened);
                running_max = widened;
            }
        }
        if (lane == 0u) {
            score_m[strip] = running_max;
            score_l[strip] = running_sum;
        }
        GroupMemoryBarrierWithGroupSync();

        float folded_max = NEG_INF;
        for (uint u = 0u; u < uint(PIE_STRIPS); u++) {
            folded_max = max(folded_max, score_m[u]);
        }
        float denominator = 0.0;
        for (uint u = 0u; u < uint(PIE_STRIPS); u++) {
            denominator += score_l[u] * exp(score_m[u] - folded_max);
        }
        float inv = denominator > 0.0 ? 1.0 / denominator : 0.0;

        for (int n = 0; n < steps; n++) {
            int j = n * PIE_STRIPS + int(strip);
            bool live = j < limit;
            float dot = strip_sum(strip, lane, live ? key_dot(j, kv_head, lane, page_first, head_dim) : 0.0);
            if (lane == 0u && live && j < params.kv_max) {
                uint at = out_base + uint(j);
                scores[at] = scores[at] + exp(dot * params.sm_scale - folded_max) * inv * inv_rows;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// pie:instantiate attn_score_capture_bf16_d_64 PIE_HEAD_DIM_MAX=64
// pie:instantiate attn_score_capture_bf16_d_128 PIE_HEAD_DIM_MAX=128
// pie:instantiate attn_score_capture_bf16_d_256 PIE_HEAD_DIM_MAX=256
