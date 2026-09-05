#include "common/bf16.slang"

#ifndef PIE_HEAD_DIM_MAX
#error "score requires -DPIE_HEAD_DIM_MAX"
#endif

#define PIE_STRIPS 8
#define PIE_LANES 32
#define PIE_THREADS (PIE_STRIPS * PIE_LANES)
#define PIE_VPT (PIE_HEAD_DIM_MAX / PIE_LANES)

#define PIE_NEG_INF (-3.0e38)

PIE_BUFFER_RO(0, q);
[[vk::binding(1)]] StructuredBuffer<int> qo_indptr;
PIE_BUFFER_RO(2, k_pages);
[[vk::binding(3)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(4)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(5)]] StructuredBuffer<int> position_ids;
[[vk::binding(6)]] RWStructuredBuffer<float> scores;

struct Push {
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
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_score_q[PIE_HEAD_DIM_MAX];
groupshared float pie_score_fold[PIE_STRIPS][PIE_LANES];
groupshared float pie_score_m[PIE_STRIPS];
groupshared float pie_score_l[PIE_STRIPS];

float pie_score_strip_sum(int strip, int lane, float v) {
    pie_score_fold[strip][lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_ = PIE_LANES >> 1; half_ > 0u; half_ >>= 1) {
        if (uint(lane) < half_) {
            pie_score_fold[strip][lane] += pie_score_fold[strip][lane + int(half_)];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float total = pie_score_fold[strip][0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

float pie_score_dot(int j, int kv_head, int lane, int page_first, uint head_dim) {
    uint slot = kv_page_indices[uint(page_first + j / pc.page_size)] * uint(pc.page_size)
        + uint(j % pc.page_size);
    uint row_stride = uint(pc.num_kv_heads) * head_dim;
    uint k_row = slot * row_stride + uint(kv_head) * head_dim;
    float dot = 0.0;
    for (int u = 0; u < PIE_VPT; u++) {
        uint d = uint(lane + u * PIE_LANES);
        if (d < head_dim) {
            dot += pie_score_q[d] * PIE_LOAD(k_pages[k_row + d]);
        }
    }
    return dot;
}

[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int request = int(group.x);
    int head = int(group.y);
    int tid = int(local.x);
    int strip = tid / PIE_LANES;
    int lane = tid % PIE_LANES;
    uint head_dim = uint(pc.head_dim);

    uint out_base = (uint(pc.lane_offset + request) * uint(pc.plane_stride)
        + uint(pc.plane + head)) * uint(pc.kv_max);
    for (uint i = uint(tid); i < uint(pc.kv_max); i += PIE_THREADS) {
        scores[out_base + i] = 0.0;
    }

    int page_first = int(kv_page_indptr[uint(request)]);
    int pages = int(kv_page_indptr[uint(request) + 1u]) - page_first;

    int capacity = pages * pc.page_size;
    int qo_hi = qo_indptr[uint(request) + 1u];
    int qo_len = qo_hi - qo_indptr[uint(request)];
    int rows = min(pc.observe, qo_len);

    GroupMemoryBarrierWithGroupSync();

    if (pages <= 0 || capacity <= 0 || rows <= 0) {
        return;
    }

    int group_size = pc.num_q_heads / pc.num_kv_heads;
    int kv_head = head / group_size;
    float inv_rows = 1.0 / float(rows);

    for (int w = 0; w < rows; w++) {
        int q_index = qo_hi - rows + w;
        int causal = position_ids[uint(q_index)] + 1;
        int limit = min(causal, capacity);
        int steps = (max(limit, 0) + PIE_STRIPS - 1) / PIE_STRIPS;

        uint q_row = (uint(q_index) * uint(pc.num_q_heads) + uint(head)) * head_dim;
        for (uint d = uint(tid); d < head_dim; d += PIE_THREADS) {
            pie_score_q[d] = PIE_LOAD(q[q_row + d]);
        }
        GroupMemoryBarrierWithGroupSync();

        float running_max = PIE_NEG_INF;
        float running_sum = 0.0;
        for (int n = 0; n < steps; n++) {
            int j = n * PIE_STRIPS + strip;
            bool live = j < limit;
            float dot = live ? pie_score_dot(j, kv_head, lane, page_first, head_dim) : 0.0;
            dot = pie_score_strip_sum(strip, lane, dot);
            if (live) {
                float score = dot * pc.sm_scale;
                float widened = max(running_max, score);
                running_sum = running_sum * exp(running_max - widened) + exp(score - widened);
                running_max = widened;
            }
        }
        if (lane == 0) {
            pie_score_m[strip] = running_max;
            pie_score_l[strip] = running_sum;
        }
        GroupMemoryBarrierWithGroupSync();

        float folded_max = PIE_NEG_INF;
        for (int u = 0; u < PIE_STRIPS; u++) {
            folded_max = max(folded_max, pie_score_m[u]);
        }
        float denominator = 0.0;
        for (int u = 0; u < PIE_STRIPS; u++) {
            denominator += pie_score_l[u] * exp(pie_score_m[u] - folded_max);
        }
        float inv = denominator > 0.0 ? 1.0 / denominator : 0.0;

        for (int n = 0; n < steps; n++) {
            int j = n * PIE_STRIPS + strip;
            bool live = j < limit;
            float dot = live ? pie_score_dot(j, kv_head, lane, page_first, head_dim) : 0.0;
            dot = pie_score_strip_sum(strip, lane, dot);

            if (lane == 0 && live && j < pc.kv_max) {
                scores[out_base + uint(j)] += exp(dot * pc.sm_scale - folded_max) * inv * inv_rows;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }
}

// pie:instantiate attn_score_capture_bfloat16_d_64 PIE_HEAD_DIM_MAX=64
// pie:instantiate attn_score_capture_bfloat16_d_128 PIE_HEAD_DIM_MAX=128
// pie:instantiate attn_score_capture_bfloat16_d_256 PIE_HEAD_DIM_MAX=256
