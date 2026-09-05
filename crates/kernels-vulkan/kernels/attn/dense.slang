#include "common/bf16.slang"

#ifndef PIE_HEAD_DIM_MAX
#error "dense requires -DPIE_HEAD_DIM_MAX"
#endif
#define PIE_STRIPS 4
#define PIE_LANES 32
#define PIE_THREADS (PIE_STRIPS * PIE_LANES)
#define PIE_VPT (PIE_HEAD_DIM_MAX / PIE_LANES)

PIE_BUFFER_RO(0, q);
PIE_BUFFER_RO(1, k);
PIE_BUFFER_RO(2, v);
PIE_BUFFER_RW(3, o);
[[vk::binding(4)]] StructuredBuffer<int> segments;
struct Push { int num_segments; int num_q_heads; int num_kv_heads; int head_dim; float sm_scale; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_dense_q[PIE_HEAD_DIM_MAX];
groupshared float pie_dense_fold[PIE_STRIPS][PIE_LANES];
groupshared float pie_dense_acc[PIE_STRIPS * PIE_HEAD_DIM_MAX];
groupshared float pie_dense_m[PIE_STRIPS];
groupshared float pie_dense_l[PIE_STRIPS];
groupshared int pie_dense_span[2];

[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int head = int(group.x);
    int row = int(group.y);
    int tid = int(local.x);
    int strip = tid / PIE_LANES;
    int lane = tid % PIE_LANES;
    uint head_dim = uint(pc.head_dim);

    if (tid == 0) {
        int begin = -1;
        int end = -1;
        int first = segments[0];
        int total = segments[uint(pc.num_segments)];
        if (row >= first && row < total) {
            int lo = 0;
            int hi = pc.num_segments - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (segments[uint(mid)] <= row) lo = mid; else hi = mid - 1;
            }
            begin = segments[uint(lo)];
            end = segments[uint(lo + 1)];
        }
        pie_dense_span[0] = begin;
        pie_dense_span[1] = end;
    }
    GroupMemoryBarrierWithGroupSync();
    int begin = pie_dense_span[0];
    int end = pie_dense_span[1];

    uint out_row = (uint(row) * uint(pc.num_q_heads) + uint(head)) * head_dim;
    if (end <= begin) {
        for (uint d = uint(tid); d < head_dim; d += PIE_THREADS) o[out_row + d] = PIE_STORE(0.0);
        return;
    }
    for (uint d = uint(tid); d < head_dim; d += PIE_THREADS) pie_dense_q[d] = PIE_LOAD(q[out_row + d]);
    GroupMemoryBarrierWithGroupSync();

    int kv_head = head / (pc.num_q_heads / pc.num_kv_heads);
    float acc[PIE_VPT];
    for (int u = 0; u < PIE_VPT; u++) acc[u] = 0.0;
    float running_max = -3.0e38;
    float running_sum = 0.0;

    int steps = (end - begin + PIE_STRIPS - 1) / PIE_STRIPS;
    for (int n = 0; n < steps; n++) {
        int j = begin + n * PIE_STRIPS + strip;
        bool live = j < end;
        uint k_row = live ? (uint(j) * uint(pc.num_kv_heads) + uint(kv_head)) * head_dim : 0u;
        float dot = 0.0;
        if (live) {
            for (int u = 0; u < PIE_VPT; u++) {
                uint d = uint(lane + u * PIE_LANES);
                if (d < head_dim) dot += pie_dense_q[d] * PIE_LOAD(k[k_row + d]);
            }
        }
        pie_dense_fold[strip][lane] = dot;
        GroupMemoryBarrierWithGroupSync();
        for (uint half_ = PIE_LANES >> 1; half_ > 0u; half_ >>= 1) {
            if (uint(lane) < half_) pie_dense_fold[strip][lane] += pie_dense_fold[strip][lane + int(half_)];
            GroupMemoryBarrierWithGroupSync();
        }
        dot = pie_dense_fold[strip][0];
        GroupMemoryBarrierWithGroupSync();
        if (live) {
            float score = dot * pc.sm_scale;
            float widened = max(running_max, score);
            float rescale = exp(running_max - widened);
            float weight = exp(score - widened);
            for (int u = 0; u < PIE_VPT; u++) {
                uint d = uint(lane + u * PIE_LANES);
                if (d < head_dim) acc[u] = acc[u] * rescale + weight * PIE_LOAD(v[k_row + d]);
            }
            running_sum = running_sum * rescale + weight;
            running_max = widened;
        }
    }

    if (lane == 0) {
        pie_dense_m[strip] = running_max;
        pie_dense_l[strip] = running_sum;
    }
    for (int u = 0; u < PIE_VPT; u++) {
        uint d = uint(lane + u * PIE_LANES);
        if (d < head_dim) pie_dense_acc[strip * PIE_HEAD_DIM_MAX + int(d)] = acc[u];
    }
    GroupMemoryBarrierWithGroupSync();

    float folded_max = -3.0e38;
    for (int w = 0; w < PIE_STRIPS; w++) folded_max = max(folded_max, pie_dense_m[w]);
    float denominator = 0.0;
    for (int w = 0; w < PIE_STRIPS; w++) denominator += pie_dense_l[w] * exp(pie_dense_m[w] - folded_max);
    float inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
    for (uint d = uint(tid); d < head_dim; d += PIE_THREADS) {
        float sum = 0.0;
        for (int w = 0; w < PIE_STRIPS; w++) sum += pie_dense_acc[w * PIE_HEAD_DIM_MAX + int(d)] * exp(pie_dense_m[w] - folded_max);
        o[out_row + d] = PIE_STORE(sum * inv);
    }
}

// pie:instantiate dense_bidirectional_bf16_d_64 PIE_HEAD_DIM_MAX=64
// pie:instantiate dense_bidirectional_bf16_d_128 PIE_HEAD_DIM_MAX=128
// pie:instantiate dense_bidirectional_bf16_d_256 PIE_HEAD_DIM_MAX=256
