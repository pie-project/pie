#include "prim/pair.slang"
#include "prim/params.slang"

// Dense attention over segments, one workgroup a (query head, row): the row
// attends every key of its segment, bidirectionally. The segment is the
// row's own (`segments`, a patch window's images) or, with `PIE_RAGGED`, the
// keys `kv_segments` names for it. Four strips of 32 lanes walk the keys,
// each keeping an online softmax; the strips fold at the end. Rows are
// `PIE_ACT` (bf16, or f32); a lane lands two head columns. No subgroup
// operation: the dot products reduce through workgroup memory.

#ifndef PIE_HEAD_DIM_MAX
#error "attn/dense.slang is compiled at a head width bound: -DPIE_HEAD_DIM_MAX"
#endif
#define PIE_STRIPS 4
#define PIE_LANES 32
#define PIE_THREADS (PIE_STRIPS * PIE_LANES)
#define PIE_VPT (PIE_HEAD_DIM_MAX / PIE_LANES)

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> q;
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> k;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> v;
[[vk::binding(3)]] RWStructuredBuffer<PIE_ACT> o;
[[vk::binding(4)]] StructuredBuffer<int> segments;
#ifdef PIE_RAGGED
[[vk::binding(5)]] StructuredBuffer<int> kv_segments;
#define PIE_PARAMS_AT 6
#else
#define PIE_PARAMS_AT 5
#endif

struct Params {
    int num_segments;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    float sm_scale;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

groupshared float dense_q[PIE_HEAD_DIM_MAX];
groupshared float dense_fold[PIE_STRIPS][PIE_LANES];
groupshared float dense_acc[PIE_STRIPS * PIE_HEAD_DIM_MAX];
groupshared float dense_m[PIE_STRIPS];
groupshared float dense_l[PIE_STRIPS];
groupshared int dense_begin;
groupshared int dense_end;

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int head = int(group.x);
    int row = int(group.y);
    uint tid = local.x;
    uint strip = tid / uint(PIE_LANES);
    uint lane = tid % uint(PIE_LANES);
    uint head_dim = uint(params.head_dim);

    // The key span of this row's segment, by bisection over the boundaries.
    if (tid == 0u) {
        int begin = -1;
        int end = -1;
        if (row >= segments[0] && row < segments[uint(params.num_segments)]) {
            int lo = 0;
            int hi = params.num_segments - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (segments[uint(mid)] <= row) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
#ifdef PIE_RAGGED
            begin = kv_segments[uint(lo)];
            end = kv_segments[uint(lo + 1)];
#else
            begin = segments[uint(lo)];
            end = segments[uint(lo + 1)];
#endif
        }
        dense_begin = begin;
        dense_end = end;
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: WGSL lets a barrier follow only workgroup-uniform
    // control flow, and the span bounds the key loop.
    int begin = workgroupUniformLoad(dense_begin);
    int end = workgroupUniformLoad(dense_end);

    uint out_row = (uint(row) * uint(params.num_q_heads) + uint(head)) * head_dim;
    if (end <= begin) {
        for (uint d = 2u * tid; d < head_dim; d += 2u * uint(PIE_THREADS)) {
            pair_put(o, out_row + d, true, float2(0.0, 0.0));
        }
        return;
    }
    for (uint d = tid; d < head_dim; d += uint(PIE_THREADS)) {
        dense_q[d] = act_at(q, out_row + d);
    }
    GroupMemoryBarrierWithGroupSync();

    uint kv_head = uint(head / (params.num_q_heads / params.num_kv_heads));
    float acc[PIE_VPT];
    for (uint u = 0u; u < uint(PIE_VPT); u++) {
        acc[u] = 0.0;
    }
    float running_max = -3.0e38;
    float running_sum = 0.0;

    int steps = (end - begin + PIE_STRIPS - 1) / PIE_STRIPS;
    for (int n = 0; n < steps; n++) {
        int j = begin + n * PIE_STRIPS + int(strip);
        bool live = j < end;
        uint k_row = live ? (uint(j) * uint(params.num_kv_heads) + kv_head) * head_dim : 0u;
        float dot = 0.0;
        if (live) {
            for (uint u = 0u; u < uint(PIE_VPT); u++) {
                uint d = lane + u * uint(PIE_LANES);
                if (d < head_dim) {
                    dot += dense_q[d] * act_at(k, k_row + d);
                }
            }
        }
        dense_fold[strip][lane] = dot;
        GroupMemoryBarrierWithGroupSync();
        for (uint half_ = uint(PIE_LANES) >> 1; half_ > 0u; half_ >>= 1) {
            if (lane < half_) {
                dense_fold[strip][lane] += dense_fold[strip][lane + half_];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        dot = dense_fold[strip][0];
        GroupMemoryBarrierWithGroupSync();
        if (live) {
            float score = dot * params.sm_scale;
            float widened = max(running_max, score);
            float rescale = exp(running_max - widened);
            float weight = exp(score - widened);
            for (uint u = 0u; u < uint(PIE_VPT); u++) {
                uint d = lane + u * uint(PIE_LANES);
                if (d < head_dim) {
                    acc[u] = acc[u] * rescale + weight * act_at(v, k_row + d);
                }
            }
            running_sum = running_sum * rescale + weight;
            running_max = widened;
        }
    }

    if (lane == 0u) {
        dense_m[strip] = running_max;
        dense_l[strip] = running_sum;
    }
    for (uint u = 0u; u < uint(PIE_VPT); u++) {
        uint d = lane + u * uint(PIE_LANES);
        if (d < head_dim) {
            dense_acc[strip * uint(PIE_HEAD_DIM_MAX) + d] = acc[u];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float folded_max = -3.0e38;
    for (uint w = 0u; w < uint(PIE_STRIPS); w++) {
        folded_max = max(folded_max, dense_m[w]);
    }
    float denominator = 0.0;
    for (uint w = 0u; w < uint(PIE_STRIPS); w++) {
        denominator += dense_l[w] * exp(dense_m[w] - folded_max);
    }
    float inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
    for (uint d = 2u * tid; d < head_dim; d += 2u * uint(PIE_THREADS)) {
        float2 sum = float2(0.0, 0.0);
        for (uint w = 0u; w < uint(PIE_STRIPS); w++) {
            float scale = exp(dense_m[w] - folded_max);
            uint at = w * uint(PIE_HEAD_DIM_MAX) + d;
            sum += float2(dense_acc[at], dense_acc[at + 1u]) * scale;
        }
        pair_put(o, out_row + d, true, sum * inv);
    }
}

// pie:instantiate attn_dense_bf16_d_64 PIE_HEAD_DIM_MAX=64
// pie:instantiate attn_dense_bf16_d_128 PIE_HEAD_DIM_MAX=128
// pie:instantiate attn_dense_bf16_d_256 PIE_HEAD_DIM_MAX=256
// pie:instantiate attn_dense_ragged_bf16_d_64 PIE_HEAD_DIM_MAX=64 PIE_RAGGED=1
// pie:instantiate attn_dense_ragged_bf16_d_128 PIE_HEAD_DIM_MAX=128 PIE_RAGGED=1
// pie:instantiate attn_dense_ragged_bf16_d_256 PIE_HEAD_DIM_MAX=256 PIE_RAGGED=1
// pie:instantiate attn_dense_ragged_f32_d_64 PIE_HEAD_DIM_MAX=64 PIE_RAGGED=1 PIE_ACT_F32=1
// pie:instantiate attn_dense_ragged_f32_d_128 PIE_HEAD_DIM_MAX=128 PIE_RAGGED=1 PIE_ACT_F32=1
// pie:instantiate attn_dense_ragged_f32_d_256 PIE_HEAD_DIM_MAX=256 PIE_RAGGED=1 PIE_ACT_F32=1
