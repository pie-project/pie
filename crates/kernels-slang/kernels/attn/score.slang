#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/pool.slang"

// Score capture (`attn.score_capture`, `attn.score_pages`): each capturing
// lane's one f32 row, a mass a key of the load's context, summed over this
// layer's query heads and averaged over the lane's last `observe` query rows
// and the model's query heads; then, once after the last layer, the row
// summed a kv page.
//
// A fire's capturing lanes are compacted: `capture` holds seven ints for
// each, in lane order: the lane, its first and end query row in the fire,
// then the pool segment and word its key row starts at, and its page row's
// (segment -1 where its program reads no pages). A lane's rows lie at its
// reading instance's place in the chunk pool (`prim/pool.slang`'s four
// segments), each whole in one chunk, so in one segment: they are addressed
// through those places alone, wherever the host lays them.
//
// Two dispatches a layer, wide over the keys, deterministic (no atomics, one
// order a key, so a run reproduces):
//   - `PIE_NORM`, a workgroup a (capturing lane, query head): each observed
//     row's softmax maximum and inverse sum, into `stats`;
//   - the mass kernel, a workgroup a (256 keys, capturing lane), a thread a
//     key: every head's and observed row's mass summed in the thread (no
//     per-head row is ever laid), added into the lane's key row or into
//     zero.
// `PIE_PAGES` sums the final key row a page, a thread a page.

#define PIE_THREADS 256
#define NEG_INF -3.0e38

#define TABLE_INTS 7u

#if defined(PIE_PAGES)

[[vk::binding(0)]] StructuredBuffer<int> capture;
// The chunk pool, its segments read and written as f32 words.
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 1)

struct Params {
    int context;
    int pages;
    int page_size;
};
PIE_PARAMS(5, Params, params);

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint c = group.y;
    uint keys_seg = uint(capture[c * TABLE_INTS + 3u]);
    uint keys_at = uint(capture[c * TABLE_INTS + 4u]);
    int pages_seg = capture[c * TABLE_INTS + 5u];
    uint pages_at = uint(capture[c * TABLE_INTS + 6u]);
    int p = int(group.x) * PIE_THREADS + int(local.x);
    if (pages_seg < 0 || p >= params.pages) {
        return;
    }
    int lo = p * params.page_size;
    int hi = min(lo + params.page_size, params.context);
    float sum = 0.0;
#define SCORE_PAGE_SUM(buf_) \
    for (int j = lo; j < hi; j++) { sum += asfloat(buf_[keys_at + uint(j)]); }
    PIE_POOL_BRANCH(keys_seg, SCORE_PAGE_SUM)
#define SCORE_PAGE_PUT(buf_) buf_[pages_at + uint(p)] = asuint(sum);
    PIE_POOL_BRANCH(uint(pages_seg), SCORE_PAGE_PUT)
}

// pie:instantiate attn_score_pages PIE_PAGES=1

#else

#ifndef PIE_HEAD_DIM_MAX
#error "attn/score.slang is compiled at a head width bound: -DPIE_HEAD_DIM_MAX"
#endif

[[vk::binding(0)]] StructuredBuffer<uint> q;
[[vk::binding(1)]] StructuredBuffer<int> capture;
[[vk::binding(2)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(3)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(4)]] StructuredBuffer<int> position_ids;
#if defined(PIE_NORM)
[[vk::binding(5)]] RWStructuredBuffer<float> stats;
#define PIE_SEG_AT 6
#define PIE_PARAMS_AT 10
// The key pages, in the chunk pool's segments (`prim/pool.slang`).
PIE_POOL_SEGMENTS(StructuredBuffer, uint, PIE_SEG_AT)
#else
[[vk::binding(5)]] StructuredBuffer<float> stats;
#define PIE_SEG_AT 6
#define PIE_PARAMS_AT 10
// The key pages, and the capturing lanes' key rows at their readers'
// places, in the chunk pool's segments (`prim/pool.slang`).
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, PIE_SEG_AT)
#endif

struct Params {
    uint seg_words;
    uint k_origin;
    uint page_stride;
    int page_size;
    // This rank's query heads: the q row's.
    int q_heads;
    int kv_heads;
    int head_dim;
    float sm_scale;
    int observe;
    // The keys of the load's context: a row's width.
    int context;
    // Whether the row is added into (1) or into zero (0).
    int into;
    // The model's query heads, every rank's: what the mass is averaged over.
    int q_heads_total;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);
PIE_POOL_READ(uint, params.seg_words)

groupshared float score_q[PIE_HEAD_DIM_MAX];
groupshared int score_first;
groupshared int score_cap;
groupshared int score_qohi;
groupshared int score_rows;
groupshared uint score_keys_seg;
groupshared uint score_keys_at;

// The lane's capture entry, staged for the workgroup: read uniformly, as
// WGSL lets a barrier follow only workgroup-uniform control flow.
void stage_lane(uint c, uint tid) {
    if (tid == 0u) {
        int lane = capture[c * TABLE_INTS];
        int qo_lo = capture[c * TABLE_INTS + 1u];
        int qo_hi = capture[c * TABLE_INTS + 2u];
        score_keys_seg = uint(capture[c * TABLE_INTS + 3u]);
        score_keys_at = uint(capture[c * TABLE_INTS + 4u]);
        int page_first = int(kv_page_indptr[uint(lane)]);
        int pages = int(kv_page_indptr[uint(lane) + 1u]) - page_first;
        score_first = page_first;
        score_cap = pages * params.page_size;
        score_qohi = qo_hi;
        score_rows = min(params.observe, qo_hi - qo_lo);
    }
    GroupMemoryBarrierWithGroupSync();
}

// Query row `qi`'s head `head`, staged as f32.
void stage_q(int qi, int head, uint tid) {
    uint head_dim = uint(params.head_dim);
    uint at = (uint(qi) * uint(params.q_heads) + uint(head)) * head_dim;
    for (uint d = tid; d < head_dim; d += uint(PIE_THREADS)) {
        score_q[d] = bf16_at(q, at + d);
    }
    GroupMemoryBarrierWithGroupSync();
}

// Key `j` of the lane's pages, kv head `kv_head`, dotted with the staged
// query: the row's words from the one segment its page lies in.
float key_dot(int j, uint kv_head, int page_first) {
    uint head_dim = uint(params.head_dim);
    uint page = kv_page_indices[uint(page_first + j / params.page_size)];
    uint half_at = page * (2u * params.page_stride)
        + (uint(j % params.page_size) * uint(params.kv_heads) + kv_head) * head_dim;
    uint seg, at;
    pie_pool_find(params.k_origin + (half_at >> 1), seg, at);
    float dot = 0.0;
    uint pairs = head_dim >> 1;
    // Eight words loaded before any is summed: a load at a time, each
    // waiting on the last, leaves a thread's row latency-bound.
#define SCORE_KEY_WORDS(buf_) \
    for (uint p0 = 0u; p0 < pairs; p0 += 8u) { \
        uint words[8]; \
        [ForceUnroll] for (uint k = 0u; k < 8u; k++) { words[k] = buf_[at + p0 + k]; } \
        [ForceUnroll] for (uint k = 0u; k < 8u; k++) { \
            uint p = p0 + k; \
            dot = fma(score_q[2u * p], bf16_to_f32(words[k] & 0xffffu), dot); \
            dot = fma(score_q[2u * p + 1u], bf16_to_f32(words[k] >> 16), dot); \
        } \
    }
    PIE_POOL_BRANCH(seg, SCORE_KEY_WORDS)
    return dot;
}

#if defined(PIE_NORM)

groupshared float score_m[PIE_THREADS];
groupshared float score_l[PIE_THREADS];

// The workgroup's maximum and sum of `m` and `l` (each thread's online
// softmax over its keys), left in `score_m[0]` and `score_l[0]`.
void fold_softmax(uint tid, float m, float l) {
    score_m[tid] = m;
    score_l[tid] = l;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_ = uint(PIE_THREADS) >> 1; half_ > 0u; half_ >>= 1) {
        if (tid < half_) {
            float m0 = score_m[tid];
            float m1 = score_m[tid + half_];
            float wide = max(m0, m1);
            score_l[tid] = score_l[tid] * exp(m0 - wide) + score_l[tid + half_] * exp(m1 - wide);
            score_m[tid] = wide;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

#endif

#if defined(PIE_NORM)

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint c = group.x;
    int head = int(group.y);
    uint tid = local.x;
    stage_lane(c, tid);
    int page_first = workgroupUniformLoad(score_first);
    int capacity = workgroupUniformLoad(score_cap);
    int qo_hi = workgroupUniformLoad(score_qohi);
    int rows = workgroupUniformLoad(score_rows);
    uint kv_head = uint(head / (params.q_heads / params.kv_heads));
    uint at = (c * uint(params.q_heads) + uint(head)) * uint(params.observe);

    for (int w = 0; w < rows; w++) {
        int qi = qo_hi - rows + w;
        stage_q(qi, head, tid);
        int limit = min(position_ids[uint(qi)] + 1, capacity);
        float m = NEG_INF;
        float l = 0.0;
        for (int j = int(tid); j < limit; j += PIE_THREADS) {
            float s = key_dot(j, kv_head, page_first) * params.sm_scale;
            float wide = max(m, s);
            l = l * exp(m - wide) + exp(s - wide);
            m = wide;
        }
        fold_softmax(tid, m, l);
        if (tid == 0u) {
            float sum = score_l[0];
            stats[(at + uint(w)) * 2u] = score_m[0];
            stats[(at + uint(w)) * 2u + 1u] = sum > 0.0 ? 1.0 / sum : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// pie:instantiate attn_score_norm_bf16_d_64 PIE_NORM=1 PIE_HEAD_DIM_MAX=64
// pie:instantiate attn_score_norm_bf16_d_128 PIE_NORM=1 PIE_HEAD_DIM_MAX=128
// pie:instantiate attn_score_norm_bf16_d_256 PIE_NORM=1 PIE_HEAD_DIM_MAX=256

#else

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint c = group.y;
    uint tid = local.x;
    int j = int(group.x) * PIE_THREADS + int(tid);
    stage_lane(c, tid);
    int page_first = workgroupUniformLoad(score_first);
    int capacity = workgroupUniformLoad(score_cap);
    int qo_hi = workgroupUniformLoad(score_qohi);
    int rows = workgroupUniformLoad(score_rows);
    uint keys_seg = workgroupUniformLoad(score_keys_seg);
    uint keys_at = workgroupUniformLoad(score_keys_at);
    int groups = params.q_heads / params.kv_heads;

    float acc = 0.0;
    // A tile past the lane's keys holds no mass: its row is zeroed or kept.
    if (int(group.x) * PIE_THREADS < capacity) {
        for (int head = 0; head < params.q_heads; head++) {
            uint kv_head = uint(head / groups);
            uint at = (c * uint(params.q_heads) + uint(head)) * uint(params.observe);
            for (int w = 0; w < rows; w++) {
                int qi = qo_hi - rows + w;
                stage_q(qi, head, tid);
                int limit = min(position_ids[uint(qi)] + 1, capacity);
                if (j < limit) {
                    float m = stats[(at + uint(w)) * 2u];
                    float inv = stats[(at + uint(w)) * 2u + 1u];
                    acc += exp(key_dot(j, kv_head, page_first) * params.sm_scale - m) * inv;
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
    }
    if (j < params.context) {
        float share = rows > 0 ? 1.0 / (float(rows) * float(params.q_heads_total)) : 0.0;
        uint dst = keys_at + uint(j);
#define SCORE_ROW_ADD(buf_) \
        { \
            float prior = params.into != 0 ? asfloat(buf_[dst]) : 0.0; \
            buf_[dst] = asuint(prior + acc * share); \
        }
        PIE_POOL_BRANCH(keys_seg, SCORE_ROW_ADD)
    }
}

// pie:instantiate attn_score_mass_bf16_d_64 PIE_HEAD_DIM_MAX=64
// pie:instantiate attn_score_mass_bf16_d_128 PIE_HEAD_DIM_MAX=128
// pie:instantiate attn_score_mass_bf16_d_256 PIE_HEAD_DIM_MAX=256

#endif
#endif
