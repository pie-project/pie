#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/pool.slang"

// Score capture (`attn.score_capture`, `attn.score_pages`): each capturing
// lane's one f32 row, a mass a key of the load's context, summed over this
// layer's query heads and averaged over the lane's last `observe` query rows
// and the model's query heads; then, once after the last layer, the row
// summed a kv page.
//
// A fire's capturing lanes are compacted: `capture` holds five ints for
// each, in lane order: the lane, its first and end query row in the fire,
// and the f32 its key row starts at in `pie_room_scores` and its page row in
// `pie_room_pages` (-1 where its program reads no pages). A lane's rows are
// addressed through those bases alone, wherever the host lays them.
//
// Two dispatches a layer, wide over the keys, deterministic (no atomics, one
// order a key, so a run reproduces):
//   - `PIE_NORM`, a workgroup a (capturing lane, query head): each observed
//     row's softmax maximum and inverse sum, into `stats`;
//   - the mass kernel, a workgroup a tile of keys of a capturing lane: the
//     tile's keys' rows, every kv head's, read coalesced into workgroup
//     memory with the row's query heads, a thread a (key, head) dot, and a
//     key's heads summed in head order (no per-head row is ever laid), added
//     into the lane's key row or into zero.
// `PIE_PAGES` sums the final key row a page, a thread a page.

#define PIE_THREADS 256
#define NEG_INF -3.0e38

#if defined(PIE_PAGES)

[[vk::binding(0)]] StructuredBuffer<int> capture;
// A pair's score room, which an unpark or an evacuation moves: bindings named
// `pie_room_*`, which engine-vulkan lays update-after-bind and rewrites in a
// recording as the room moves.
[[vk::binding(1)]] StructuredBuffer<float> pie_room_keys;
[[vk::binding(2)]] RWStructuredBuffer<float> pie_room_pages;

struct Params {
    int context;
    int pages;
    int page_size;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint c = group.y;
    int keys_at = capture[c * 5u + 3u];
    int pages_at = capture[c * 5u + 4u];
    int p = int(group.x) * PIE_THREADS + int(local.x);
    if (pages_at < 0 || p >= params.pages) {
        return;
    }
    int lo = p * params.page_size;
    int hi = min(lo + params.page_size, params.context);
    float sum = 0.0;
    for (int j = lo; j < hi; j++) {
        sum += pie_room_keys[uint(keys_at + j)];
    }
    pie_room_pages[uint(pages_at + p)] = sum;
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
#else
[[vk::binding(5)]] StructuredBuffer<float> stats;
[[vk::binding(6)]] RWStructuredBuffer<float> pie_room_scores;
#define PIE_SEG_AT 7
#define PIE_PARAMS_AT 11
#endif
// The key pages, in the chunk pool's segments (`prim/pool.slang`).
PIE_POOL_SEGMENTS(StructuredBuffer, uint, PIE_SEG_AT)

struct Params {
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
    // The mass kernel's keys a tile, as the host sized it to its rows.
    int tile;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);
PIE_POOL_READ(uint)

#if defined(PIE_NORM)
groupshared float score_q[PIE_HEAD_DIM_MAX];
#endif
groupshared int score_first;
groupshared int score_cap;
groupshared int score_qohi;
groupshared int score_rows;
groupshared int score_keys_at;

// The lane's capture entry, staged for the workgroup: read uniformly, as
// WGSL lets a barrier follow only workgroup-uniform control flow.
void stage_lane(uint c, uint tid) {
    if (tid == 0u) {
        int lane = capture[c * 5u];
        int qo_lo = capture[c * 5u + 1u];
        int qo_hi = capture[c * 5u + 2u];
        score_keys_at = capture[c * 5u + 3u];
        int page_first = int(kv_page_indptr[uint(lane)]);
        int pages = int(kv_page_indptr[uint(lane) + 1u]) - page_first;
        score_first = page_first;
        score_cap = pages * params.page_size;
        score_qohi = qo_hi;
        score_rows = min(params.observe, qo_hi - qo_lo);
    }
    GroupMemoryBarrierWithGroupSync();
}

#if defined(PIE_NORM)

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
    uint word_at = pie_unit_at(page, params.page_stride)
        + (((uint(j % params.page_size) * uint(params.kv_heads) + kv_head) * head_dim) >> 1);
    uint seg, at;
    pie_pool_find(params.k_origin + word_at, seg, at);
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

#endif

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

// A tile of keys' rows, every kv head's, and the row's query heads, as
// words of two bf16, in workgroup memory: each read from device memory a
// workgroup makes is coalesced (consecutive threads, consecutive words of
// one key's row), never a row a thread. Each kv head's run of a key, and
// each query head's row, is one word longer than its words, so the heads a
// warp reads at once lie in different banks.
groupshared uint mass_rows[PIE_MASS_ROWS];
groupshared uint mass_q[PIE_MASS_Q];
groupshared float mass_top[PIE_MASS_HEADS];
groupshared float mass_inv[PIE_MASS_HEADS];
groupshared float mass_term[PIE_THREADS];
groupshared int mass_limit;

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint c = group.y;
    uint tid = local.x;
    stage_lane(c, tid);
    int page_first = workgroupUniformLoad(score_first);
    int capacity = workgroupUniformLoad(score_cap);
    int qo_hi = workgroupUniformLoad(score_qohi);
    int rows = workgroupUniformLoad(score_rows);
    int keys_at = workgroupUniformLoad(score_keys_at);
    uint head_dim = uint(params.head_dim);
    uint pairs = head_dim >> 1;
    uint heads = uint(params.q_heads);
    uint kv_heads = uint(params.kv_heads);
    uint groups = heads / kv_heads;
    uint run = pairs + 1u;
    uint key_words = kv_heads * run;
    uint tile = uint(params.tile);
    float share = rows > 0 ? 1.0 / (float(rows) * float(params.q_heads_total)) : 0.0;
    uint tiles = (uint(params.context) + tile - 1u) / tile;
    // Thread `k < tile` owns key `first + k` of the tile it walks; thread
    // `k * heads + h` dots head `h` against it.
    uint my_key = tid / heads;
    uint my_head = tid % heads;
    // The grid walks the tiles, its workgroups `PIE_MASS_GRID` tiles apart.
    for (uint t = group.x; t < tiles; t += uint(PIE_MASS_GRID)) {
        int first = int(t * tile);
        float acc = 0.0;
        if (first < capacity) {
            int live = min(int(tile), capacity - first);
            // Every kv head's row of the tile's keys, each key's rows one
            // contiguous run of the page.
            for (uint i = tid; i < uint(live) * kv_heads * pairs; i += uint(PIE_THREADS)) {
                uint k = i / (kv_heads * pairs);
                uint within = i % (kv_heads * pairs);
                int j = first + int(k);
                uint page = kv_page_indices[uint(page_first + j / params.page_size)];
                uint word_at = pie_unit_at(page, params.page_stride)
                    + ((uint(j % params.page_size) * kv_heads * head_dim) >> 1);
                mass_rows[k * key_words + (within / pairs) * run + within % pairs] =
                    pie_pool_get(params.k_origin + word_at + within);
            }
            for (int w = 0; w < rows; w++) {
                int qi = qo_hi - rows + w;
                uint q_at = uint(qi) * heads * pairs;
                for (uint i = tid; i < heads * pairs; i += uint(PIE_THREADS)) {
                    mass_q[(i / pairs) * run + i % pairs] = q[q_at + i];
                }
                for (uint h = tid; h < heads; h += uint(PIE_THREADS)) {
                    uint at = ((c * heads + h) * uint(params.observe) + uint(w)) * 2u;
                    mass_top[h] = stats[at];
                    mass_inv[h] = stats[at + 1u];
                }
                if (tid == 0u) {
                    mass_limit = min(position_ids[uint(qi)] + 1, capacity);
                }
                GroupMemoryBarrierWithGroupSync();
                int limit = workgroupUniformLoad(mass_limit);
                if (my_key < tile) {
                    int j = first + int(my_key);
                    float term = 0.0;
                    if (int(my_key) < live && j < limit) {
                        uint row_at = my_key * key_words + (my_head / groups) * run;
                        uint q_row = my_head * run;
                        float dot = 0.0;
                        for (uint p = 0u; p < pairs; p++) {
                            uint kw = mass_rows[row_at + p];
                            uint qw = mass_q[q_row + p];
                            dot = fma(bf16_to_f32(qw & 0xffffu), bf16_to_f32(kw & 0xffffu), dot);
                            dot = fma(bf16_to_f32(qw >> 16), bf16_to_f32(kw >> 16), dot);
                        }
                        term = exp(dot * params.sm_scale - mass_top[my_head]) * mass_inv[my_head];
                    }
                    mass_term[tid] = term;
                }
                GroupMemoryBarrierWithGroupSync();
                // A key's heads summed in head order.
                if (tid < tile) {
                    for (uint h = 0u; h < heads; h++) {
                        acc += mass_term[tid * heads + h];
                    }
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
        int j = first + int(tid);
        if (tid < tile && j < params.context) {
            uint dst = uint(keys_at + j);
            float prior = params.into != 0 ? pie_room_scores[dst] : 0.0;
            pie_room_scores[dst] = prior + acc * share;
        }
    }
}

// The grid's workgroups a capturing lane: the tiles past them are walked.
// pie:instantiate attn_score_mass_s16 PIE_HEAD_DIM_MAX=64 PIE_MASS_ROWS=2080 PIE_MASS_Q=1088 PIE_MASS_HEADS=64 PIE_MASS_GRID=1024
// pie:instantiate attn_score_mass_s32 PIE_HEAD_DIM_MAX=64 PIE_MASS_ROWS=5120 PIE_MASS_Q=2112 PIE_MASS_HEADS=64 PIE_MASS_GRID=1024

#endif
#endif
