#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// The compressed KV pool, bf16 two a word. A row closes a block when its
// position + 1 is a multiple of `ratio`; `PIE_BOUNDARY` lands each row's
// block position (or -1), its rope position and its request.
// `PIE_STATE_WRITE` files a row's kv and gate-score planes into its state
// pages; `PIE_GATHER` pools each closed block, a softmax over the gate scores
// of its window (`coff` blocks back, an optional learned position bias)
// weighting the state kv; `PIE_BLOCK_MEAN` averages a block's index keys;
// `PIE_STORE` files the pooled entries into the compressed pages. Without a
// kernel define, the attention over pooled entries, causal or over a
// selection (`PIE_SELECTED`), landing the reading and its base-2 LSE; one
// workgroup a (row, head), reductions through workgroup memory.

float neg_inf() {
    return asfloat(0xff800000u);
}

#if defined(PIE_BOUNDARY)

[[vk::binding(0)]] StructuredBuffer<int> positions;
[[vk::binding(1)]] RWStructuredBuffer<int> out_pos;
[[vk::binding(2)]] RWStructuredBuffer<int> out_req;
[[vk::binding(3)]] RWStructuredBuffer<int> out_rope;
[[vk::binding(4)]] StructuredBuffer<uint> row_valid;
[[vk::binding(5)]] StructuredBuffer<int> req_of_token;

struct Params {
    int n;
    int ratio;
};
PIE_PARAMS(6, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int t = int(gid.x);
    if (t >= params.n) {
        return;
    }
    int p = positions[uint(t)];
    bool valid = ((row_valid[uint(t) >> 2] >> ((uint(t) & 3u) * 8u)) & 0xffu) != 0u;
    bool boundary = valid && ((p + 1) % params.ratio) == 0;
    out_pos[uint(t)] = boundary ? p : -1;
    out_rope[uint(t)] = boundary ? (p / params.ratio) * params.ratio : 0;
    // The fire's request: the pages a block files into are the fire's.
    out_req[uint(t)] = req_of_token[uint(t)];
}

#elif defined(PIE_STATE_WRITE)

[[vk::binding(0)]] StructuredBuffer<uint> kv;
[[vk::binding(1)]] StructuredBuffer<uint> score;
[[vk::binding(2)]] RWStructuredBuffer<uint> state_kv;
[[vk::binding(3)]] RWStructuredBuffer<uint> state_score;
[[vk::binding(4)]] StructuredBuffer<uint> w_page;
[[vk::binding(5)]] StructuredBuffer<uint> w_off;

struct Params {
    int width;
    int page_size;
    int state_pitch;
    int rows;
};
PIE_PARAMS(6, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int d = int(gid.x) * 2;
    uint i = gid.y;
    if (d >= params.width || i >= uint(params.rows)) {
        return;
    }
    int page = int(w_page[i]);
    int off = int(w_off[i]);
    if (page < 0 || off < 0 || off >= params.page_size) {
        return;
    }
    uint slot = uint(page) * uint(params.page_size) + uint(off);
    uint dst = (slot * uint(params.state_pitch) + uint(d)) >> 1;
    uint src = (i * uint(params.width) + uint(d)) >> 1;
    state_kv[dst] = kv[src];
    state_score[dst] = score[src];
}

#elif defined(PIE_BLOCK_MEAN) || defined(PIE_GATHER) || defined(PIE_STORE)

#if defined(PIE_BLOCK_MEAN)
[[vk::binding(0)]] StructuredBuffer<uint> key_pages;
[[vk::binding(1)]] RWStructuredBuffer<uint> o;
[[vk::binding(2)]] StructuredBuffer<int> boundary_pos;
[[vk::binding(3)]] StructuredBuffer<int> boundary_req;
[[vk::binding(4)]] StructuredBuffer<uint> page_indices;
[[vk::binding(5)]] StructuredBuffer<uint> page_indptr;

struct Params {
    int head_dim;
    int ratio;
    int page_size;
    int rows;
};
PIE_PARAMS(6, Params, params);
#elif defined(PIE_GATHER)
[[vk::binding(0)]] StructuredBuffer<uint> state_kv;
[[vk::binding(1)]] StructuredBuffer<uint> state_score;
[[vk::binding(2)]] StructuredBuffer<float> ape;
[[vk::binding(3)]] StructuredBuffer<int> boundary_pos;
[[vk::binding(4)]] StructuredBuffer<int> boundary_req;
[[vk::binding(5)]] StructuredBuffer<uint> page_indices;
[[vk::binding(6)]] StructuredBuffer<uint> page_indptr;
[[vk::binding(7)]] RWStructuredBuffer<uint> o;

struct Params {
    int head_dim;
    int ratio;
    int coff;
    int page_size;
    int has_ape;
    int state_pitch;
    int rows;
};
PIE_PARAMS(8, Params, params);
#else
[[vk::binding(0)]] StructuredBuffer<uint> entries;
[[vk::binding(1)]] RWStructuredBuffer<uint> comp_kv_pages;
[[vk::binding(2)]] StructuredBuffer<int> boundary_pos;
[[vk::binding(3)]] StructuredBuffer<int> boundary_req;
[[vk::binding(4)]] StructuredBuffer<uint> page_indices;
[[vk::binding(5)]] StructuredBuffer<uint> page_indptr;

struct Params {
    int head_dim;
    int page_size;
    int rows;
};
PIE_PARAMS(6, Params, params);
#endif

// The page slot of position `pos` of request `req`.
uint pool_slot(int req, int pos) {
    uint page = page_indices[page_indptr[uint(req)] + uint(pos / params.page_size)];
    return page * uint(params.page_size) + uint(pos % params.page_size);
}

#if defined(PIE_BLOCK_MEAN)
float block_value(int d, int bpos, int req) {
    float acc = 0.0;
    for (int i = 0; i < params.ratio; i++) {
        int pos = bpos + i - (params.ratio - 1);
        if (pos >= 0) {
            acc += bf16_at(key_pages, pool_slot(req, pos) * uint(params.head_dim) + uint(d));
        }
    }
    return acc / float(params.ratio);
}
#elif defined(PIE_GATHER)
// A closed block pooled: a softmax over the gate scores of its window,
// `coff` blocks back (the earlier blocks' columns `head_dim` along), plus the
// learned position bias when there is one, weighting the state kv.
float block_value(int d, int bpos, int req) {
    int window = params.coff * params.ratio;
    uint width = uint(params.coff * params.head_dim);
    uint pitch = uint(params.state_pitch);
    float max_s = neg_inf();
    for (int i = 0; i < window; i++) {
        int pos = bpos + i - (window - 1);
        if (pos >= 0) {
            uint col = uint((i >= params.ratio ? params.head_dim : 0) + d);
            uint e = pool_slot(req, pos) * pitch + col;
            float sc = bf16_at(state_score, e);
            if (params.has_ape != 0) {
                sc += ape[uint(pos % params.ratio) * width + col];
            }
            max_s = max(max_s, sc);
        }
    }
    if (!(max_s > -3.0e38)) {
        return 0.0;
    }
    float sum_e = 0.0;
    float acc = 0.0;
    for (int i = 0; i < window; i++) {
        int pos = bpos + i - (window - 1);
        if (pos >= 0) {
            uint col = uint((i >= params.ratio ? params.head_dim : 0) + d);
            uint e = pool_slot(req, pos) * pitch + col;
            float sc = bf16_at(state_score, e);
            if (params.has_ape != 0) {
                sc += ape[uint(pos % params.ratio) * width + col];
            }
            float ex = exp(sc - max_s);
            sum_e += ex;
            acc += ex * bf16_at(state_kv, e);
        }
    }
    return sum_e > 0.0 ? acc / sum_e : 0.0;
}
#endif

// Two head columns a lane; the grid is (head words, blocks).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int d = int(gid.x) * 2;
    uint c = gid.y;
    if (d >= params.head_dim || c >= uint(params.rows)) {
        return;
    }
    int bpos = boundary_pos[c];
    int req = boundary_req[c];
    uint at = (c * uint(params.head_dim) + uint(d)) >> 1;
#if defined(PIE_STORE)
    if (bpos >= 0) {
        comp_kv_pages[(pool_slot(req, bpos) * uint(params.head_dim) + uint(d)) >> 1] = entries[at];
    }
#else
    o[at] = bpos < 0 ? 0u : pack_bf16(block_value(d, bpos, req), block_value(d + 1, bpos, req));
#endif
}

#else

#define POOL_HEAD_MAX 512
#define PAIRS_PER_THREAD (POOL_HEAD_MAX / (2 * PIE_GROUP_X))

[[vk::binding(0)]] StructuredBuffer<uint> q;
[[vk::binding(1)]] StructuredBuffer<uint> comp_kv_pages;
#if defined(PIE_SELECTED)
[[vk::binding(2)]] StructuredBuffer<int> selection;
#define PIE_AFTER 3
#else
#define PIE_AFTER 2
#endif
[[vk::binding(PIE_AFTER)]] RWStructuredBuffer<uint> o;
[[vk::binding(PIE_AFTER + 1)]] RWStructuredBuffer<float> lse_out;
[[vk::binding(PIE_AFTER + 2)]] StructuredBuffer<int> positions;
[[vk::binding(PIE_AFTER + 3)]] StructuredBuffer<uint> page_indices;
[[vk::binding(PIE_AFTER + 4)]] StructuredBuffer<uint> page_indptr;
[[vk::binding(PIE_AFTER + 5)]] StructuredBuffer<int> req_of_token;

struct Params {
    int num_q_heads;
    int head_dim;
    int ratio;
#if defined(PIE_SELECTED)
    int top_k;
#endif
    int page_size;
    float scale;
};
PIE_PARAMS(PIE_AFTER + 6, Params, params);

groupshared float pool_q[POOL_HEAD_MAX];
groupshared int pool_visible;

uint pool_slot(int req, int pos) {
    uint page = page_indices[page_indptr[uint(req)] + uint(pos / params.page_size)];
    return page * uint(params.page_size) + uint(pos % params.page_size);
}

// The pooled entry step `n` reads, or -1.
int key_of(int n, uint qi, int num_visible) {
#if defined(PIE_SELECTED)
    int c = selection[qi * uint(params.top_k) + uint(n)];
    return c < 0 || c >= num_visible ? -1 : c;
#else
    return n;
#endif
}

// The grid is (lanes, rows, heads): one workgroup a (row, head).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint qi = group.y;
    uint q_head = group.z;
    int tid = int(local.x);
    uint lanes = uint(PIE_GROUP_X);
    uint head_dim = uint(params.head_dim);
    int req = req_of_token[qi];
    if (tid == 0) {
        pool_visible = (positions[qi] + 1) / params.ratio;
    }
    uint q_row = (qi * uint(params.num_q_heads) + q_head) * head_dim;
    for (uint d = 2u * uint(tid); d < head_dim; d += 2u * lanes) {
        uint word = q[(q_row + d) >> 1];
        pool_q[d] = bf16_to_f32(word & 0xffffu);
        pool_q[d + 1u] = bf16_to_f32(word >> 16);
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: it bounds the loop of barriers below.
    int num_visible = workgroupUniformLoad(pool_visible);
#if defined(PIE_SELECTED)
    int steps = num_visible <= 0 ? 0 : params.top_k;
#else
    int steps = max(num_visible, 0);
#endif

    // The row's maximum score, a key a lane.
    float local_max = neg_inf();
    for (int n = tid; n < steps; n += PIE_GROUP_X) {
        int c = key_of(n, qi, num_visible);
        if (c >= 0) {
            uint k_row = pool_slot(req, (c + 1) * params.ratio - 1) * head_dim;
            float dot_ = 0.0;
            for (uint d = 0u; d < head_dim; d += 2u) {
                uint word = comp_kv_pages[(k_row + d) >> 1];
                dot_ += pool_q[d] * bf16_to_f32(word & 0xffffu) + pool_q[d + 1u] * bf16_to_f32(word >> 16);
            }
            local_max = max(local_max, dot_ * params.scale);
        }
    }
    float row_max = pie_workgroup_max(local_max, uint(tid));
    bool dead = !(row_max > -3.0e38);

    // The reading, a key at a time across the workgroup.
    float acc[2 * PAIRS_PER_THREAD];
    for (uint i = 0u; i < uint(2 * PAIRS_PER_THREAD); i++) {
        acc[i] = 0.0;
    }
    float local_z = 0.0;
    for (int n = 0; n < steps; n++) {
        int c = key_of(n, qi, num_visible);
        bool live = c >= 0;
        uint k_row = 0u;
        float dot_ = 0.0;
        if (live) {
            k_row = pool_slot(req, (c + 1) * params.ratio - 1) * head_dim;
            for (uint d = 2u * uint(tid); d < head_dim; d += 2u * lanes) {
                uint word = comp_kv_pages[(k_row + d) >> 1];
                dot_ += pool_q[d] * bf16_to_f32(word & 0xffffu) + pool_q[d + 1u] * bf16_to_f32(word >> 16);
            }
        }
        float s = pie_workgroup_sum(dot_, uint(tid));
        if (live) {
            float w = exp(s * params.scale - row_max);
            local_z += w;
            for (uint i = 0u; i < uint(PAIRS_PER_THREAD); i++) {
                uint d = (uint(tid) + i * lanes) * 2u;
                if (d < head_dim) {
                    uint word = comp_kv_pages[(k_row + d) >> 1];
                    acc[2u * i] += w * bf16_to_f32(word & 0xffffu);
                    acc[2u * i + 1u] += w * bf16_to_f32(word >> 16);
                }
            }
        }
    }
    bool empty = !(local_z > 0.0) || dead || num_visible <= 0;
    float inv_z = empty ? 0.0 : 1.0 / local_z;
    if (tid == 0) {
        lse_out[qi * uint(params.num_q_heads) + q_head] = empty ? neg_inf() : (log(local_z) + row_max) * 1.44269504088896340736;
    }
    for (uint i = 0u; i < uint(PAIRS_PER_THREAD); i++) {
        uint d = (uint(tid) + i * lanes) * 2u;
        if (d < head_dim) {
            o[(q_row + d) >> 1] = pack_bf16(acc[2u * i] * inv_z, acc[2u * i + 1u] * inv_z);
        }
    }
}

#endif

// pie:instantiate attn_pool_boundary PIE_BOUNDARY=1 PIE_GROUP_X=128
// pie:instantiate attn_pool_state_write_bf16 PIE_STATE_WRITE=1 PIE_GROUP_X=256
// pie:instantiate attn_pool_gather_bf16 PIE_GATHER=1 PIE_GROUP_X=256
// pie:instantiate attn_pool_block_mean_bf16 PIE_BLOCK_MEAN=1 PIE_GROUP_X=256
// pie:instantiate attn_pool_store_bf16 PIE_STORE=1 PIE_GROUP_X=256
// pie:instantiate attn_pool_lse_bf16 PIE_GROUP_X=128
// pie:instantiate attn_pool_lse_selected_bf16 PIE_SELECTED=1 PIE_GROUP_X=128
