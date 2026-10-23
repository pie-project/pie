#include "prim/barrier.slang"
#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// The sparse-attention indexer, bf16 two a word. `PIE_KNORM`: each index
// key row layer-normed in place, then its rotated prefix turned by its
// position (interleaved pairs). `PIE_Q_ROPE`: each index query head's
// rotated prefix turned. Without either, the top-k: each query row scores
// every `ratio`-th cached key of its request, `sum_h w_h relu(q_h . k)`, into
// an f32 slab, then selects the `topk` best by bisecting a threshold over
// the scores (40 halvings) and lands their indices, earliest first, -1
// past the end. Reductions run through workgroup memory.

// Pair `i` of a rotated prefix turned by `pos * theta^(-2i / rope_dim)`.
uint rope_word(uint word, int i, int rope_dim, int pos, float theta) {
    float ang = float(pos) * pow(theta, -2.0 * float(i) / float(rope_dim));
    float c = cos(ang);
    float s = sin(ang);
    float a = bf16_to_f32(word & 0xffffu);
    float b = bf16_to_f32(word >> 16);
    return pack_bf16(a * c - b * s, b * c + a * s);
}

float2 word_pair(uint w) {
    return float2(bf16_to_f32(w & 0xffffu), bf16_to_f32(w >> 16));
}

#if defined(PIE_KNORM)

[[vk::binding(0)]] RWStructuredBuffer<uint> idx_k;
[[vk::binding(1)]] StructuredBuffer<uint> w;
[[vk::binding(2)]] StructuredBuffer<uint> b;
[[vk::binding(3)]] StructuredBuffer<int> positions;

struct Params {
    int head_dim;
    int rope_dim;
    float theta;
    float eps;
};
PIE_PARAMS(4, Params, params);

// One workgroup a key row.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = group.x;
    uint tid = local.x;
    uint head_dim = uint(params.head_dim);
    uint base = n * head_dim;
    uint span = 2u * uint(PIE_GROUP_X);

    float s = 0.0;
    for (uint d = 2u * tid; d < head_dim; d += span) {
        float2 v = word_pair(idx_k[(base + d) >> 1]);
        s += v.x + v.y;
    }
    float mean = pie_workgroup_sum(s, tid) / float(head_dim);
    float vv = 0.0;
    for (uint d = 2u * tid; d < head_dim; d += span) {
        float2 v = word_pair(idx_k[(base + d) >> 1]) - mean;
        vv += dot(v, v);
    }
    float inv = rsqrt(pie_workgroup_sum(vv, tid) / float(head_dim) + params.eps);
    for (uint d = 2u * tid; d < head_dim; d += span) {
        uint at = (base + d) >> 1;
        float2 v = (word_pair(idx_k[at]) - mean) * inv * word_pair(w[d >> 1]) + word_pair(b[d >> 1]);
        idx_k[at] = pack_bf16(v.x, v.y);
    }
    // The normed row is in device memory; its rotated words are read back
    // by other lanes.
    pie_storage_barrier();

    int pos = positions[n];
    for (int i = int(tid); i < params.rope_dim / 2; i += PIE_GROUP_X) {
        uint at = (base >> 1) + uint(i);
        idx_k[at] = rope_word(idx_k[at], i, params.rope_dim, pos, params.theta);
    }
}

#elif defined(PIE_Q_ROPE)

[[vk::binding(0)]] RWStructuredBuffer<uint> idx_q;
[[vk::binding(1)]] StructuredBuffer<int> positions;

struct Params {
    int n_heads;
    int head_dim;
    int rope_dim;
    float theta;
};
PIE_PARAMS(2, Params, params);

// A head a lane; the grid is (heads, rows).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h = int(gid.x);
    if (h >= params.n_heads) {
        return;
    }
    uint n = gid.y;
    uint base = ((n * uint(params.n_heads) + uint(h)) * uint(params.head_dim)) >> 1;
    int pos = positions[n];
    for (int i = 0; i < params.rope_dim / 2; i++) {
        uint at = base + uint(i);
        idx_q[at] = rope_word(idx_q[at], i, params.rope_dim, pos, params.theta);
    }
}

#else

#define LANES_PER_KEY 64
#define KEYS_PER_PASS (PIE_GROUP_X / LANES_PER_KEY)

[[vk::binding(0)]] StructuredBuffer<uint> idx_q;
[[vk::binding(1)]] StructuredBuffer<uint> idx_w;
[[vk::binding(2)]] StructuredBuffer<uint> key_pages;
[[vk::binding(3)]] StructuredBuffer<int> positions;
[[vk::binding(4)]] StructuredBuffer<int> req_of_token;
[[vk::binding(5)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(6)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(7)]] RWStructuredBuffer<float> scores;
[[vk::binding(8)]] RWStructuredBuffer<int> selection;

struct Params {
    int heads;
    int head_dim;
    int page_size;
    int score_stride;
    int topk;
    int ratio;
    int has_weights;
};
PIE_PARAMS(9, Params, params);

groupshared float key_acc[PIE_GROUP_X];
groupshared int key_total;

// One workgroup a query row.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint t = group.x;
    int tid = int(local.x);
    uint srow = t * uint(params.topk);
    int topk = params.topk;
    uint pages_first = kv_page_indptr[uint(req_of_token[t])];
    int stride = params.ratio > 0 ? params.ratio : 1;

    if (tid == 0) {
        key_total = positions[t] + 1;
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: it bounds the loops of barriers below.
    int total = workgroupUniformLoad(key_total);
    int nkeys = min(total > 0 ? total / stride : 0, params.score_stride);

    uint frow = t * uint(params.score_stride);
    uint heads = uint(params.heads);
    uint hd = uint(params.head_dim);
    uint qi = t * heads * hd;
    uint wi = t * heads;
    int slot = tid / LANES_PER_KEY;
    int hh = tid % LANES_PER_KEY;
    for (int j0 = 0; j0 < nkeys; j0 += KEYS_PER_PASS) {
        int j = j0 + slot;
        float acc = 0.0;
        if (j < nkeys) {
            int cell = (j + 1) * stride - 1;
            uint page = kv_page_indices[pages_first + uint(cell / params.page_size)];
            uint kj = (page * uint(params.page_size) + uint(cell % params.page_size)) * hd;
            for (uint h = uint(hh); h < heads; h += uint(LANES_PER_KEY)) {
                uint qh = qi + h * hd;
                float d_ = 0.0;
                for (uint d = 0u; d < hd; d += 2u) {
                    d_ += dot(word_pair(idx_q[(qh + d) >> 1]), word_pair(key_pages[(kj + d) >> 1]));
                }
                float weight = params.has_weights != 0 ? bf16_at(idx_w, wi + h) : 1.0;
                acc += max(d_, 0.0) * weight;
            }
        }
        key_acc[tid] = acc;
        GroupMemoryBarrierWithGroupSync();
        if (tid < KEYS_PER_PASS && j0 + tid < nkeys) {
            float sum = 0.0;
            for (int g = 0; g < LANES_PER_KEY; g++) {
                sum += key_acc[tid * LANES_PER_KEY + g];
            }
            scores[frow + uint(j0 + tid)] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // The scores are in device memory, read back by every lane.
    pie_storage_barrier();

    if (nkeys <= topk) {
        for (int n = tid; n < topk; n += PIE_GROUP_X) {
            selection[srow + uint(n)] = n < nkeys ? n : -1;
        }
        return;
    }

    float lo_l = 3.0e38;
    float hi_l = -3.0e38;
    for (int j = tid; j < nkeys; j += PIE_GROUP_X) {
        float s = scores[frow + uint(j)];
        lo_l = min(lo_l, s);
        hi_l = max(hi_l, s);
    }
    float lo = -pie_workgroup_max(-lo_l, uint(tid));
    float hi = pie_workgroup_max(hi_l, uint(tid));
    float thr = hi;
    for (int it = 0; it < 40; it++) {
        float mid = 0.5 * (lo + hi);
        float c = 0.0;
        for (int j = tid; j < nkeys; j += PIE_GROUP_X) {
            if (scores[frow + uint(j)] >= mid) {
                c += 1.0;
            }
        }
        if (int(pie_workgroup_sum(c, uint(tid))) > topk) {
            lo = mid;
        } else {
            hi = mid;
        }
        thr = hi;
    }

    if (tid == 0) {
        int n = 0;
        for (int j = 0; j < nkeys && n < topk; j++) {
            if (scores[frow + uint(j)] >= thr) {
                selection[srow + uint(n)] = j;
                n++;
            }
        }
        for (; n < topk; n++) {
            selection[srow + uint(n)] = -1;
        }
    }
}

#endif

// pie:instantiate attn_index_knorm_rope_bf16 PIE_KNORM=1 PIE_GROUP_X=256
// pie:instantiate attn_index_q_rope_bf16 PIE_Q_ROPE=1 PIE_GROUP_X=32
// pie:instantiate attn_index_topk_bf16 PIE_GROUP_X=256
