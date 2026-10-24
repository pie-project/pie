#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// Multi-head latent attention, bf16 two a word throughout. The prologue:
// `PIE_LATENTS` splits each projected row into its RMS-normed latent and its
// rope tail; `PIE_SPLIT_Q` cuts q_b into its no-rope and rope parts per head;
// `PIE_KV_APPEND` writes a row's latent and rope tail into their pages;
// `PIE_ABSORB_Q` and `PIE_ABSORB_OUT` fold the up-projection into the query
// and out of the reading. Without a prologue define, the attention: one
// 32-lane workgroup a (head, row), causal over the request's pages or, with
// `PIE_SELECTED`, over the keys a selection row names; scores reduce through
// workgroup memory, no subgroup operation.

float2 word_pair(uint w) {
    return float2(bf16_to_f32(w & 0xffffu), bf16_to_f32(w >> 16));
}

#if defined(PIE_LATENTS)

[[vk::binding(0)]] StructuredBuffer<uint> kv_a;
[[vk::binding(1)]] StructuredBuffer<uint> norm_weight;
[[vk::binding(2)]] RWStructuredBuffer<uint> kv_c;
[[vk::binding(3)]] RWStructuredBuffer<uint> k_pe;

struct Params {
    int kv_lora;
    int rope;
    int src_row_stride;
    float eps;
};
PIE_PARAMS(4, Params, params);

// One workgroup a row.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.x;
    uint lid = local.x;
    uint base = row * uint(params.src_row_stride);
    uint kv_lora = uint(params.kv_lora);
    uint rope = uint(params.rope);
    uint span = 2u * uint(PIE_GROUP_X);
    for (uint d = 2u * lid; d < rope; d += span) {
        k_pe[(row * rope + d) >> 1] = kv_a[(base + kv_lora + d) >> 1];
    }
    float sq = 0.0;
    for (uint d = 2u * lid; d < kv_lora; d += span) {
        float2 v = word_pair(kv_a[(base + d) >> 1]);
        sq += dot(v, v);
    }
    float inv = pie_inv_rms(sq, lid, kv_lora, params.eps);
    for (uint d = 2u * lid; d < kv_lora; d += span) {
        float2 v = word_pair(kv_a[(base + d) >> 1]) * inv * word_pair(norm_weight[d >> 1]);
        kv_c[(row * kv_lora + d) >> 1] = pack_bf16(v.x, v.y);
    }
}

#elif defined(PIE_SPLIT_Q)

[[vk::binding(0)]] StructuredBuffer<uint> q_b;
[[vk::binding(1)]] RWStructuredBuffer<uint> q_nope;
[[vk::binding(2)]] RWStructuredBuffer<uint> q_pe;

struct Params {
    int total;
    int heads;
    int nope;
    int rope;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int i = int(gid.x) * 2;
    if (i >= params.total) {
        return;
    }
    int per = params.nope + params.rope;
    int d = i % per;
    int h = (i / per) % params.heads;
    int n = i / (params.heads * per);
    uint word = q_b[uint(i) >> 1];
    if (d < params.nope) {
        q_nope[uint((n * params.heads + h) * params.nope + d) >> 1] = word;
    } else {
        q_pe[uint((n * params.heads + h) * params.rope + d - params.nope) >> 1] = word;
    }
}

#elif defined(PIE_KV_APPEND)

[[vk::binding(0)]] StructuredBuffer<uint> kv_c;
[[vk::binding(1)]] StructuredBuffer<uint> k_pe;
[[vk::binding(2)]] RWStructuredBuffer<uint> ckv_pages;
[[vk::binding(3)]] RWStructuredBuffer<uint> kpe_pages;
[[vk::binding(4)]] StructuredBuffer<uint> w_page;
[[vk::binding(5)]] StructuredBuffer<uint> w_off;

struct Params {
    int page_size;
    int kv_lora;
    int rope;
    int rows;
};
PIE_PARAMS(6, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int d = int(gid.x) * 2;
    uint row = gid.y;
    if (row >= uint(params.rows)) {
        return;
    }
    int page = int(w_page[row]);
    int off = int(w_off[row]);
    if (page < 0 || off < 0 || off >= params.page_size) {
        return;
    }
    uint slot = uint(page) * uint(params.page_size) + uint(off);
    if (d < params.kv_lora) {
        ckv_pages[(slot * uint(params.kv_lora) + uint(d)) >> 1] = kv_c[(row * uint(params.kv_lora) + uint(d)) >> 1];
    }
    if (d < params.rope) {
        kpe_pages[(slot * uint(params.rope) + uint(d)) >> 1] = k_pe[(row * uint(params.rope) + uint(d)) >> 1];
    }
}

#elif defined(PIE_ABSORB_Q)

[[vk::binding(0)]] StructuredBuffer<uint> q_nope;
[[vk::binding(1)]] StructuredBuffer<uint> kv_b;
[[vk::binding(2)]] RWStructuredBuffer<uint> q_latent;

struct Params {
    int heads;
    int rank;
    int nope;
    int v_dim;
};
PIE_PARAMS(3, Params, params);

// `q_latent[t, h, i] = sum_j q_nope[t, h, j] * kv_b[h, j, i]`, two latent
// columns a lane; the grid is (latent words, heads, rows).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = 2u * gid.x;
    uint h = gid.y;
    uint t = gid.z;
    uint rank = uint(params.rank);
    if (i >= rank) {
        return;
    }
    uint qn_base = (t * uint(params.heads) + h) * uint(params.nope);
    uint kb_base = h * uint(params.nope + params.v_dim) * rank;
    float2 acc = float2(0.0, 0.0);
    for (uint j = 0u; j < uint(params.nope); j += 2u) {
        float2 qv = word_pair(q_nope[(qn_base + j) >> 1]);
        acc += qv.x * word_pair(kv_b[(kb_base + j * rank + i) >> 1])
            + qv.y * word_pair(kv_b[(kb_base + (j + 1u) * rank + i) >> 1]);
    }
    q_latent[((t * uint(params.heads) + h) * rank + i) >> 1] = pack_bf16(acc.x, acc.y);
}

#elif defined(PIE_ABSORB_OUT)

[[vk::binding(0)]] StructuredBuffer<uint> latent;
[[vk::binding(1)]] StructuredBuffer<uint> kv_b;
[[vk::binding(2)]] RWStructuredBuffer<uint> o;

struct Params {
    int heads;
    int rank;
    int v_dim;
    int nope;
};
PIE_PARAMS(3, Params, params);

// `o[t, h, j] = sum_i latent[t, h, i] * kv_b[h, nope + j, i]`, two value
// columns a lane; the grid is (value words, heads, rows).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint j = 2u * gid.x;
    uint h = gid.y;
    uint t = gid.z;
    if (j >= uint(params.v_dim)) {
        return;
    }
    uint rank = uint(params.rank);
    uint lat_base = (t * uint(params.heads) + h) * rank;
    uint row0 = h * uint(params.nope + params.v_dim) * rank + (uint(params.nope) + j) * rank;
    uint row1 = row0 + rank;
    float2 acc = float2(0.0, 0.0);
    for (uint i = 0u; i < rank; i += 2u) {
        float2 l = word_pair(latent[(lat_base + i) >> 1]);
        acc += float2(dot(l, word_pair(kv_b[(row0 + i) >> 1])), dot(l, word_pair(kv_b[(row1 + i) >> 1])));
    }
    o[((t * uint(params.heads) + h) * uint(params.v_dim) + j) >> 1] = pack_bf16(acc.x, acc.y);
}

#else

#define MLA_LANES 32
#define MAX_CKV_PAIRS 8
#define MAX_KPE_PAIRS 2

[[vk::binding(0)]] StructuredBuffer<uint> q_nope;
[[vk::binding(1)]] StructuredBuffer<uint> q_pe;
[[vk::binding(2)]] StructuredBuffer<uint> ckv_pages;
[[vk::binding(3)]] StructuredBuffer<uint> kpe_pages;
[[vk::binding(4)]] RWStructuredBuffer<uint> o;
[[vk::binding(5)]] StructuredBuffer<int> position_ids;
[[vk::binding(6)]] StructuredBuffer<int> req_of_token;
[[vk::binding(7)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(8)]] StructuredBuffer<uint> kv_page_indptr;
#if defined(PIE_SELECTED)
[[vk::binding(9)]] StructuredBuffer<int> selection;
#define PIE_PARAMS_AT 10
#else
#define PIE_PARAMS_AT 9
#endif

struct Params {
    int page_size;
    int heads;
    int ckv;
    int kpe;
    float sm_scale;
#if defined(PIE_SELECTED)
    int top_k;
#endif
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

groupshared float mla_fold[MLA_LANES];
groupshared int mla_steps;

float lane_sum(float v, uint lane) {
    mla_fold[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_ = uint(MLA_LANES) >> 1; half_ > 0u; half_ >>= 1) {
        if (lane < half_) {
            mla_fold[lane] += mla_fold[lane + half_];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float total = mla_fold[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// One 32-lane workgroup a (head, row); a lane holds latent pairs
// `lane + p * 32`.
[shader("compute")]
[numthreads(MLA_LANES, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint h = group.x;
    uint row = group.y;
    uint lane = local.x;
    uint ckv = uint(params.ckv);
    uint kpe = uint(params.kpe);
    uint pairs = ckv / (2u * uint(MLA_LANES));
    uint ppairs = kpe / (2u * uint(MLA_LANES));
    uint heads = uint(params.heads);

    int r = req_of_token[row];
    int j_end = position_ids[row] + 1;
    uint page_base = kv_page_indptr[uint(r)];

    uint qn_base = ((row * heads + h) * ckv) >> 1;
    uint qp_base = ((row * heads + h) * kpe) >> 1;
    float2 qn[MAX_CKV_PAIRS];
    float2 qp[MAX_KPE_PAIRS];
    for (uint p = 0u; p < uint(MAX_CKV_PAIRS); p++) {
        qn[p] = p < pairs ? word_pair(q_nope[qn_base + lane + p * uint(MLA_LANES)]) : float2(0.0, 0.0);
    }
    for (uint p = 0u; p < uint(MAX_KPE_PAIRS); p++) {
        qp[p] = p < ppairs ? word_pair(q_pe[qp_base + lane + p * uint(MLA_LANES)]) : float2(0.0, 0.0);
    }

    float2 acc[MAX_CKV_PAIRS];
    for (uint p = 0u; p < uint(MAX_CKV_PAIRS); p++) {
        acc[p] = float2(0.0, 0.0);
    }
    float m = -3.0e38;
    float lsum = 0.0;

    if (lane == 0u) {
#if defined(PIE_SELECTED)
        mla_steps = params.top_k;
#else
        mla_steps = j_end;
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: it bounds the loop of barriers below.
    int steps = workgroupUniformLoad(mla_steps);
#if defined(PIE_SELECTED)
    uint srow = row * uint(params.top_k);
#endif
    for (int n = 0; n < steps; n++) {
        int j = n;
        bool live = true;
#if defined(PIE_SELECTED)
        j = selection[srow + uint(n)];
        live = j >= 0 && j < j_end;
#endif
        float2 kv[MAX_CKV_PAIRS];
        float pd = 0.0;
        for (uint p = 0u; p < uint(MAX_CKV_PAIRS); p++) {
            kv[p] = float2(0.0, 0.0);
        }
        if (live) {
            uint page = kv_page_indices[page_base + uint(j / params.page_size)];
            uint slot = page * uint(params.page_size) + uint(j % params.page_size);
            uint ckv_j = (slot * ckv) >> 1;
            uint kpe_j = (slot * kpe) >> 1;
            for (uint p = 0u; p < uint(MAX_CKV_PAIRS); p++) {
                if (p < pairs) {
                    kv[p] = word_pair(ckv_pages[ckv_j + lane + p * uint(MLA_LANES)]);
                    pd += dot(qn[p], kv[p]);
                }
            }
            for (uint p = 0u; p < uint(MAX_KPE_PAIRS); p++) {
                if (p < ppairs) {
                    pd += dot(qp[p], word_pair(kpe_pages[kpe_j + lane + p * uint(MLA_LANES)]));
                }
            }
        }
        pd = lane_sum(pd, lane);
        if (live) {
            float score = pd * params.sm_scale;
            float m_new = max(m, score);
            float corr = exp(m - m_new);
            float w = exp(score - m_new);
            lsum = lsum * corr + w;
            for (uint p = 0u; p < uint(MAX_CKV_PAIRS); p++) {
                acc[p] = acc[p] * corr + w * kv[p];
            }
            m = m_new;
        }
    }

    float inv = lsum > 0.0 ? 1.0 / lsum : 0.0;
    uint o_base = ((row * heads + h) * ckv) >> 1;
    for (uint p = 0u; p < pairs; p++) {
        o[o_base + lane + p * uint(MLA_LANES)] = pack_bf16(acc[p].x * inv, acc[p].y * inv);
    }
}

#endif

// pie:instantiate attn_mla_latents_bf16 PIE_LATENTS=1 PIE_GROUP_X=256
// pie:instantiate attn_mla_split_q_b_bf16 PIE_SPLIT_Q=1 PIE_GROUP_X=256
// pie:instantiate attn_mla_kv_append_bf16 PIE_KV_APPEND=1 PIE_GROUP_X=256
// pie:instantiate attn_mla_absorb_q_bf16 PIE_ABSORB_Q=1 PIE_GROUP_X=64
// pie:instantiate attn_mla_absorb_out_bf16 PIE_ABSORB_OUT=1 PIE_GROUP_X=64
// pie:instantiate attn_mla_paged_bf16 PIE_GROUP_X=32
// pie:instantiate attn_mla_paged_selected_bf16 PIE_SELECTED=1 PIE_GROUP_X=32
