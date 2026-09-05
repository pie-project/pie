#include "common/bf16.slang"
#include "common/reduce.slang"

#if defined(PIE_MLA_LATENTS)

#define PIE_GROUP_X 256
PIE_BUFFER_RO(0, kv_a);
PIE_BUFFER_RO(1, norm_weight);
PIE_BUFFER_RW(2, kv_c);
PIE_BUFFER_RW(3, k_pe);
struct Push { int kv_lora; int rope; int src_row_stride; float eps; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.x;
    uint lid = local.x;
    uint base = row * uint(pc.src_row_stride);
    uint kv_lora = uint(pc.kv_lora);
    uint rope = uint(pc.rope);
    for (uint d = lid; d < rope; d += PIE_GROUP_X) {
        k_pe[row * rope + d] = kv_a[base + kv_lora + d];
    }
    float sq = 0.0;
    for (uint d = lid; d < kv_lora; d += PIE_GROUP_X) {
        float v = PIE_LOAD(kv_a[base + d]);
        sq += v * v;
    }
    float inv = pie_inv_rms(sq, kv_lora, pc.eps);
    for (uint d = lid; d < kv_lora; d += PIE_GROUP_X) {
        float v = PIE_LOAD(kv_a[base + d]);
        float w = PIE_LOAD(norm_weight[d]);
        kv_c[row * kv_lora + d] = PIE_STORE(v * inv * w);
    }
}

#elif defined(PIE_MLA_SPLIT_Q)

#define PIE_GROUP_X 256
PIE_BUFFER_RO(0, q_b);
PIE_BUFFER_RW(1, q_nope);
PIE_BUFFER_RW(2, q_pe);
struct Push { int total; int heads; int nope; int rope; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int i = int(gid.x);
    if (i >= pc.total) return;
    int per = pc.nope + pc.rope;
    int d = i % per;
    int h = (i / per) % pc.heads;
    int n = i / (pc.heads * per);
    PIE_ACT v = q_b[uint(i)];
    if (d < pc.nope) {
        q_nope[uint((n * pc.heads + h) * pc.nope + d)] = v;
    } else {
        q_pe[uint((n * pc.heads + h) * pc.rope + (d - pc.nope))] = v;
    }
}

#elif defined(PIE_MLA_KV_APPEND)

#define PIE_GROUP_X 256
PIE_BUFFER_RO(0, kv_c);
PIE_BUFFER_RO(1, k_pe);
PIE_BUFFER_RW(2, ckv_pages);
PIE_BUFFER_RW(3, kpe_pages);
[[vk::binding(4)]] StructuredBuffer<uint> w_page;
[[vk::binding(5)]] StructuredBuffer<uint> w_off;
struct Push { int page_size; int kv_lora; int rope; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int d = int(gid.x);
    uint row = gid.y;
    if (row >= uint(pc.rows)) return;

    int page = int(w_page[row]);
    int off = int(w_off[row]);
    if (page < 0 || off < 0 || off >= pc.page_size) return;
    uint slot = uint(page) * uint(pc.page_size) + uint(off);
    if (d < pc.kv_lora) {
        ckv_pages[slot * uint(pc.kv_lora) + uint(d)] = kv_c[row * uint(pc.kv_lora) + uint(d)];
    }
    if (d < pc.rope) {
        kpe_pages[slot * uint(pc.rope) + uint(d)] = k_pe[row * uint(pc.rope) + uint(d)];
    }
}

#elif defined(PIE_MLA_ABSORB_Q)

#define PIE_GROUP_X 64
PIE_BUFFER_RO(0, q_nope);
PIE_BUFFER_RO(1, kv_b);
PIE_BUFFER_RW(2, q_latent);
struct Push { int heads; int rank; int nope; int v_dim; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int i = int(gid.x);
    int h = int(gid.y);
    int t = int(gid.z);
    if (i >= pc.rank) return;
    uint qn_base = uint((t * pc.heads + h) * pc.nope);
    uint kb_base = uint(h) * uint(pc.nope + pc.v_dim) * uint(pc.rank);
    float acc = 0.0;
    for (int j = 0; j < pc.nope; j++) {
        acc += PIE_LOAD(q_nope[qn_base + uint(j)]) * PIE_LOAD(kv_b[kb_base + uint(j) * uint(pc.rank) + uint(i)]);
    }
    q_latent[uint((t * pc.heads + h) * pc.rank + i)] = PIE_STORE(acc);
}

#elif defined(PIE_MLA_ABSORB_OUT)

#define PIE_GROUP_X 64
PIE_BUFFER_RO(0, latent);
PIE_BUFFER_RO(1, kv_b);
PIE_BUFFER_RW(2, out_);
struct Push { int heads; int rank; int v_dim; int nope; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int j = int(gid.x);
    int h = int(gid.y);
    int t = int(gid.z);
    if (j >= pc.v_dim) return;
    uint lat_base = uint((t * pc.heads + h) * pc.rank);
    uint wv_base = uint(h) * uint(pc.nope + pc.v_dim) * uint(pc.rank) + uint(pc.nope) * uint(pc.rank);
    float acc = 0.0;
    for (int i = 0; i < pc.rank; i++) {
        acc += PIE_LOAD(latent[lat_base + uint(i)]) * PIE_LOAD(kv_b[wv_base + uint(j) * uint(pc.rank) + uint(i)]);
    }
    out_[uint((t * pc.heads + h) * pc.v_dim + j)] = PIE_STORE(acc);
}

#else

#define PIE_MLA_LANES 32
#define PIE_MAX_CKV_PER 16
#define PIE_MAX_KPE_PER 4

PIE_BUFFER_RO(0, q_nope);
PIE_BUFFER_RO(1, q_pe);
PIE_BUFFER_RO(2, ckv_pages);
PIE_BUFFER_RO(3, kpe_pages);
PIE_BUFFER_RW(4, out_);
[[vk::binding(5)]] StructuredBuffer<int> position_ids;
[[vk::binding(6)]] StructuredBuffer<int> req_of_token;
[[vk::binding(7)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(8)]] StructuredBuffer<uint> kv_page_indptr;
#if defined(PIE_SELECTED)
[[vk::binding(9)]] StructuredBuffer<int> selection;
#endif
struct Push {
    int page_size;
    int heads;
    int ckv;
    int kpe;
    float sm_scale;
#if defined(PIE_SELECTED)
    int top_k;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_mla_fold[PIE_MLA_LANES];

float mla_lane_sum(float v, uint lane) {
    pie_mla_fold[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_ = PIE_MLA_LANES >> 1; half_ > 0u; half_ >>= 1) {
        if (lane < half_) pie_mla_fold[lane] += pie_mla_fold[lane + half_];
        GroupMemoryBarrierWithGroupSync();
    }
    float total = pie_mla_fold[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[numthreads(PIE_MLA_LANES, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint h = group.x;
    uint row = group.y;
    uint lane = local.x;
    uint ckv = uint(pc.ckv);
    uint kpe = uint(pc.kpe);
    uint per = ckv / PIE_MLA_LANES;
    uint pper = kpe / PIE_MLA_LANES;
    uint heads = uint(pc.heads);

    int r = req_of_token[row];
    int q_pos = position_ids[row];
    int j_end = q_pos + 1;
    uint page_base = kv_page_indptr[uint(r)];

    uint qn_base = (row * heads + h) * ckv;
    uint qp_base = (row * heads + h) * kpe;
    float qn_r[PIE_MAX_CKV_PER];
    float qp_r[PIE_MAX_KPE_PER];
    for (uint i = 0u; i < PIE_MAX_CKV_PER; i++) qn_r[i] = i < per ? PIE_LOAD(q_nope[qn_base + lane + i * PIE_MLA_LANES]) : 0.0;
    for (uint i = 0u; i < PIE_MAX_KPE_PER; i++) qp_r[i] = i < pper ? PIE_LOAD(q_pe[qp_base + lane + i * PIE_MLA_LANES]) : 0.0;

    float acc[PIE_MAX_CKV_PER];
    for (uint i = 0u; i < PIE_MAX_CKV_PER; i++) acc[i] = 0.0;
    float m = -3.0e38;
    float lsum = 0.0;

#if defined(PIE_SELECTED)
    int steps = pc.top_k;
    uint srow = row * uint(pc.top_k);
#else
    int steps = j_end;
#endif
    for (int n = 0; n < steps; n++) {
        int j = n;
#if defined(PIE_SELECTED)
        j = selection[srow + uint(n)];
        if (j < 0 || j >= j_end) continue;
#endif
        uint page = kv_page_indices[page_base + uint(j / pc.page_size)];
        uint slot = page * uint(pc.page_size) + uint(j % pc.page_size);
        uint ckv_j = slot * ckv;
        uint kpe_j = slot * kpe;
        float kv[PIE_MAX_CKV_PER];
        float pd = 0.0;
        for (uint i = 0u; i < PIE_MAX_CKV_PER; i++) {
            kv[i] = i < per ? PIE_LOAD(ckv_pages[ckv_j + lane + i * PIE_MLA_LANES]) : 0.0;
            pd += qn_r[i] * kv[i];
        }
        for (uint i = 0u; i < PIE_MAX_KPE_PER; i++) {
            if (i < pper) pd += qp_r[i] * PIE_LOAD(kpe_pages[kpe_j + lane + i * PIE_MLA_LANES]);
        }
        pd = mla_lane_sum(pd, lane);
        float score = pd * pc.sm_scale;
        float m_new = max(m, score);
        float corr = exp(m - m_new);
        float p = exp(score - m_new);
        lsum = lsum * corr + p;
        for (uint i = 0u; i < PIE_MAX_CKV_PER; i++) acc[i] = acc[i] * corr + p * kv[i];
        m = m_new;
    }

    float inv = lsum > 0.0 ? (1.0 / lsum) : 0.0;
    uint o_base = (row * heads + h) * ckv;
    for (uint i = 0u; i < per; i++) out_[o_base + lane + i * PIE_MLA_LANES] = PIE_STORE(acc[i] * inv);
}
#endif

// pie:instantiate mla_latents_bf16 PIE_MLA_LATENTS=1
// pie:instantiate mla_split_q_b_bf16 PIE_MLA_SPLIT_Q=1
// pie:instantiate mla_kv_append_bf16 PIE_MLA_KV_APPEND=1
// pie:instantiate mla_absorb_q_bf16 PIE_MLA_ABSORB_Q=1
// pie:instantiate mla_absorb_out_bf16 PIE_MLA_ABSORB_OUT=1
// pie:instantiate mla_naive_paged_bf16
// pie:instantiate mla_naive_paged_selected_bf16 PIE_SELECTED=1
