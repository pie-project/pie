#include "common/bf16.slang"
#include "common/reduce.slang"

#define PIE_NEG_INF asfloat(0xff800000u)

#if defined(PIE_POOL_BOUNDARY)

#define PIE_GROUP_X 128
[[vk::binding(0)]] StructuredBuffer<int> positions;
#if defined(PIE_PREFILL)
[[vk::binding(1)]] StructuredBuffer<uint> qo_indptr;
[[vk::binding(2)]] RWStructuredBuffer<int> out_pos;
[[vk::binding(3)]] RWStructuredBuffer<int> out_req;
[[vk::binding(4)]] RWStructuredBuffer<int> out_rope;
[[vk::binding(5)]] StructuredBuffer<uint8_t> row_valid;
struct Push { int n; int num_requests; int ratio; };
#else
[[vk::binding(1)]] RWStructuredBuffer<int> out_pos;
[[vk::binding(2)]] RWStructuredBuffer<int> out_req;
[[vk::binding(3)]] RWStructuredBuffer<int> out_rope;
[[vk::binding(4)]] StructuredBuffer<uint8_t> row_valid;
struct Push { int n; int ratio; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int t = int(gid.x);
    if (t >= pc.n) return;
    int p = positions[uint(t)];
    bool valid = row_valid[uint(t)] != uint8_t(0);
    bool is_boundary = valid && (((p + 1) % pc.ratio) == 0);
    out_pos[uint(t)] = is_boundary ? p : -1;
    out_rope[uint(t)] = is_boundary ? (p / pc.ratio) * pc.ratio : 0;
#if defined(PIE_PREFILL)
    int lo = 0;
    int hi = pc.num_requests;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (int(qo_indptr[uint(mid)]) <= t) lo = mid; else hi = mid;
    }
    out_req[uint(t)] = lo;
#else
    out_req[uint(t)] = t;
#endif
}

#elif defined(PIE_POOL_STATE_WRITE)

#define PIE_GROUP_X 256
PIE_BUFFER_RO(0, kv);
PIE_BUFFER_RO(1, score);
PIE_BUFFER_RW(2, state_kv);
PIE_BUFFER_RW(3, state_score);
[[vk::binding(4)]] StructuredBuffer<uint> w_page;
[[vk::binding(5)]] StructuredBuffer<uint> w_off;
struct Push { int width; int page_size; int state_pitch; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int d = int(gid.x);
    uint i = gid.y;
    if (d >= pc.width || i >= uint(pc.rows)) return;
    int page = int(w_page[i]);
    int off = int(w_off[i]);
    if (page < 0 || off < 0 || off >= pc.page_size) return;
    uint slot = uint(page) * uint(pc.page_size) + uint(off);
    uint dst = slot * uint(pc.state_pitch) + uint(d);
    uint src = i * uint(pc.width) + uint(d);
    state_kv[dst] = kv[src];
    state_score[dst] = score[src];
}

#elif defined(PIE_POOL_GATHER)

#define PIE_GROUP_X 256
PIE_BUFFER_RO(0, state_kv);
PIE_BUFFER_RO(1, state_score);
[[vk::binding(2)]] StructuredBuffer<float> ape;
[[vk::binding(3)]] StructuredBuffer<int> boundary_pos;
[[vk::binding(4)]] StructuredBuffer<int> boundary_req;
[[vk::binding(5)]] StructuredBuffer<uint> page_indices;
[[vk::binding(6)]] StructuredBuffer<uint> page_indptr;
PIE_BUFFER_RW(7, out_);
struct Push { int head_dim; int ratio; int coff; int page_size; int has_ape; int state_pitch; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

uint pool_slot(int req, int pos) {
    uint page = page_indices[page_indptr[uint(req)] + uint(pos / pc.page_size)];
    return page * uint(pc.page_size) + uint(pos % pc.page_size);
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int d = int(gid.x);
    uint c = gid.y;
    if (d >= pc.head_dim || c >= uint(pc.rows)) return;
    int window = pc.coff * pc.ratio;
    uint width = uint(pc.coff * pc.head_dim);
    uint pitch = uint(pc.state_pitch);
    int bpos = boundary_pos[c];
    int req = boundary_req[c];
    uint at = c * uint(pc.head_dim) + uint(d);
    if (bpos < 0) {
        out_[at] = PIE_STORE(0.0);
        return;
    }
    float max_s = PIE_NEG_INF;
    for (int i = 0; i < window; i++) {
        int pos = bpos + i - (window - 1);
        if (pos < 0) continue;
        uint col = uint((i >= pc.ratio ? pc.head_dim : 0) + d);
        uint slot = pool_slot(req, pos);
        float sc = PIE_LOAD(state_score[slot * pitch + col]);
        if (pc.has_ape != 0) sc += ape[uint(pos % pc.ratio) * width + col];
        max_s = max(max_s, sc);
    }
    if (!(max_s > -3.0e38)) {
        out_[at] = PIE_STORE(0.0);
        return;
    }
    float sum_e = 0.0;
    float acc = 0.0;
    for (int i = 0; i < window; i++) {
        int pos = bpos + i - (window - 1);
        if (pos < 0) continue;
        uint col = uint((i >= pc.ratio ? pc.head_dim : 0) + d);
        uint slot = pool_slot(req, pos);
        float sc = PIE_LOAD(state_score[slot * pitch + col]);
        if (pc.has_ape != 0) sc += ape[uint(pos % pc.ratio) * width + col];
        float e = exp(sc - max_s);
        sum_e += e;
        acc += e * PIE_LOAD(state_kv[slot * pitch + col]);
    }
    out_[at] = PIE_STORE(sum_e > 0.0 ? acc / sum_e : 0.0);
}

#elif defined(PIE_POOL_STORE)

#define PIE_GROUP_X 256
PIE_BUFFER_RO(0, entries);
PIE_BUFFER_RW(1, comp_kv_pages);
[[vk::binding(2)]] StructuredBuffer<int> boundary_pos;
[[vk::binding(3)]] StructuredBuffer<int> boundary_req;
[[vk::binding(4)]] StructuredBuffer<uint> page_indices;
[[vk::binding(5)]] StructuredBuffer<uint> page_indptr;
struct Push { int head_dim; int page_size; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int d = int(gid.x);
    uint c = gid.y;
    if (d >= pc.head_dim || c >= uint(pc.rows)) return;
    int bpos = boundary_pos[c];
    if (bpos < 0) return;
    int req = boundary_req[c];
    uint page = page_indices[page_indptr[uint(req)] + uint(bpos / pc.page_size)];
    uint slot = page * uint(pc.page_size) + uint(bpos % pc.page_size);
    comp_kv_pages[slot * uint(pc.head_dim) + uint(d)] = entries[c * uint(pc.head_dim) + uint(d)];
}

#else

#define PIE_GROUP_X 128
#define PIE_POOL_HEAD_MAX 512
#define PIE_DIMS_PER_THREAD (PIE_POOL_HEAD_MAX / PIE_GROUP_X)
PIE_BUFFER_RO(0, q);
PIE_BUFFER_RO(1, comp_kv_pages);
#if defined(PIE_SELECTED)
[[vk::binding(2)]] StructuredBuffer<int> selection;
PIE_BUFFER_RW(3, out_);
[[vk::binding(4)]] RWStructuredBuffer<float> lse_out;
[[vk::binding(5)]] StructuredBuffer<int> positions;
[[vk::binding(6)]] StructuredBuffer<uint> page_indices;
[[vk::binding(7)]] StructuredBuffer<uint> page_indptr;
[[vk::binding(8)]] StructuredBuffer<int> req_of_token;
struct Push { int num_q_heads; int head_dim; int ratio; int top_k; int page_size; float scale; };
#else
PIE_BUFFER_RW(2, out_);
[[vk::binding(3)]] RWStructuredBuffer<float> lse_out;
[[vk::binding(4)]] StructuredBuffer<int> positions;
[[vk::binding(5)]] StructuredBuffer<uint> page_indices;
[[vk::binding(6)]] StructuredBuffer<uint> page_indptr;
[[vk::binding(7)]] StructuredBuffer<int> req_of_token;
struct Push { int num_q_heads; int head_dim; int ratio; int page_size; float scale; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_pool_q[PIE_POOL_HEAD_MAX];

uint pool_slot(int req, int pos) {
    uint page = page_indices[page_indptr[uint(req)] + uint(pos / pc.page_size)];
    return page * uint(pc.page_size) + uint(pos % pc.page_size);
}

int key_of(int n, uint qi, int num_visible) {
#if defined(PIE_SELECTED)
    int c = selection[qi * uint(pc.top_k) + uint(n)];
    return (c < 0 || c >= num_visible) ? -1 : c;
#else
    return n;
#endif
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint qi = group.y;
    uint q_head = group.z;
    int tid = int(local.x);
    uint head_dim = uint(pc.head_dim);
    int req = req_of_token[qi];
    int qpos = positions[qi];
    int num_visible = (qpos + 1) / pc.ratio;

    uint q_row = (qi * uint(pc.num_q_heads) + q_head) * head_dim;
    for (uint d = uint(tid); d < head_dim; d += PIE_GROUP_X) pie_pool_q[d] = PIE_LOAD(q[q_row + d]);
    GroupMemoryBarrierWithGroupSync();

    uint lse_at = qi * uint(pc.num_q_heads) + q_head;
    if (num_visible <= 0) {
        for (uint d = uint(tid); d < head_dim; d += PIE_GROUP_X) out_[q_row + d] = PIE_STORE(0.0);
        if (tid == 0) lse_out[lse_at] = PIE_NEG_INF;
        return;
    }

#if defined(PIE_SELECTED)
    int steps = pc.top_k;
#else
    int steps = num_visible;
#endif

    float local_max = PIE_NEG_INF;
    for (int n = tid; n < steps; n += PIE_GROUP_X) {
        int c = key_of(n, qi, num_visible);
        if (c < 0) continue;
        uint k_row = pool_slot(req, (c + 1) * pc.ratio - 1) * head_dim;
        float dot = 0.0;
        for (uint d = 0u; d < head_dim; d++) dot += pie_pool_q[d] * PIE_LOAD(comp_kv_pages[k_row + d]);
        local_max = max(local_max, dot * pc.scale);
    }
    float row_max = pie_workgroup_max(local_max);
    if (!(row_max > -3.0e38)) {
        for (uint d = uint(tid); d < head_dim; d += PIE_GROUP_X) out_[q_row + d] = PIE_STORE(0.0);
        if (tid == 0) lse_out[lse_at] = PIE_NEG_INF;
        return;
    }

    float acc[PIE_DIMS_PER_THREAD];
    for (uint i = 0u; i < PIE_DIMS_PER_THREAD; i++) acc[i] = 0.0;
    float local_z = 0.0;
    for (int n = 0; n < steps; n++) {
        int c = key_of(n, qi, num_visible);
        if (c < 0) continue;
        uint k_row = pool_slot(req, (c + 1) * pc.ratio - 1) * head_dim;
        float dot = 0.0;
        for (uint d = uint(tid); d < head_dim; d += PIE_GROUP_X) dot += pie_pool_q[d] * PIE_LOAD(comp_kv_pages[k_row + d]);
        float s = pie_workgroup_sum(dot);
        float w = exp(s * pc.scale - row_max);
        local_z += w;
        for (uint i = 0u; i < PIE_DIMS_PER_THREAD; i++) {
            uint d = uint(tid) + i * PIE_GROUP_X;
            if (d < head_dim) acc[i] += w * PIE_LOAD(comp_kv_pages[k_row + d]);
        }
    }
    float inv_z = local_z > 0.0 ? 1.0 / local_z : 0.0;
    if (tid == 0) {
        lse_out[lse_at] = local_z > 0.0 ? (log(local_z) + row_max) * 1.44269504088896340736 : PIE_NEG_INF;
    }
    for (uint i = 0u; i < PIE_DIMS_PER_THREAD; i++) {
        uint d = uint(tid) + i * PIE_GROUP_X;
        if (d < head_dim) out_[q_row + d] = PIE_STORE(acc[i] * inv_z);
    }
}
#endif

// pie:instantiate pool_boundary_decode PIE_POOL_BOUNDARY=1
// pie:instantiate pool_boundary_prefill PIE_POOL_BOUNDARY=1 PIE_PREFILL=1
// pie:instantiate pool_state_write_bf16 PIE_POOL_STATE_WRITE=1
// pie:instantiate pool_gather_paged_bf16 PIE_POOL_GATHER=1
// pie:instantiate pool_store_entries_bf16 PIE_POOL_STORE=1
// pie:instantiate pool_lse_paged
// pie:instantiate pool_lse_selected_paged PIE_SELECTED=1
