#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"
#include "prim/pool.slang"

// The gated delta rule, token by token: a workgroup a value head of a row
// (or of a request), its `v_dim x k_dim` f32 state walked one token at a
// time. Each token's query and key are L2-normalized (the query also scaled
// by `1 / sqrt(k_dim)`), the state decays by `exp(g)`, takes the delta
// `beta (v - S k)` along `k`, and is read by `q`; the f32 output lands one
// value plane a row. `qkv` is the post-convolution `[q | k | v]` row (key
// heads, then value heads; a key head serves `v_heads / k_heads` value
// heads), `gates` the fused f32 `[g | beta]` row.
//
// The default form is one decode step a row (`slots[row]`); `PIE_CHUNKED`
// walks a request's rows of the query CSR `indptr`; `PIE_COMMITTED` walks a
// speculative window (`ssm/conv1d.slang` says how), on a copy of the slot's
// state in `work` (a lane's), and the slot takes the state as of row
// `commit[lane0 + r]`.
//
// The committed form also lands each lane's own rows' keys (raw, the key
// head's) and corrected values `u = beta (v - g S^T k)` (the delta each row
// writes) in its rs buffer pages, through the fire's staged rs table
// (`prim/carry.slang`): a later decode step reads them as a linear attention
// (`ssm/gdn_carried.slang`). A lane landing none states it there.
//
// `PIE_SNAPS` (with either of the last two): a request's lane also lands
// the state as of up to `PIE_RS_SNAPSHOTS` rows inside it in records of
// their own (`prim/pool.slang`, CONTRACT-BC 16).
//
// `PIE_DK_REG` holds a lane's state row in registers, for `k_dim ==
// PIE_DK_REG` and `v_dim <= PIE_GROUP_X`; without it the state is walked in
// memory. The `subgroup` tier folds the norms through subgroups.

#define KMAX 256

#if defined(PIE_COMMITTED)
[[vk::binding(0)]] StructuredBuffer<uint> qkv;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] StructuredBuffer<float> gates;
[[vk::binding(6)]] RWStructuredBuffer<float> work;
[[vk::binding(7)]] RWStructuredBuffer<float> y;
[[vk::binding(8)]] StructuredBuffer<int> carry;
#if defined(PIE_SNAPS)
[[vk::binding(9)]] StructuredBuffer<int> pie_snaps;
#define SEGMENTS_AT 10
#else
#define SEGMENTS_AT 9
#endif
#define STATE work
#elif defined(PIE_CHUNKED)
[[vk::binding(0)]] StructuredBuffer<uint> qkv;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<float> gates;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] RWStructuredBuffer<float> y;
#if defined(PIE_SNAPS)
[[vk::binding(5)]] StructuredBuffer<int> pie_snaps;
#define SEGMENTS_AT 6
#else
#define SEGMENTS_AT 5
#endif
#define STATE rstate
#else
[[vk::binding(0)]] StructuredBuffer<uint> qkv;
[[vk::binding(1)]] StructuredBuffer<float> gates;
[[vk::binding(2)]] StructuredBuffer<uint> slots;
[[vk::binding(3)]] RWStructuredBuffer<float> y;
#define SEGMENTS_AT 4
#define STATE rstate
#endif
// The state bank (and the rs list), in the chunk pool's segments
// (`prim/pool.slang`), as words: f32 state and bf16 keys alike.
// The register forms read a state row as vectors of four words (the pool's
// segments, the state's origin and its slot stride whole vectors: the host
// refuses another pool); the others as words.
#if defined(PIE_DK_REG)
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint4, SEGMENTS_AT)
#else
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, SEGMENTS_AT)
#endif
#define PARAMS_AT (SEGMENTS_AT + 4)

struct Params {
    uint seg_words;
    uint state_origin;
    uint slot_stride;
    uint pieces;
    uint piece_first;
    uint piece_atoms;
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int k_heads;
    int v_heads;
    int k_dim;
    int v_dim;
    // The fire row a window's row 0 is: `qkv`, `gates`, `y` and the slots
    // table are the fire's whole, bound once whatever window a fire cuts
    // (0 for the committed form, whose extended planes are the window's).
    uint row0;
#if defined(PIE_CHUNKED) || defined(PIE_COMMITTED)
    // The window's CSR lies at int `indptr_at` of the fire's packed windows
    // table, which is bound whole: one view whatever window a fire cuts.
    int indptr_at;
#endif
#if defined(PIE_SNAPS)
    int snap_lane0;
#endif
#if defined(PIE_COMMITTED)
    // The rs list's pages, and where a row's keys and `u` lie in one.
    uint page_tokens;
    uint stride_words;
    uint k_at;
    uint k_row;
    uint u_at;
    uint u_row;
#endif
};
PIE_PARAMS(PARAMS_AT, Params, params);

#if defined(PIE_CHUNKED) || defined(PIE_COMMITTED)
// The window's CSR boundary `i`.
int csr(uint i) {
    return indptr[uint(params.indptr_at) + i];
}
#endif
#if defined(PIE_DK_REG)
PIE_POOL_READ(uint4, params.seg_words / 4u)
// Virtual word `v` of the pool, set.
void pie_pool_set(uint v, uint x) {
    uint seg, at;
    pie_pool_find(v >> 2, seg, at);
#define GDN_WORD_SET(buf_) buf_[at][v & 3u] = x;
    PIE_POOL_BRANCH(seg, GDN_WORD_SET)
}
#else
PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)
// The state from its origin on, as f32.
struct rstate_pool_view {
    __subscript(uint i)->float {
        get { return asfloat(pie_pool_get(params.state_origin + i)); }
        set { pie_pool_set(params.state_origin + i, asuint(newValue)); }
    }
};
static rstate_pool_view rstate;
#endif

PIE_STATE_PIECES
#if defined(PIE_COMMITTED)
#if defined(PIE_DK_REG)
#define PIE_CARRY_POOL_WORD(v) (pie_pool_get((v) >> 2)[(v) & 3u])
#endif
#include "prim/carry.slang"
// The virtual word the token's `u` lands at (its head's), `CARRY_NONE`
// where it lands none.
#define CARRY_NONE 0xffffffffu
static uint u_into = CARRY_NONE;
#endif
#if defined(PIE_SNAPS)
PIE_STATE_SNAPSHOTS
#endif

// Where value head `hv`'s state starts in the record of `slots` entry `at`.
uint state_at(uint at, uint hv, uint head) {
    return pie_state_atom(at, hv, head);
}

groupshared float sq[KMAX];
groupshared float sk[KMAX];
groupshared int gdn_lo;
groupshared int gdn_hi;
groupshared int gdn_keep;

#if defined(PIE_DK_REG)
static float st[PIE_DK_REG];
#endif

// Token `t` of value head `hv` (key head `hk`), its state at `state_base`.
void token(uint tid, uint window_t, uint hv, uint hk, uint state_base) {
    uint t = params.row0 + window_t;
    uint dk = uint(params.k_dim);
    uint dv = uint(params.v_dim);
    uint keys = uint(params.k_heads) * dk;
    uint pitch = 2u * keys + uint(params.v_heads) * dv;
    uint row = t * pitch;
    uint qbase = row + hk * dk;
    uint kbase = qbase + keys;
    uint vbase = row + 2u * keys + hv * dv;

    float qsum = 0.0;
    float ksum = 0.0;
    for (uint i = tid; i < dk; i += uint(PIE_GROUP_X)) {
        float qv = bf16_at(qkv, qbase + i);
        float kv = bf16_at(qkv, kbase + i);
        sq[i] = qv;
        sk[i] = kv;
        qsum += qv * qv;
        ksum += kv * kv;
    }
    float qtot = pie_workgroup_sum(qsum, tid);
    float ktot = pie_workgroup_sum(ksum, tid);
    float scale = 1.0 / sqrt(float(dk));
    float qinv = rsqrt(qtot + 1e-6) * scale;
    float kinv = rsqrt(ktot + 1e-6);
    for (uint i = tid; i < dk; i += uint(PIE_GROUP_X)) {
        sq[i] *= qinv;
        sk[i] *= kinv;
    }
    GroupMemoryBarrierWithGroupSync();

    uint fused = t * 2u * uint(params.v_heads) + hv;
    float decay = exp(gates[fused]);
    float beta = gates[fused + uint(params.v_heads)];
    uint out = (t * uint(params.v_heads) + hv) * dv;
#if defined(PIE_DK_REG)
    if (tid < dv) {
        float kv_mem = 0.0;
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            st[i] *= decay;
            kv_mem += st[i] * sk[i];
        }
        float delta = (bf16_at(qkv, vbase + tid) - kv_mem) * beta;
#if defined(PIE_COMMITTED)
        if (u_into != CARRY_NONE) {
            pie_pool_set(u_into + tid, asuint(delta));
        }
#endif
        float acc = 0.0;
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            st[i] += sk[i] * delta;
            acc += st[i] * sq[i];
        }
        y[out + tid] = acc;
    }
#else
    for (uint c = tid; c < dv; c += uint(PIE_GROUP_X)) {
        uint cell = state_base + c * dk;
        float kv_mem = 0.0;
        for (uint i = 0; i < dk; i++) {
            float s = STATE[cell + i] * decay;
            STATE[cell + i] = s;
            kv_mem += s * sk[i];
        }
        float delta = (bf16_at(qkv, vbase + c) - kv_mem) * beta;
#if defined(PIE_COMMITTED)
        if (u_into != CARRY_NONE) {
            pie_pool_set(u_into + c, asuint(delta));
        }
#endif
        float acc = 0.0;
        for (uint i = 0; i < dk; i++) {
            float s = STATE[cell + i] + sk[i] * delta;
            STATE[cell + i] = s;
            acc += s * sq[i];
        }
        y[out + c] = acc;
    }
#endif
    GroupMemoryBarrierWithGroupSync();
}

#if defined(PIE_COMMITTED)
// Where run position `t` of a lane (its `replay` replayed rows, then its
// own, fire row `row`) lands its `u` (`u_into`, read by `token`), and its
// keys, landed here by its key head's first value head.
void land_at(CarryLane cl, int replay, int t, uint tid, uint row, uint hv, uint hk) {
    u_into = CARRY_NONE;
    int e = t - replay;
    if (cl.at < 0 || e < 0) {
        return;
    }
    uint tok = uint(cl.at + e);
    u_into = carry_word(cl, tok, params.u_at, params.u_row) + hv * uint(params.v_dim);
    if (hv % (uint(params.v_heads) / uint(params.k_heads)) == 0u) {
        uint dk = uint(params.k_dim);
        uint keys = uint(params.k_heads) * dk;
        uint pitch = 2u * keys + uint(params.v_heads) * uint(params.v_dim);
        uint from = (row * pitch + keys + hk * dk) / 2u;
        uint into = carry_word(cl, tok, params.k_at, params.k_row) + hk * (dk / 2u);
        for (uint w = tid; w < dk / 2u; w += uint(PIE_GROUP_X)) {
            pie_pool_set(into + w, qkv[from + w]);
        }
    }
}
#endif

#if defined(PIE_DK_REG)
// A lane's state row lies in one slot, so in one segment: found once, and
// read and written four words a vector.
void reg_load(uint tid, uint base) {
    uint dk = uint(params.k_dim);
    if (tid >= uint(params.v_dim)) {
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            st[i] = 0.0;
        }
        return;
    }
    uint seg, at;
    pie_pool_find((params.state_origin + base + tid * dk) / 4u, seg, at);
#define GDN_REG_LOAD(buf_) \
    [ForceUnroll] for (uint m = 0; m < uint(PIE_DK_REG) / 4u; m++) { \
        uint4 v = buf_[at + m]; \
        st[4u * m] = asfloat(v.x); \
        st[4u * m + 1u] = asfloat(v.y); \
        st[4u * m + 2u] = asfloat(v.z); \
        st[4u * m + 3u] = asfloat(v.w); \
    }
    PIE_POOL_BRANCH(seg, GDN_REG_LOAD)
}

void reg_store(uint tid, uint base) {
    uint dk = uint(params.k_dim);
    if (tid < uint(params.v_dim)) {
        uint seg, at;
        pie_pool_find((params.state_origin + base + tid * dk) / 4u, seg, at);
#define GDN_REG_STORE(buf_) \
        [ForceUnroll] for (uint m = 0; m < uint(PIE_DK_REG) / 4u; m++) { \
            buf_[at + m] = uint4(asuint(st[4u * m]), asuint(st[4u * m + 1u]), \
                                 asuint(st[4u * m + 2u]), asuint(st[4u * m + 3u])); \
        }
        PIE_POOL_BRANCH(seg, GDN_REG_STORE)
    }
}
#endif

#if defined(PIE_SNAPS)
// The next snapshot of lane `snap` from `n` on: its row, 0 past the last.
int snap_next(int snap, uint n) {
    int at = pie_snap_row(snap, n);
    return at > 0 ? at : 0;
}

#if !defined(PIE_DK_REG)
// This lane's cells of the head state at `from` (in `STATE`) into the same
// cells of snapshot `n` of lane `snap`: the cells it walks, so no other
// lane's writes are read.
void snap_cells(uint tid, uint from, int snap, uint n, uint hv, uint head) {
    uint dk = uint(params.k_dim);
    uint into = pie_snap_atom(uint(snap), n, hv, head);
    for (uint c = tid; c < uint(params.v_dim); c += uint(PIE_GROUP_X)) {
        for (uint i = 0; i < dk; i++) {
            rstate[into + c * dk + i] = STATE[from + c * dk + i];
        }
    }
}
#endif
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint tid = local.x;
    uint hv = group.y;
    uint hk = hv / (uint(params.v_heads) / uint(params.k_heads));
    uint dk = uint(params.k_dim);
    uint dv = uint(params.v_dim);
    uint head = dv * dk;
#if defined(PIE_DK_REG)
    if (dk != uint(PIE_DK_REG) || dv > uint(PIE_GROUP_X)) {
        return;
    }
#endif
    uint r = group.z;
#if defined(PIE_COMMITTED)
    uint lane = uint(params.lane0) + r;
    if (tid == 0u) {
        int begin = csr(r);
        for (uint j = 0; j < r; j++) {
            begin += replay[uint(params.lane0) + j];
        }
        int span = (csr(r + 1) - csr(r)) + replay[lane];
        gdn_lo = begin;
        gdn_hi = pie_state_none(lane) ? begin : begin + span;
        gdn_keep = min(commit[lane], span);
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: WGSL lets a barrier follow only workgroup-uniform
    // control flow, and these bound the token loop.
    int begin = workgroupUniformLoad(gdn_lo);
    int span = workgroupUniformLoad(gdn_hi) - begin;
    int keep = workgroupUniformLoad(gdn_keep);
    if (span <= 0) {
        return;
    }
    uint bank = state_at(lane, hv, head);
    CarryLane cl = carry_lane(lane);
    int rep = replay[lane];
#if defined(PIE_SNAPS)
    int snap = pie_snap_lane(r);
    uint snapped = 0u;
    int snap_at = snap_next(snap, 0u);
#endif
#if defined(PIE_DK_REG)
    reg_load(tid, bank);
    for (int t = 0; t < span; t++) {
        land_at(cl, rep, t, tid, params.row0 + uint(begin + t), hv, hk);
        token(tid, uint(begin + t), hv, hk, 0u);
        if (t + 1 == keep) {
            reg_store(tid, bank);
        }
#if defined(PIE_SNAPS)
        if (t + 1 == snap_at) {
            reg_store(tid, pie_snap_atom(uint(snap), snapped, hv, head));
            snapped++;
            snap_at = snap_next(snap, snapped);
        }
#endif
    }
#else
    // The work plane holds the dispatch's lanes, the lane's at its own `r`:
    // a fire's lanes take as many dispatches as the plane holds lanes.
    uint state_base = (r * uint(params.v_heads) + hv) * head;
    // A row of the head's state lies in one segment: found once a row.
    for (uint c = tid; c < dv; c += uint(PIE_GROUP_X)) {
        uint seg, at;
        pie_pool_find(params.state_origin + bank + c * dk, seg, at);
#define GDN_BANK_LOAD(buf_) \
        for (uint i = 0; i < dk; i++) { \
            work[state_base + c * dk + i] = asfloat(buf_[at + i]); \
        }
        PIE_POOL_BRANCH(seg, GDN_BANK_LOAD)
    }
    GroupMemoryBarrierWithGroupSync();
    for (int t = 0; t < span; t++) {
        land_at(cl, rep, t, tid, params.row0 + uint(begin + t), hv, hk);
        token(tid, uint(begin + t), hv, hk, state_base);
        if (t + 1 == keep) {
            for (uint c = tid; c < dv; c += uint(PIE_GROUP_X)) {
                uint seg, at;
                pie_pool_find(params.state_origin + bank + c * dk, seg, at);
#define GDN_BANK_STORE(buf_) \
                for (uint i = 0; i < dk; i++) { \
                    buf_[at + i] = asuint(work[state_base + c * dk + i]); \
                }
                PIE_POOL_BRANCH(seg, GDN_BANK_STORE)
            }
        }
#if defined(PIE_SNAPS)
        if (t + 1 == snap_at) {
            snap_cells(tid, state_base, snap, snapped, hv, head);
            snapped++;
            snap_at = snap_next(snap, snapped);
        }
#endif
    }
#endif
#elif defined(PIE_CHUNKED)
    if (tid == 0u) {
        gdn_lo = csr(r);
        gdn_hi = csr(r + 1);
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: they bound the token loop.
    int begin = workgroupUniformLoad(gdn_lo);
    int end = workgroupUniformLoad(gdn_hi);
    if (end <= begin) {
        return;
    }
    uint state_base = state_at(params.row0 + uint(begin), hv, head);
#if defined(PIE_SNAPS)
    int snap = pie_snap_lane(r);
    uint snapped = 0u;
    int snap_at = snap_next(snap, 0u);
#endif
#if defined(PIE_DK_REG)
    reg_load(tid, state_base);
#endif
    for (int t = begin; t < end; t++) {
        token(tid, uint(t), hv, hk, state_base);
#if defined(PIE_SNAPS)
        if (t - begin + 1 == snap_at) {
#if defined(PIE_DK_REG)
            reg_store(tid, pie_snap_atom(uint(snap), snapped, hv, head));
#else
            snap_cells(tid, state_base, snap, snapped, hv, head);
#endif
            snapped++;
            snap_at = snap_next(snap, snapped);
        }
#endif
    }
#if defined(PIE_DK_REG)
    reg_store(tid, state_base);
#endif
#else
    uint state_base = state_at(params.row0 + r, hv, head);
#if defined(PIE_DK_REG)
    reg_load(tid, state_base);
#endif
    token(tid, r, hv, hk, state_base);
#if defined(PIE_DK_REG)
    reg_store(tid, state_base);
#endif
#endif
}

// pie:instantiate ssm_gdn_recurrent_bf16 PIE_GROUP_X=128
// pie:instantiate ssm_gdn_recurrent_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate ssm_gdn_recurrent_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
// pie:instantiate ssm_gdn_recurrent_r128_bf16 PIE_GROUP_X=128 PIE_DK_REG=128
// pie:instantiate ssm_gdn_recurrent_chunked_r128_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_DK_REG=128
// pie:instantiate ssm_gdn_recurrent_committed_r128_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_DK_REG=128
// pie:instantiate ssm_gdn_recurrent_bf16 PIE_GROUP_X=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_r128_bf16 PIE_GROUP_X=128 PIE_DK_REG=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_chunked_r128_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_DK_REG=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_committed_r128_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_DK_REG=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_chunked_snap_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_SNAPS=1
// pie:instantiate ssm_gdn_recurrent_committed_snap_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_SNAPS=1
// pie:instantiate ssm_gdn_recurrent_chunked_snap_r128_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_DK_REG=128 PIE_SNAPS=1
// pie:instantiate ssm_gdn_recurrent_committed_snap_r128_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_DK_REG=128 PIE_SNAPS=1
// pie:instantiate ssm_gdn_recurrent_chunked_snap_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_SNAPS=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_committed_snap_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_SNAPS=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_chunked_snap_r128_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_DK_REG=128 PIE_SNAPS=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_recurrent_committed_snap_r128_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_DK_REG=128 PIE_SNAPS=1 PIE_TIER=subgroup PIE_TARGET=spirv
