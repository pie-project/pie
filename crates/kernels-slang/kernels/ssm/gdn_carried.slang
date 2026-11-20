#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"
#include "prim/pool.slang"

// The gated delta rule's decode step over a lane's buffered rows then its
// own (`prim/carry.slang`): the lane's committed state is read once, its
// buffered rows from its rs buffer pages where they lie, and none is
// re-stepped. A row's step is `S_s = g_s S_{s-1} + k_s u_s^T`, its corrected
// value `u_s = beta_s (v_s - g_s S_{s-1}^T k_s)`; from the committed state
// `S`, after buffered rows `1..m`,
//   S_m = G_{0,m} S + sum_s G_{s,m} k_s u_s^T,   G_{s,m} = prod_{s<i<=m} g_i,
// so the own row reads the state once and each buffered row as two dots:
// `S_m^T x = G_{0,m} S^T x + sum_s G_{s,m} (k_s . x) u_s`. A buffered row
// keeps its keys (raw, the conv output's key heads), its `u` and its `[b |
// a]` (the planes `k`, `u`, `ba` of the lane's pages), each landed once, when
// it was a fire's own row. The rows the lane commits among its carried ones
// fold into the state (`S <- g S + k u^T`) before the rest are read; the own
// row folds too where the lane commits it; the state is stored once if any
// row folded. The own row's keys, `u` and `[b | a]` land in its buffer.
//
// A workgroup a value head of a lane (one decode row a lane), a lane a value
// column holding its `k_dim = PIE_DK` state row in registers (`v_dim <=
// PIE_GROUP_X`). The own row's gates are formed from its `[b | a]` as
// `ssm/gdn_prep.slang` forms them, a buffered row's from its stored pair.

#define CH 16u

[[vk::binding(0)]] StructuredBuffer<uint> qkv;
[[vk::binding(1)]] StructuredBuffer<uint> ba;
[[vk::binding(2)]] StructuredBuffer<uint> dt_bias;
[[vk::binding(3)]] StructuredBuffer<float> a_log;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] StructuredBuffer<int> carry;
[[vk::binding(6)]] RWStructuredBuffer<float> y;
// The state bank and the rs list, in the chunk pool's segments, as words.
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 7)

struct Params {
    uint seg_words;
    uint state_origin;
    uint slot_stride;
    uint pieces;
    uint piece_first;
    uint piece_atoms;
    // The window's first lane in the fire's tables (`slots`, `carry`).
    int lane0;
    int k_heads;
    int v_heads;
    int k_dim;
    int v_dim;
    // The fire row the window's row 0 is.
    uint row0;
    uint page_tokens;
    uint stride_words;
    // Where the planes lie in a page, and the words a token of each.
    uint ba_at;
    uint ba_row;
    uint k_at;
    uint k_row;
    uint u_at;
    uint u_row;
};
PIE_PARAMS(11, Params, params);

PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)
PIE_STATE_PIECES

#include "prim/carry.slang"

groupshared float sq[PIE_DK];
groupshared float sk[PIE_DK];
groupshared float skeys[CH * PIE_DK];
groupshared float sglog[CH];
groupshared uint su[CH];
groupshared float snj[CH];
groupshared float sdq[CH];
groupshared float sdk[CH];
groupshared int g_n;
groupshared int g_commit;
groupshared int g_none;

static float st[PIE_DK];

// A row's log-gate from its `a` (`ssm/gdn_prep.slang`'s `g`).
float log_gate(float a, uint h) {
    float z = a + bf16_at(dt_bias, h);
    float sp = z > 20.0 ? z : log(1.0 + exp(z));
    return -exp(a_log[h]) * sp;
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint tid = local.x;
    uint hv = group.y;
    uint r = group.z;
    uint vh = uint(params.v_heads);
    uint kh = uint(params.k_heads);
    uint per_key = vh / kh;
    uint hk = hv / per_key;
    uint dk = uint(params.k_dim);
    uint dv = uint(params.v_dim);
    if (dk != uint(PIE_DK) || dv > uint(PIE_GROUP_X)) {
        return;
    }
    uint lane = uint(params.lane0) + r;
    uint row = params.row0 + r;
    CarryLane cl = carry_lane(lane);
    if (tid == 0u) {
        g_n = cl.n;
        g_commit = min(cl.commit, cl.n + 1);
        g_none = pie_state_none(lane) ? 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: WGSL lets a barrier follow only workgroup-uniform
    // control flow, and these bound the loops.
    int n = workgroupUniformLoad(g_n);
    int commit = workgroupUniformLoad(g_commit);
    if (workgroupUniformLoad(g_none) != 0) {
        return;
    }

    // The own row: q (normalized, scaled), k (normalized), its gates.
    uint keys = kh * dk;
    uint pitch = 2u * keys + vh * dv;
    uint qbase = row * pitch + hk * dk;
    uint kbase = qbase + keys;
    uint vbase = row * pitch + 2u * keys + hv * dv;
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
    float qinv = rsqrt(qtot + 1e-6) / sqrt(float(dk));
    float kinv = rsqrt(ktot + 1e-6);
    float qk = 0.0;
    for (uint i = tid; i < dk; i += uint(PIE_GROUP_X)) {
        sq[i] *= qinv;
        sk[i] *= kinv;
        qk += sq[i] * sk[i];
    }
    float kq = pie_workgroup_sum(qk, tid);
    uint ba_row = row * 2u * vh;
    float glog_own = log_gate(bf16_at(ba, ba_row + vh + hv), hv);
    float beta = 1.0 / (1.0 + exp(-bf16_at(ba, ba_row + hv)));

    // The lane's column of the committed state, found once (a state row
    // lies in one slot, so in one segment).
    uint bank = pie_state_atom(lane, hv, dv * dk);
    uint seg = 0u;
    uint cell = 0u;
    if (tid < dv) {
        pie_pool_find(params.state_origin + bank + tid * dk, seg, cell);
#define GDN_CARRIED_LOAD(buf_) \
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK); i++) { \
            st[i] = asfloat(buf_[cell + i]); \
        }
        PIE_POOL_BRANCH(seg, GDN_CARRIED_LOAD)
    }

    // The buffered rows, CH at a time: each row's keys (its key head's) and
    // gate staged once for the workgroup, then each row's norm and its dots
    // with the own row's q and k (a lane a row), then each lane walks the
    // rows in order: a committed row folds into its column, a later one adds
    // to the attention sums, every sum decaying by the rows after it.
    float decay = 1.0;
    float aq = 0.0;
    float ak = 0.0;
    uint half_dk = dk / 2u;
    for (int base = 0; base < n; base += int(CH)) {
        uint rows = uint(min(int(CH), n - base));
        for (uint idx = tid; idx < rows * half_dk; idx += uint(PIE_GROUP_X)) {
            uint e = idx / half_dk;
            uint w = idx % half_dk;
            uint t = carry_tok(cl, uint(base) + e);
            uint word = pie_pool_get(carry_word(cl, t, params.k_at, params.k_row) + hk * half_dk + w);
            skeys[e * dk + 2u * w] = bf16_to_f32(word & 0xffffu);
            skeys[e * dk + 2u * w + 1u] = bf16_to_f32(word >> 16);
        }
        if (tid < rows) {
            uint t = carry_tok(cl, uint(base) + tid);
            uint at = vh + hv;
            uint word = pie_pool_get(carry_word(cl, t, params.ba_at, params.ba_row) + at / 2u);
            sglog[tid] = log_gate(bf16_to_f32((at & 1u) != 0u ? (word >> 16) : (word & 0xffffu)), hv);
            su[tid] = carry_word(cl, t, params.u_at, params.u_row) + hv * dv;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < rows) {
            float kk = 0.0;
            float dq = 0.0;
            float dkk = 0.0;
            for (uint i = 0u; i < dk; i++) {
                float v = skeys[tid * dk + i];
                kk += v * v;
                dq += v * sq[i];
                dkk += v * sk[i];
            }
            float nj = rsqrt(kk + 1e-6);
            snj[tid] = nj;
            sdq[tid] = dq * nj;
            sdk[tid] = dkk * nj;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < dv) {
            for (uint e = 0u; e < rows; e++) {
                float g = exp(sglog[e]);
                float u = asfloat(pie_pool_get(su[e] + tid));
                if (base + int(e) < commit) {
                    float ku = snj[e] * u;
                    [ForceUnroll] for (uint i = 0; i < uint(PIE_DK); i++) {
                        st[i] = g * st[i] + skeys[e * dk + i] * ku;
                    }
                } else {
                    aq = g * aq + sdq[e] * u;
                    ak = g * ak + sdk[e] * u;
                    decay *= g;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < dv) {
        float xk = 0.0;
        float xq = 0.0;
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK); i++) {
            xk += st[i] * sk[i];
            xq += st[i] * sq[i];
        }
        float g = exp(glog_own);
        float u = (bf16_at(qkv, vbase + tid) - g * (decay * xk + ak)) * beta;
        y[(row * vh + hv) * dv + tid] = g * (decay * xq + aq) + kq * u;
        if (commit == n + 1) {
            [ForceUnroll] for (uint i = 0; i < uint(PIE_DK); i++) {
                st[i] = g * st[i] + sk[i] * u;
            }
        }
        if (commit > 0) {
#define GDN_CARRIED_STORE(buf_) \
            [ForceUnroll] for (uint i = 0; i < uint(PIE_DK); i++) { \
                buf_[cell + i] = asuint(st[i]); \
            }
            PIE_POOL_BRANCH(seg, GDN_CARRIED_STORE)
        }
        if (cl.at >= 0) {
            pie_pool_set(carry_word(cl, uint(cl.at), params.u_at, params.u_row) + hv * dv + tid, asuint(u));
        }
    }
    // The own row's keys (its key head's first value head lands them) and
    // its `[b | a]` (the first value head's), raw, a word at a time.
    if (cl.at >= 0) {
        uint at = uint(cl.at);
        if (hv % per_key == 0u) {
            uint into = carry_word(cl, at, params.k_at, params.k_row) + hk * half_dk;
            for (uint w = tid; w < half_dk; w += uint(PIE_GROUP_X)) {
                pie_pool_set(into + w, qkv[kbase / 2u + w]);
            }
        }
        if (hv == 0u) {
            uint into = carry_word(cl, at, params.ba_at, params.ba_row);
            for (uint w = tid; w < vh; w += uint(PIE_GROUP_X)) {
                pie_pool_set(into + w, ba[ba_row / 2u + w]);
            }
        }
    }
}

// pie:instantiate ssm_gdn_carried_r128_bf16 PIE_GROUP_X=128 PIE_DK=128
// pie:instantiate ssm_gdn_carried_r128_bf16 PIE_GROUP_X=128 PIE_DK=128 PIE_TIER=subgroup PIE_TARGET=spirv
