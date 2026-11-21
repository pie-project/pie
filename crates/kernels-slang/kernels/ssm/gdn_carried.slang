#include "prim/bf16.slang"
#include "prim/params.slang"
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
// PIE_GROUP_X = PIE_DK`). The own row's gates are formed from its `[b | a]`
// as `ssm/gdn_prep.slang` forms them, a buffered row's from its stored pair.
// The own row's q and k stay raw; their norms are applied as scalars. A chunk's
// rows are resolved once a row (their page's word and their token in it,
// under a barrier the step takes anyway), then staged one load an element,
// all in flight together; the first chunk is staged before the own row's
// sums, so its loads overlap them. A chunk's rows' dots take `DOT_LANES`
// lanes a row.

#define CH 16u
#define DOT_LANES 8u
#define HALF_DK (PIE_DK / 2)

[[vk::binding(0)]] StructuredBuffer<uint> qkv;
[[vk::binding(1)]] StructuredBuffer<uint> ba;
[[vk::binding(2)]] StructuredBuffer<uint> dt_bias;
[[vk::binding(3)]] StructuredBuffer<float> a_log;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] StructuredBuffer<int> carry;
[[vk::binding(6)]] RWStructuredBuffer<float> y;
// The state bank and the rs list, in the chunk pool's segments, as vectors
// of four words (every segment, the state's origin and slot stride whole
// vectors: the host refuses another pool).
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint4, 7)

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

PIE_POOL_READ(uint4, params.seg_words / 4u)
PIE_STATE_PIECES

// Virtual word `v` of the pool.
uint pool_word(uint v) {
    return pie_pool_get(v >> 2)[v & 3u];
}

void pool_set_word(uint v, uint x) {
    uint seg, at;
    pie_pool_find(v >> 2, seg, at);
#define GDN_WORD_SET(buf_) buf_[at][v & 3u] = x;
    PIE_POOL_BRANCH(seg, GDN_WORD_SET)
}

#define PIE_CARRY_POOL_WORD(v) pool_word(v)
#include "prim/carry.slang"

// The own row's raw q and k.
groupshared float sq[PIE_DK];
groupshared float sk[PIE_DK];
// A chunk's rows: keys (raw bf16 pairs), `u` (the head's columns) and
// log-gates; then each row's key norm and its dots with the own q and k.
groupshared uint skeys[CH * HALF_DK];
groupshared float su[CH * PIE_GROUP_X];
groupshared float sglog[CH];
groupshared uint spage[CH];
groupshared uint stok[CH];
groupshared float snj[CH];
groupshared float sdq[CH];
groupshared float sdk[CH];
// Partial sums: the own row's three, or a chunk's rows' dots.
groupshared float3 red[PIE_GROUP_X];
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

float2 unpack2(uint word) {
    return float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
}

// The workgroup's sum of `v`, three at once.
float3 sum3(float3 v, uint tid) {
#if defined(PIE_SUBGROUP)
    float3 s = WaveActiveSum(v);
    uint width = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u) {
        red[tid / width] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    float3 total = float3(0.0);
    for (uint i = 0u; i < (uint(PIE_GROUP_X) + width - 1u) / width; i++) {
        total += red[i];
    }
#else
    red[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint half = uint(PIE_GROUP_X) / 2u; half > 0u; half >>= 1u) {
        if (tid < half) {
            red[tid] += red[tid + half];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float3 total = red[0];
#endif
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// A lane's share of a chunk's stage, held in registers while it lands:
// its keys words, its column's `u` a row, and (a lane a row) a log-gate.
#define KEYS_PER_LANE (CH * uint(HALF_DK) / uint(PIE_GROUP_X))
struct Staged {
    uint keys[KEYS_PER_LANE];
    float u[CH];
    // The word holding its row's `a` (a lane a row), its gate formed as it
    // lands, so no lane waits on its loads before then.
    uint ba;
};

// Rows `base .. base + rows` of the lane's buffered rows resolved, a lane
// a row: the row's page's first word and its token in the page.
void resolve(CarryLane cl, uint base, uint rows, uint tid) {
    if (tid < rows) {
        uint t = carry_tok(cl, base + tid);
        spage[tid] = carry_page(cl, t) * params.stride_words;
        stok[tid] = t % params.page_tokens;
    }
}

// The first word of plane (`at`, `row`) of resolved row `e`.
uint row_word(uint e, uint at, uint row) {
    return spage[e] + at + stok[e] * row;
}

// The resolved rows (`resolve`, a barrier since), this lane's share: one
// load an element, every one in flight together, nothing waited on until
// `land`.
Staged fetch(uint rows, uint tid, uint hk, uint hv, uint vh, uint dv) {
    Staged out;
    [ForceUnroll] for (uint k = 0u; k < KEYS_PER_LANE; k++) {
        uint idx = tid + k * uint(PIE_GROUP_X);
        uint e = idx / uint(HALF_DK);
        out.keys[k] = e < rows
            ? pool_word(row_word(e, params.k_at, params.k_row) + hk * uint(HALF_DK) + idx % uint(HALF_DK))
            : 0u;
    }
    [ForceUnroll] for (uint e = 0u; e < CH; e++) {
        out.u[e] = (e < rows && tid < dv)
            ? asfloat(pool_word(row_word(e, params.u_at, params.u_row) + hv * dv + tid))
            : 0.0;
    }
    out.ba = tid < rows
        ? pool_word(row_word(tid, params.ba_at, params.ba_row) + (vh + hv) / 2u)
        : 0u;
    return out;
}

// This lane's share of the chunk into the stage.
void land(Staged staged, uint rows, uint tid, uint hv, uint vh, uint dv) {
    [ForceUnroll] for (uint k = 0u; k < KEYS_PER_LANE; k++) {
        skeys[tid + k * uint(PIE_GROUP_X)] = staged.keys[k];
    }
    if (tid < dv) {
        [ForceUnroll] for (uint e = 0u; e < CH; e++) {
            su[e * uint(PIE_GROUP_X) + tid] = staged.u[e];
        }
    }
    if (tid < rows) {
        uint word = staged.ba;
        sglog[tid] = log_gate(bf16_to_f32(((vh + hv) & 1u) != 0u ? (word >> 16) : (word & 0xffffu)), hv);
    }
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
    // The lane's column of the committed state, found once (a state row
    // lies in one slot, so in one segment) and loaded first: its loads, the
    // step's largest, overlap all that follows. A lane naming no record
    // loads nothing and returns below.
    bool none = pie_state_none(lane);
    CarryLane cl = carry_lane(lane);
    if (tid == 0u) {
        g_n = cl.n;
        g_commit = min(cl.commit, cl.n + 1);
        g_none = none ? 1 : 0;
    }
    // The first chunk's rows resolve under this barrier; their loads go out
    // right after it and land in the stage after the own row's sums.
    resolve(cl, 0u, uint(clamp(cl.n, 0, int(CH))), tid);
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: WGSL lets a barrier follow only workgroup-uniform
    // control flow, and these bound the loops.
    int n = workgroupUniformLoad(g_n);
    int commit = workgroupUniformLoad(g_commit);
    if (workgroupUniformLoad(g_none) != 0) {
        return;
    }
    Staged first = fetch(uint(min(int(CH), n)), tid, hk, hv, vh, dv);
    // The lane's column of the committed state (a slot's, so in one
    // segment), a vector of four key columns a load.
    uint seg = 0u;
    uint cell = 0u;
    if (tid < dv) {
        pie_pool_find((params.state_origin + pie_state_atom(lane, hv, dv * dk) + tid * dk) / 4u, seg, cell);
#define GDN_CARRIED_LOAD(buf_) \
        [ForceUnroll] for (uint m = 0u; m < uint(PIE_DK) / 4u; m++) { \
            uint4 v = buf_[cell + m]; \
            st[4u * m] = asfloat(v.x); \
            st[4u * m + 1u] = asfloat(v.y); \
            st[4u * m + 2u] = asfloat(v.z); \
            st[4u * m + 3u] = asfloat(v.w); \
        }
        PIE_POOL_BRANCH(seg, GDN_CARRIED_LOAD)
    }

#if defined(PIE_FOLD_ONLY)
    // A group's fold (BEAM-PAGE-LISTS §11.3): the lane is a beam group's
    // set, its carried rows the ones every live beam shares, all committed;
    // it has no own row, reads no input and writes no output.
    sq[tid] = 0.0;
    sk[tid] = 0.0;
    float qinv = 0.0;
    float kinv = 0.0;
    land(first, uint(min(int(CH), n)), tid, hv, vh, dv);
#else
    // The own row: raw q and k, their norms and their dot in one sum.
    uint keys = kh * dk;
    uint pitch = 2u * keys + vh * dv;
    uint qbase = row * pitch + hk * dk;
    uint kbase = qbase + keys;
    uint vbase = row * pitch + 2u * keys + hv * dv;
    float qv = bf16_at(qkv, qbase + tid);
    float kv = bf16_at(qkv, kbase + tid);
    sq[tid] = qv;
    sk[tid] = kv;
    float3 tot = sum3(float3(qv * qv, kv * kv, qv * kv), tid);
    float qinv = rsqrt(tot.x + 1e-6) / sqrt(float(dk));
    float kinv = rsqrt(tot.y + 1e-6);
    float kq = tot.z * qinv * kinv;
    land(first, uint(min(int(CH), n)), tid, hv, vh, dv);
    uint ba_row = row * 2u * vh;
    float glog_own = log_gate(bf16_at(ba, ba_row + vh + hv), hv);
    float beta = 1.0 / (1.0 + exp(-bf16_at(ba, ba_row + hv)));
#endif


    // The buffered rows, a chunk at a time: each row's norm and dots with
    // the own row's q and k (`DOT_LANES` lanes a row), then each lane walks
    // the rows in order: a committed row folds into its column, a later one
    // adds to the attention sums, every sum decaying by the rows after it.
    float decay = 1.0;
    float aq = 0.0;
    float ak = 0.0;
    for (int base = 0; base < n; base += int(CH)) {
        uint rows = uint(min(int(CH), n - base));
        if (base > 0) {
            resolve(cl, uint(base), rows, tid);
            GroupMemoryBarrierWithGroupSync();
            land(fetch(rows, tid, hk, hv, vh, dv), rows, tid, hv, vh, dv);
        }
        GroupMemoryBarrierWithGroupSync();
        {
            uint e = tid / DOT_LANES;
            uint part = tid % DOT_LANES;
            float3 s = float3(0.0);
            if (e < rows) {
                for (uint w = part; w < uint(HALF_DK); w += DOT_LANES) {
                    float2 v = unpack2(skeys[e * uint(HALF_DK) + w]);
                    s += float3(v.x * v.x + v.y * v.y,
                                v.x * sq[2u * w] + v.y * sq[2u * w + 1u],
                                v.x * sk[2u * w] + v.y * sk[2u * w + 1u]);
                }
            }
            red[tid] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < rows) {
            float3 s = float3(0.0);
            for (uint p = 0u; p < DOT_LANES; p++) {
                s += red[tid * DOT_LANES + p];
            }
            float nj = rsqrt(s.x + 1e-6);
            snj[tid] = nj;
            sdq[tid] = s.y * nj * qinv;
            sdk[tid] = s.z * nj * kinv;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid < dv) {
            for (uint e = 0u; e < rows; e++) {
                float g = exp(sglog[e]);
                float u = su[e * uint(PIE_GROUP_X) + tid];
                if (base + int(e) < commit) {
                    float ku = snj[e] * u;
                    [ForceUnroll] for (uint w = 0; w < uint(HALF_DK); w++) {
                        float2 v = unpack2(skeys[e * uint(HALF_DK) + w]);
                        st[2u * w] = g * st[2u * w] + v.x * ku;
                        st[2u * w + 1u] = g * st[2u * w + 1u] + v.y * ku;
                    }
                } else {
                    aq = g * aq + sdq[e] * u;
                    ak = g * ak + sdk[e] * u;
                    decay *= g;
                }
            }
        }
        if (base + int(CH) < n) {
            GroupMemoryBarrierWithGroupSync();
        }
    }

#if !defined(PIE_FOLD_ONLY)
    if (tid < dv) {
        float xk = 0.0;
        float xq = 0.0;
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK); i++) {
            xk += st[i] * sk[i];
            xq += st[i] * sq[i];
        }
        float g = exp(glog_own);
        float u = (bf16_at(qkv, vbase + tid) - g * (decay * xk * kinv + ak)) * beta;
        y[(row * vh + hv) * dv + tid] = g * (decay * xq * qinv + aq) + kq * u;
        if (commit == n + 1) {
            float ku = kinv * u;
            [ForceUnroll] for (uint i = 0; i < uint(PIE_DK); i++) {
                st[i] = g * st[i] + sk[i] * ku;
            }
        }
        if (cl.at >= 0) {
            pool_set_word(carry_word(cl, uint(cl.at), params.u_at, params.u_row) + hv * dv + tid, asuint(u));
        }
    }
#endif
    // The state as of the commit, a vector a store.
    if (commit > 0 && tid < dv) {
#define GDN_CARRIED_STORE(buf_) \
        [ForceUnroll] for (uint m = 0u; m < uint(PIE_DK) / 4u; m++) { \
            buf_[cell + m] = uint4(asuint(st[4u * m]), asuint(st[4u * m + 1u]), \
                                   asuint(st[4u * m + 2u]), asuint(st[4u * m + 3u])); \
        }
        PIE_POOL_BRANCH(seg, GDN_CARRIED_STORE)
    }
    // The own row's keys (its key head's first value head lands them) and
    // its `[b | a]` (the first value head's), raw, a word at a time.
#if !defined(PIE_FOLD_ONLY)
    if (cl.at >= 0) {
        uint at = uint(cl.at);
        if (hv % per_key == 0u) {
            uint into = carry_word(cl, at, params.k_at, params.k_row) + hk * uint(HALF_DK);
            for (uint w = tid; w < uint(HALF_DK); w += uint(PIE_GROUP_X)) {
                pool_set_word(into + w, qkv[kbase / 2u + w]);
            }
        }
        if (hv == 0u) {
            uint into = carry_word(cl, at, params.ba_at, params.ba_row);
            for (uint w = tid; w < vh; w += uint(PIE_GROUP_X)) {
                pool_set_word(into + w, ba[ba_row / 2u + w]);
            }
        }
    }
#endif
}

// pie:instantiate ssm_gdn_carried_r128_bf16 PIE_GROUP_X=128 PIE_DK=128
// pie:instantiate ssm_gdn_carried_r128_bf16 PIE_GROUP_X=128 PIE_DK=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_gdn_carried_fold_r128_bf16 PIE_GROUP_X=128 PIE_DK=128 PIE_FOLD_ONLY=1
// pie:instantiate ssm_gdn_carried_fold_r128_bf16 PIE_GROUP_X=128 PIE_DK=128 PIE_FOLD_ONLY=1 PIE_TIER=subgroup PIE_TARGET=spirv
