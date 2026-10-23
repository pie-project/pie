#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

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
[[vk::binding(6)]] RWStructuredBuffer<float> rstate;
[[vk::binding(7)]] RWStructuredBuffer<float> work;
[[vk::binding(8)]] RWStructuredBuffer<float> y;
#define PARAMS_AT 9
#define STATE work
#elif defined(PIE_CHUNKED)
[[vk::binding(0)]] StructuredBuffer<uint> qkv;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<float> gates;
[[vk::binding(3)]] RWStructuredBuffer<float> rstate;
[[vk::binding(4)]] StructuredBuffer<uint> slots;
[[vk::binding(5)]] RWStructuredBuffer<float> y;
#define PARAMS_AT 6
#define STATE rstate
#else
[[vk::binding(0)]] StructuredBuffer<uint> qkv;
[[vk::binding(1)]] StructuredBuffer<float> gates;
[[vk::binding(2)]] RWStructuredBuffer<float> rstate;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] RWStructuredBuffer<float> y;
#define PARAMS_AT 5
#define STATE rstate
#endif

struct Params {
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int k_heads;
    int v_heads;
    int k_dim;
    int v_dim;
};
PIE_PARAMS(PARAMS_AT, Params, params);

groupshared float sq[KMAX];
groupshared float sk[KMAX];
groupshared int gdn_lo;
groupshared int gdn_hi;
groupshared int gdn_keep;

#if defined(PIE_DK_REG)
static float st[PIE_DK_REG];
#endif

// Token `t` of value head `hv` (key head `hk`), its state at `state_base`.
void token(uint tid, uint t, uint hv, uint hk, uint state_base) {
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

#if defined(PIE_DK_REG)
void reg_load(uint tid, uint base) {
    uint dk = uint(params.k_dim);
    [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
        st[i] = tid < uint(params.v_dim) ? rstate[base + tid * dk + i] : 0.0;
    }
}

void reg_store(uint tid, uint base) {
    uint dk = uint(params.k_dim);
    if (tid < uint(params.v_dim)) {
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            rstate[base + tid * dk + i] = st[i];
        }
    }
}
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
        int begin = indptr[r];
        for (uint j = 0; j < r; j++) {
            begin += replay[uint(params.lane0) + j];
        }
        int span = (indptr[r + 1] - indptr[r]) + replay[lane];
        gdn_lo = begin;
        gdn_hi = slots[lane] < 0 ? begin : begin + span;
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
    uint bank = (uint(slots[lane]) * uint(params.v_heads) + hv) * head;
#if defined(PIE_DK_REG)
    reg_load(tid, bank);
    for (int t = 0; t < span; t++) {
        token(tid, uint(begin + t), hv, hk, 0u);
        if (t + 1 == keep) {
            reg_store(tid, bank);
        }
    }
#else
    uint state_base = (lane * uint(params.v_heads) + hv) * head;
    for (uint c = tid; c < dv; c += uint(PIE_GROUP_X)) {
        for (uint i = 0; i < dk; i++) {
            work[state_base + c * dk + i] = rstate[bank + c * dk + i];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (int t = 0; t < span; t++) {
        token(tid, uint(begin + t), hv, hk, state_base);
        if (t + 1 == keep) {
            for (uint c = tid; c < dv; c += uint(PIE_GROUP_X)) {
                for (uint i = 0; i < dk; i++) {
                    rstate[bank + c * dk + i] = work[state_base + c * dk + i];
                }
            }
        }
    }
#endif
#elif defined(PIE_CHUNKED)
    if (tid == 0u) {
        gdn_lo = indptr[r];
        gdn_hi = indptr[r + 1];
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: they bound the token loop.
    int begin = workgroupUniformLoad(gdn_lo);
    int end = workgroupUniformLoad(gdn_hi);
    if (end <= begin) {
        return;
    }
    uint state_base = (slots[uint(begin)] * uint(params.v_heads) + hv) * head;
#if defined(PIE_DK_REG)
    reg_load(tid, state_base);
#endif
    for (int t = begin; t < end; t++) {
        token(tid, uint(t), hv, hk, state_base);
    }
#if defined(PIE_DK_REG)
    reg_store(tid, state_base);
#endif
#else
    uint state_base = (slots[r] * uint(params.v_heads) + hv) * head;
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
