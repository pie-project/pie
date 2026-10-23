#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// Kimi delta attention, token by token: a workgroup a head of a row (or of a
// request), its `head_dim x head_dim` f32 state walked one token at a time.
// The delta rule of `ssm/gdn_recurrent.slang` with a gate per key channel:
// `g_i = exp(-exp(a_log) softplus(f_i + dt_bias_i))`, or with a gate floor
// `g_i = exp(floor * sigmoid(exp(a_log) (f_i + dt_bias_i)))`, and `beta =
// sigmoid(b)`. The query and key are L2-normalized with `norm_eps`, the
// query also scaled by `1 / sqrt(head_dim)`. `mixed` is the post-convolution
// `[q | k | v]` row, `f` the forget projection (a head plane), `b` one entry
// a head.
//
// The default form is one decode step a row; `PIE_CHUNKED` walks a request's
// rows of the query CSR; `PIE_COMMITTED` walks a speculative window on a copy
// of the slot's state in `work`, and the slot takes the state as of row
// `commit[lane0 + r]` (`ssm/conv1d.slang` says how). The `subgroup` tier
// folds the norms through subgroups.

#define DMAX 256

#if defined(PIE_COMMITTED)
[[vk::binding(0)]] StructuredBuffer<uint> mixed;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] StructuredBuffer<uint> f;
[[vk::binding(6)]] StructuredBuffer<uint> b;
[[vk::binding(7)]] StructuredBuffer<float> dt_bias;
[[vk::binding(8)]] StructuredBuffer<float> a_log;
[[vk::binding(9)]] RWStructuredBuffer<float> rstate;
[[vk::binding(10)]] RWStructuredBuffer<float> work;
[[vk::binding(11)]] RWStructuredBuffer<float> y;
#define PARAMS_AT 12
#define STATE work
#elif defined(PIE_CHUNKED)
[[vk::binding(0)]] StructuredBuffer<uint> mixed;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<uint> f;
[[vk::binding(3)]] StructuredBuffer<uint> b;
[[vk::binding(4)]] StructuredBuffer<float> dt_bias;
[[vk::binding(5)]] StructuredBuffer<float> a_log;
[[vk::binding(6)]] RWStructuredBuffer<float> rstate;
[[vk::binding(7)]] StructuredBuffer<uint> slots;
[[vk::binding(8)]] RWStructuredBuffer<float> y;
#define PARAMS_AT 9
#define STATE rstate
#else
[[vk::binding(0)]] StructuredBuffer<uint> mixed;
[[vk::binding(1)]] StructuredBuffer<uint> f;
[[vk::binding(2)]] StructuredBuffer<uint> b;
[[vk::binding(3)]] StructuredBuffer<float> dt_bias;
[[vk::binding(4)]] StructuredBuffer<float> a_log;
[[vk::binding(5)]] RWStructuredBuffer<float> rstate;
[[vk::binding(6)]] StructuredBuffer<uint> slots;
[[vk::binding(7)]] RWStructuredBuffer<float> y;
#define PARAMS_AT 8
#define STATE rstate
#endif

struct Params {
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int heads;
    int head_dim;
    float norm_eps;
    float gate_floor;
};
PIE_PARAMS(PARAMS_AT, Params, params);

groupshared float sq[DMAX];
groupshared float sk[DMAX];
groupshared float sg[DMAX];
groupshared int kda_lo;
groupshared int kda_hi;
groupshared int kda_keep;

// Token `t` of head `h`, its state at `state_base`.
void token(uint tid, uint t, uint h, uint state_base) {
    uint d = uint(params.head_dim);
    uint wide = uint(params.heads) * d;
    uint row = t * 3u * wide;
    uint head = h * d;

    float qsum = 0.0;
    float ksum = 0.0;
    for (uint i = tid; i < d; i += uint(PIE_GROUP_X)) {
        float qv = bf16_at(mixed, row + head + i);
        float kv = bf16_at(mixed, row + wide + head + i);
        qsum += qv * qv;
        ksum += kv * kv;
    }
    float qtot = pie_workgroup_sum(qsum, tid);
    float ktot = pie_workgroup_sum(ksum, tid);
    float qinv = rsqrt(qtot + params.norm_eps) * rsqrt(float(d));
    float kinv = rsqrt(ktot + params.norm_eps);

    float alpha = exp(a_log[h]);
    for (uint i = tid; i < d; i += uint(PIE_GROUP_X)) {
        uint at = head + i;
        sq[i] = bf16_at(mixed, row + at) * qinv;
        sk[i] = bf16_at(mixed, row + wide + at) * kinv;
        float z = bf16_at(f, t * wide + at) + dt_bias[at];
        if (params.gate_floor != 0.0) {
            sg[i] = exp(params.gate_floor / (1.0 + exp(-alpha * z)));
        } else {
            float sp = (z > 20.0) ? z : log(1.0 + exp(z));
            sg[i] = exp(-alpha * sp);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float beta = 1.0 / (1.0 + exp(-bf16_at(b, t * uint(params.heads) + h)));
    uint out = t * wide + head;
    uint vbase = row + 2u * wide + head;
    for (uint vi = tid; vi < d; vi += uint(PIE_GROUP_X)) {
        uint cell = state_base + vi * d;
        float mem = 0.0;
        for (uint i = 0; i < d; i++) {
            float s = STATE[cell + i] * sg[i];
            STATE[cell + i] = s;
            mem += s * sk[i];
        }
        float delta = (bf16_at(mixed, vbase + vi) - mem) * beta;
        float acc = 0.0;
        for (uint i = 0; i < d; i++) {
            float s = STATE[cell + i] + sk[i] * delta;
            STATE[cell + i] = s;
            acc += s * sq[i];
        }
        y[out + vi] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint tid = local.x;
    uint h = group.y;
    uint d = uint(params.head_dim);
    uint cells = d * d;
    uint r = group.z;
#if defined(PIE_COMMITTED)
    uint lane = uint(params.lane0) + r;
    if (tid == 0u) {
        int begin = indptr[r];
        for (uint j = 0; j < r; j++) {
            begin += replay[uint(params.lane0) + j];
        }
        int span = (indptr[r + 1] - indptr[r]) + replay[lane];
        kda_lo = begin;
        kda_hi = slots[lane] < 0 ? begin : begin + span;
        kda_keep = min(commit[lane], span);
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: WGSL lets a barrier follow only workgroup-uniform
    // control flow, and these bound the token loop and its commit.
    int begin = workgroupUniformLoad(kda_lo);
    int span = workgroupUniformLoad(kda_hi) - begin;
    int keep = workgroupUniformLoad(kda_keep);
    if (span <= 0) {
        return;
    }
    uint bank = (uint(slots[lane]) * uint(params.heads) + h) * cells;
    uint state_base = (lane * uint(params.heads) + h) * cells;
    for (uint i = tid; i < cells; i += uint(PIE_GROUP_X)) {
        work[state_base + i] = rstate[bank + i];
    }
    GroupMemoryBarrierWithGroupSync();
    for (int t = 0; t < span; t++) {
        token(tid, uint(begin + t), h, state_base);
        if (t + 1 == keep) {
            for (uint i = tid; i < cells; i += uint(PIE_GROUP_X)) {
                rstate[bank + i] = work[state_base + i];
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
#elif defined(PIE_CHUNKED)
    if (tid == 0u) {
        kda_lo = indptr[r];
        kda_hi = indptr[r + 1];
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: they bound the token loop.
    int begin = workgroupUniformLoad(kda_lo);
    int end = workgroupUniformLoad(kda_hi);
    if (end <= begin) {
        return;
    }
    uint state_base = (slots[uint(begin)] * uint(params.heads) + h) * cells;
    for (int t = begin; t < end; t++) {
        token(tid, uint(t), h, state_base);
    }
#else
    token(tid, r, h, (slots[r] * uint(params.heads) + h) * cells);
#endif
}

// pie:instantiate ssm_kda_bf16 PIE_GROUP_X=128
// pie:instantiate ssm_kda_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate ssm_kda_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
// pie:instantiate ssm_kda_bf16 PIE_GROUP_X=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_kda_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate ssm_kda_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_TIER=subgroup PIE_TARGET=spirv
