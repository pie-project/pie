#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 128
#endif

#define PIE_DMAX 256

#if defined(PIE_COMMITTED)
PIE_BUFFER_RO(0, mixed);
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
PIE_BUFFER_RO(5, f);
PIE_BUFFER_RO(6, b);
[[vk::binding(7)]] StructuredBuffer<float> dt_bias;
[[vk::binding(8)]] StructuredBuffer<float> a_log;
[[vk::binding(9)]] RWStructuredBuffer<float> rstate;
[[vk::binding(10)]] RWStructuredBuffer<float> work;
[[vk::binding(11)]] RWStructuredBuffer<float> y;
#define PIE_STATE work
#elif defined(PIE_CHUNKED)
PIE_BUFFER_RO(0, mixed);
[[vk::binding(1)]] StructuredBuffer<int> indptr;
PIE_BUFFER_RO(2, f);
PIE_BUFFER_RO(3, b);
[[vk::binding(4)]] StructuredBuffer<float> dt_bias;
[[vk::binding(5)]] StructuredBuffer<float> a_log;
[[vk::binding(6)]] RWStructuredBuffer<float> rstate;
[[vk::binding(7)]] StructuredBuffer<uint> slots;
[[vk::binding(8)]] RWStructuredBuffer<float> y;
#else
PIE_BUFFER_RO(0, mixed);
PIE_BUFFER_RO(1, f);
PIE_BUFFER_RO(2, b);
[[vk::binding(3)]] StructuredBuffer<float> dt_bias;
[[vk::binding(4)]] StructuredBuffer<float> a_log;
[[vk::binding(5)]] RWStructuredBuffer<float> rstate;
[[vk::binding(6)]] StructuredBuffer<uint> slots;
[[vk::binding(7)]] RWStructuredBuffer<float> y;
#endif
#ifndef PIE_STATE
#define PIE_STATE rstate
#endif

struct Push {
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int heads;
    int head_dim;
    float norm_eps;
    float gate_floor;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float sq[PIE_DMAX];
groupshared float sk[PIE_DMAX];
groupshared float sg[PIE_DMAX];

void token(uint tid, uint t, uint h, uint state_base) {
    uint d = uint(pc.head_dim);
    uint wide = uint(pc.heads) * d;
    uint row = t * 3u * wide;
    uint head = h * d;

    float qsum = 0.0;
    float ksum = 0.0;
    for (uint i = tid; i < d; i += uint(PIE_GROUP_X)) {
        float qv = PIE_LOAD(mixed[row + head + i]);
        float kv = PIE_LOAD(mixed[row + wide + head + i]);
        qsum += qv * qv;
        ksum += kv * kv;
    }
    float qtot = pie_workgroup_sum(qsum);
    float ktot = pie_workgroup_sum(ksum);
    float qinv = rsqrt(qtot + pc.norm_eps) * rsqrt(float(d));
    float kinv = rsqrt(ktot + pc.norm_eps);

    float alpha = exp(a_log[h]);
    for (uint i = tid; i < d; i += uint(PIE_GROUP_X)) {
        uint at = head + i;
        sq[i] = PIE_LOAD(mixed[row + at]) * qinv;
        sk[i] = PIE_LOAD(mixed[row + wide + at]) * kinv;
        float z = PIE_LOAD(f[t * wide + at]) + dt_bias[at];
        if (pc.gate_floor != 0.0) {
            sg[i] = exp(pc.gate_floor / (1.0 + exp(-alpha * z)));
        } else {
            float sp = (z > 20.0) ? z : log(1.0 + exp(z));
            sg[i] = exp(-alpha * sp);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float beta = 1.0 / (1.0 + exp(-PIE_LOAD(b[t * uint(pc.heads) + h])));
    uint out = t * wide + head;
    uint vbase = row + 2u * wide + head;
    for (uint vi = tid; vi < d; vi += uint(PIE_GROUP_X)) {
        uint cell = state_base + vi * d;
        float mem = 0.0;
        for (uint i = 0; i < d; i++) {
            float s = PIE_STATE[cell + i] * sg[i];
            PIE_STATE[cell + i] = s;
            mem += s * sk[i];
        }
        float delta = (PIE_LOAD(mixed[vbase + vi]) - mem) * beta;
        float acc = 0.0;
        for (uint i = 0; i < d; i++) {
            float s = PIE_STATE[cell + i] + sk[i] * delta;
            PIE_STATE[cell + i] = s;
            acc += s * sq[i];
        }
        y[out + vi] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint tid = local.x;
    uint h = group.y;
    uint d = uint(pc.head_dim);
    uint cells = d * d;
#if defined(PIE_COMMITTED)
    uint r = group.z;
    int begin = indptr[r];
    for (uint j = 0; j < r; j++) {
        begin += replay[uint(pc.lane0) + j];
    }
    int span = (indptr[r + 1] - indptr[r]) + replay[uint(pc.lane0) + r];
    if (span <= 0) {
        return;
    }
    int slot = slots[uint(pc.lane0) + r];
    if (slot < 0) {
        return;
    }
    int keep = min(commit[uint(pc.lane0) + r], span);
    uint bank = (uint(slot) * uint(pc.heads) + h) * cells;
    uint state_base = ((uint(pc.lane0) + r) * uint(pc.heads) + h) * cells;
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
    uint r = group.z;
    int begin = indptr[r];
    int end = indptr[r + 1];
    if (end <= begin) {
        return;
    }
    uint state_base = (slots[uint(begin)] * uint(pc.heads) + h) * cells;
    for (int t = begin; t < end; t++) {
        token(tid, uint(t), h, state_base);
    }
#else
    uint n = group.z;
    uint state_base = (slots[n] * uint(pc.heads) + h) * cells;
    token(tid, n, h, state_base);
#endif
}

// pie:instantiate kda_step_bf16 PIE_GROUP_X=128
// pie:instantiate kda_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate kda_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
