#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 128
#endif

#define PIE_KMAX 256

#if defined(PIE_COMMITTED)

PIE_BUFFER_RO(0, qkv);
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] StructuredBuffer<float> gates;
[[vk::binding(6)]] RWStructuredBuffer<float> rstate;
[[vk::binding(7)]] RWStructuredBuffer<float> work;
[[vk::binding(8)]] RWStructuredBuffer<float> y;
#define PIE_STATE work
#elif defined(PIE_CHUNKED)
PIE_BUFFER_RO(0, qkv);
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<float> gates;
[[vk::binding(3)]] RWStructuredBuffer<float> rstate;
[[vk::binding(4)]] StructuredBuffer<uint> slots;
[[vk::binding(5)]] RWStructuredBuffer<float> y;
#else
PIE_BUFFER_RO(0, qkv);
[[vk::binding(1)]] StructuredBuffer<float> gates;
[[vk::binding(2)]] RWStructuredBuffer<float> rstate;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] RWStructuredBuffer<float> y;
#endif
#ifndef PIE_STATE
#define PIE_STATE rstate
#endif

struct Push {
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int k_heads;
    int v_heads;
    int k_dim;
    int v_dim;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float sq[PIE_KMAX];
groupshared float sk[PIE_KMAX];

#if defined(PIE_DK_REG)

void token(uint tid, uint t, uint hv, uint hk, uint state_base, inout float st[PIE_DK_REG]) {
#else
void token(uint tid, uint t, uint hv, uint hk, uint state_base) {
#endif
    uint dk = uint(pc.k_dim);
    uint dv = uint(pc.v_dim);
    uint keys = uint(pc.k_heads) * dk;
    uint pitch = 2u * keys + uint(pc.v_heads) * dv;
    uint row = t * pitch;
    uint qbase = row + hk * dk;
    uint kbase = qbase + keys;
    uint vbase = row + 2u * keys + hv * dv;

    float qsum = 0.0;
    float ksum = 0.0;
    for (uint i = tid; i < dk; i += uint(PIE_GROUP_X)) {
        float qv = PIE_LOAD(qkv[qbase + i]);
        float kv = PIE_LOAD(qkv[kbase + i]);
        sq[i] = qv;
        sk[i] = kv;
        qsum += qv * qv;
        ksum += kv * kv;
    }
    float qtot = pie_workgroup_sum(qsum);
    float ktot = pie_workgroup_sum(ksum);
    float scale = 1.0 / sqrt(float(dk));
    float qinv = rsqrt(qtot + 1e-6) * scale;
    float kinv = rsqrt(ktot + 1e-6);
    for (uint i = tid; i < dk; i += uint(PIE_GROUP_X)) {
        sq[i] *= qinv;
        sk[i] *= kinv;
    }
    GroupMemoryBarrierWithGroupSync();

    uint fused = t * 2u * uint(pc.v_heads) + hv;
    float decay = exp(gates[fused]);
    float beta = gates[fused + uint(pc.v_heads)];
    uint out = (t * uint(pc.v_heads) + hv) * dv;
#if defined(PIE_DK_REG)
    if (tid < dv) {
        uint c = tid;
        float kv_mem = 0.0;
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            st[i] *= decay;
            kv_mem += st[i] * sk[i];
        }
        float delta = (PIE_LOAD(qkv[vbase + c]) - kv_mem) * beta;
        float acc = 0.0;
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            st[i] += sk[i] * delta;
            acc += st[i] * sq[i];
        }
        y[out + c] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
}
#else
    for (uint c = tid; c < dv; c += uint(PIE_GROUP_X)) {
        uint cell = state_base + c * dk;
        float kv_mem = 0.0;
        for (uint i = 0; i < dk; i++) {
            float s = PIE_STATE[cell + i] * decay;
            PIE_STATE[cell + i] = s;
            kv_mem += s * sk[i];
        }
        float delta = (PIE_LOAD(qkv[vbase + c]) - kv_mem) * beta;
        float acc = 0.0;
        for (uint i = 0; i < dk; i++) {
            float s = PIE_STATE[cell + i] + sk[i] * delta;
            PIE_STATE[cell + i] = s;
            acc += s * sq[i];
        }
        y[out + c] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint tid = local.x;
    uint hv = group.y;
    uint hk = hv / (uint(pc.v_heads) / uint(pc.k_heads));
    uint head = uint(pc.v_dim) * uint(pc.k_dim);
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
    uint dk = uint(pc.k_dim);
    uint dv = uint(pc.v_dim);
    uint bank = (uint(slot) * uint(pc.v_heads) + hv) * head;
    uint state_base = ((uint(pc.lane0) + r) * uint(pc.v_heads) + hv) * head;

    for (uint c = tid; c < dv; c += uint(PIE_GROUP_X)) {
        for (uint i = 0; i < dk; i++) {
            work[state_base + c * dk + i] = rstate[bank + c * dk + i];
        }
    }
    GroupMemoryBarrierWithGroupSync();
#if defined(PIE_DK_REG)
    if (dk != uint(PIE_DK_REG) || dv > uint(PIE_GROUP_X)) {
        return;
    }
    float st[PIE_DK_REG];
    [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
        st[i] = tid < dv ? rstate[bank + tid * dk + i] : 0.0;
    }
    for (int t = 0; t < span; t++) {
        token(tid, uint(begin + t), hv, hk, state_base, st);
        if (t + 1 == keep && tid < dv) {
            [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
                rstate[bank + tid * dk + i] = st[i];
            }
        }
    }
#else
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
    uint r = group.z;
    int begin = indptr[r];
    int end = indptr[r + 1];
    if (end <= begin) {
        return;
    }
    uint state_base = (slots[uint(begin)] * uint(pc.v_heads) + hv) * head;
#if defined(PIE_DK_REG)
    uint dk = uint(pc.k_dim);
    uint dv = uint(pc.v_dim);
    if (dk != uint(PIE_DK_REG) || dv > uint(PIE_GROUP_X)) {
        return;
    }
    float st[PIE_DK_REG];
    [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
        st[i] = tid < dv ? rstate[state_base + tid * dk + i] : 0.0;
    }
    for (int t = begin; t < end; t++) {
        token(tid, uint(t), hv, hk, state_base, st);
    }
    if (tid < dv) {
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            rstate[state_base + tid * dk + i] = st[i];
        }
    }
#else
    for (int t = begin; t < end; t++) {
        token(tid, uint(t), hv, hk, state_base);
    }
#endif
#else
    uint n = group.z;
    uint state_base = (slots[n] * uint(pc.v_heads) + hv) * head;
#if defined(PIE_DK_REG)
    uint dk = uint(pc.k_dim);
    uint dv = uint(pc.v_dim);
    if (dk != uint(PIE_DK_REG) || dv > uint(PIE_GROUP_X)) {
        return;
    }
    float st[PIE_DK_REG];
    [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
        st[i] = tid < dv ? rstate[state_base + tid * dk + i] : 0.0;
    }
    token(tid, n, hv, hk, state_base, st);
    if (tid < dv) {
        [ForceUnroll] for (uint i = 0; i < uint(PIE_DK_REG); i++) {
            rstate[state_base + tid * dk + i] = st[i];
        }
    }
#else
    token(tid, n, hv, hk, state_base);
#endif
#endif
}

// pie:instantiate gated_delta_bf16 PIE_GROUP_X=128
// pie:instantiate gated_delta_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate gated_delta_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
// pie:instantiate gated_delta_r128_bf16 PIE_GROUP_X=128 PIE_DK_REG=128
// pie:instantiate gated_delta_chunked_r128_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_DK_REG=128
// pie:instantiate gated_delta_committed_r128_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_DK_REG=128
