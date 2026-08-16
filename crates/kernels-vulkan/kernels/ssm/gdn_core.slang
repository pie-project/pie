// Fused GDN core for decode.
//
// Ported from Metal's `gdn_core.metal`: the conv/silu prologue, q/k L2 norms,
// gated recurrent update, and ping-pong conv-state writeback are unchanged. The
// Vulkan port deliberately reduces through shared memory over the 32 x-lanes so
// correctness does not depend on the implementation's subgroup width.


#include "common/bf16.slang"
#include "ssm/gdn_params.slang"


PIE_BUFFER_RO(0, mixed);
[[vk::binding(1)]] StructuredBuffer<float> conv_state;
[[vk::binding(2)]] RWStructuredBuffer<float> rstate;
PIE_BUFFER_RW(3, core_out);
PIE_BUFFER_RO(4, conv_w);
PIE_BUFFER_RO(5, conv_b);
[[vk::binding(6)]] StructuredBuffer<float> A_log;
PIE_BUFFER_RO(7, dt_bias);
PIE_BUFFER_RO(8, a_gate);
PIE_BUFFER_RO(9, b_gate);
[[vk::binding(10)]] RWStructuredBuffer<float> new_conv_state;
PIE_PARAMS(11, GdnCoreParams, p);
#if defined(PIE_SLOTTED)
[[vk::binding(12)]] StructuredBuffer<uint> slot_ids;
#endif

groupshared float sh_reduce[4 * 32];

float silu(float x) { return x / (1.0 + exp(-x)); }

// The workgroup ids arrive as parameters rather than as builtins, because
// Slang puts the thread indices on the entrypoint's signature and a helper is
// not the entrypoint. Same values, stated at the one place they enter the
// shader.
float row_sum32(float v, uint lx, uint ly) {
    uint at = ly * 32u + lx;
    sh_reduce[at] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16u; stride > 0u; stride >>= 1) {
        if (lx < stride) sh_reduce[at] += sh_reduce[at + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    float outv = sh_reduce[ly * 32u];
    GroupMemoryBarrierWithGroupSync();
    return outv;
}

float convsilu(int slot, int b_idx, int c) {
    float acc = PIE_LOAD(conv_b[c]);
    for (int j = 0; j < p.Kc - 1; ++j)
        acc += conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c] * PIE_LOAD(conv_w[c * p.Kc + j]);
    acc += PIE_LOAD(mixed[b_idx * p.conv_dim + c]) * PIE_LOAD(conv_w[c * p.Kc + (p.Kc - 1)]);
    return silu(acc);
}

void write_conv(int slot, int b_idx, int c) {
    for (int j = 0; j < p.Kc - 1; ++j)
        new_conv_state[(slot * p.Kc + j) * p.conv_dim + c] =
            conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c];
    new_conv_state[(slot * p.Kc + (p.Kc - 1)) * p.conv_dim + c] =
        PIE_LOAD(mixed[b_idx * p.conv_dim + c]);
}

[numthreads(32, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint3 tid : SV_DispatchThreadID) {
    int n = int(group.z);
    int b_idx = n / p.Hv;
    int hv_idx = n % p.Hv;
    int rep = p.Hv / p.Hk;
    int hk_idx = hv_idx / rep;
    bool hk_first = (hv_idx % rep) == 0;
    int dk_idx = int(local.x);
    int dv_idx = int(tid.y);
    int n_per_t = p.Dk / 32;
#if defined(PIE_SLOTTED)
    int slot = int(slot_ids[b_idx]);
#else
    int slot = b_idx;
#endif

    float qraw[8];
    float kraw[8];
    float qsq = 0.0;
    float ksq = 0.0;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        qraw[i] = convsilu(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
        kraw[i] = convsilu(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
        qsq += qraw[i] * qraw[i];
        ksq += kraw[i] * kraw[i];
    }
    float qinv = p.inv_sqrt_dk / sqrt(row_sum32(qsq, local.x, local.y) + p.eps);
    float kinv = 1.0 / sqrt(row_sum32(ksq, local.x, local.y) + p.eps);

    float vval = convsilu(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
    float ad = PIE_LOAD(a_gate[b_idx * p.Hv + hv_idx]) + PIE_LOAD(dt_bias[hv_idx]);
    float sp = max(ad, 0.0) + log(1.0 + exp(-abs(ad)));
    float decay = exp(-exp(A_log[hv_idx]) * sp);
    float beta = 1.0 / (1.0 + exp(-PIE_LOAD(b_gate[b_idx * p.Hv + hv_idx])));

    uint state_base = uint(((slot * p.Hv + hv_idx) * p.Dv + dv_idx) * p.Dk);
    float st[8];
    float kv = 0.0;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        st[i] = rstate[state_base + uint(d)] * decay;
        kv += st[i] * (kraw[i] * kinv);
    }
    kv = row_sum32(kv, local.x, local.y);
    float delta = (vval - kv) * beta;
    float outv = 0.0;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        float kk = kraw[i] * kinv;
        st[i] += kk * delta;
        outv += st[i] * (qraw[i] * qinv);
        rstate[state_base + uint(d)] = st[i];
    }
    outv = row_sum32(outv, local.x, local.y);
    if (dk_idx == 0) core_out[(b_idx * p.Hv + hv_idx) * p.Dv + dv_idx] = PIE_STORE(outv);

    if (dv_idx == 0 && hk_first) {
        for (int i = 0; i < n_per_t; ++i) {
            int d = n_per_t * dk_idx + i;
            write_conv(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
            write_conv(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
        }
    }
    write_conv(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
}

// pie:instantiate gdn_core_bfloat16
// pie:instantiate gdn_core_slotted_bfloat16 PIE_SLOTTED=1
