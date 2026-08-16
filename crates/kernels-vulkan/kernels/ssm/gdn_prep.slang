// Split GDN prep/recurrent path, including prompt prefill scan.
//
// The Metal split killed redundant q/k work by staging fp32 scratch between two
// dispatches. This port keeps the same scratch layout and the same ping-pong
// conv-state rules; reductions are expressed over explicit x-lane groups instead
// of Metal's fixed 32-wide simdgroup.


#include "common/bf16.slang"
#include "ssm/gdn_params.slang"

#if defined(PIE_RECURRENT) && !defined(PIE_SCAN)
#define PIE_NUMTHREADS [numthreads(32, 4, 1)]
#else
#define PIE_NUMTHREADS [numthreads(32, 1, 1)]
#endif

#if defined(PIE_SCAN)
[[vk::binding(0)]] RWStructuredBuffer<float> rstate;
PIE_BUFFER_RW(1, core_out);
[[vk::binding(2)]] StructuredBuffer<float> pre_q;
[[vk::binding(3)]] StructuredBuffer<float> pre_k;
[[vk::binding(4)]] StructuredBuffer<float> pre_gate;
PIE_PARAMS(5, GdnCoreParams, p);
[[vk::binding(6)]] StructuredBuffer<uint> slot_ids;
struct Push { int row_pitch; int n_scan; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#else
PIE_BUFFER_RO(0, mixed);
[[vk::binding(1)]] StructuredBuffer<float> conv_state;
#if defined(PIE_RECURRENT)
[[vk::binding(2)]] RWStructuredBuffer<float> rstate;
PIE_BUFFER_RW(3, core_out);
PIE_BUFFER_RO(4, conv_w);
PIE_BUFFER_RO(5, conv_b);
[[vk::binding(6)]] StructuredBuffer<float> pre_q;
[[vk::binding(7)]] StructuredBuffer<float> pre_k;
[[vk::binding(8)]] StructuredBuffer<float> pre_gate;
[[vk::binding(9)]] RWStructuredBuffer<float> new_conv_state;
PIE_PARAMS(10, GdnCoreParams, p);
#if defined(PIE_SLOTTED)
[[vk::binding(11)]] StructuredBuffer<uint> slot_ids;
#endif
#else
PIE_BUFFER_RO(2, conv_w);
PIE_BUFFER_RO(3, conv_b);
[[vk::binding(4)]] StructuredBuffer<float> A_log;
PIE_BUFFER_RO(5, dt_bias);
PIE_BUFFER_RO(6, a_gate);
PIE_BUFFER_RO(7, b_gate);
[[vk::binding(8)]] RWStructuredBuffer<float> pre_q;
[[vk::binding(9)]] RWStructuredBuffer<float> pre_k;
[[vk::binding(10)]] RWStructuredBuffer<float> pre_gate;
[[vk::binding(11)]] RWStructuredBuffer<float> new_conv_state;
PIE_PARAMS(12, GdnCoreParams, p);
#if defined(PIE_SLOTTED) || defined(PIE_PREFILL)
[[vk::binding(13)]] StructuredBuffer<uint> slot_ids;
#endif
#if defined(PIE_PREFILL)
struct Push { int row_pitch; int n_scan; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif
#endif
#endif

groupshared float sh_reduce[4 * 32];

float silu(float x) { return x / (1.0 + exp(-x)); }

// The thread indices arrive as parameters, because Slang puts them on the
// entrypoint's signature and a helper is not the entrypoint.
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

#if defined(PIE_SCAN)
float lane_sum(float v, uint lx) {
    uint base = (lx / uint(PIE_LANES)) * uint(PIE_LANES);
    sh_reduce[lx] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = uint(PIE_LANES) >> 1; stride > 0u; stride >>= 1) {
        uint rel = lx - base;
        if (rel < stride) sh_reduce[lx] += sh_reduce[lx + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    float outv = sh_reduce[base];
    GroupMemoryBarrierWithGroupSync();
    return outv;
}

PIE_NUMTHREADS
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint3 tid : SV_DispatchThreadID) {
    const int LANES = PIE_LANES;
    const int VROWS = PIE_VROWS;
    const int ROWS = 32 / LANES;
    int hv_idx = int(group.z);
    int dv_base = (int(group.y) * ROWS + (int(local.x) / LANES)) * VROWS;
    int dk_idx = int(local.x) % LANES;
    int slot = int(slot_ids[0]);
    int n_per_t = p.Dk / LANES;
    int pitch_f = pc.row_pitch / 2;
    bool row_lead = dk_idx == 0;
    bool active_group = dv_base < p.Dv;
    int vn = active_group ? min(VROWS, p.Dv - dv_base) : 0;

    float st[8][32];
    uint state_base = uint(((slot * p.Hv + hv_idx) * p.Dv + dv_base) * p.Dk);
    for (int v = 0; v < VROWS; ++v)
        for (int i = 0; i < 32; ++i)
            st[v][i] = (v < vn && i < n_per_t) ? rstate[state_base + uint(v * p.Dk + n_per_t * dk_idx + i)] : 0.0;

    for (int t = 0; t < pc.n_scan; ++t) {
        int row_t = t * pc.row_pitch;
        int row_f = t * pitch_f;
        float q[32];
        float k[32];
        for (int i = 0; i < 32; ++i) {
            int d = n_per_t * dk_idx + i;
            q[i] = i < n_per_t ? pre_q[row_f + hv_idx * p.Dk + d] : 0.0;
            k[i] = i < n_per_t ? pre_k[row_f + hv_idx * p.Dk + d] : 0.0;
        }
        float ga = pre_gate[row_f + 2 * hv_idx + 0];
        float gb = pre_gate[row_f + 2 * hv_idx + 1];
        float kv[8];
        for (int v = 0; v < VROWS; ++v) {
            float acc = 0.0;
            for (int i = 0; i < 32; ++i) if (i < n_per_t) { st[v][i] *= ga; acc += st[v][i] * k[i]; }
            kv[v] = lane_sum(v < vn ? acc : 0.0, local.x);
        }
        for (int v = 0; v < VROWS; ++v) {
            float vv = (v < vn) ? pre_gate[row_f + 2 * p.Hv + hv_idx * p.Dv + dv_base + v] : 0.0;
            float delta = (vv - kv[v]) * gb;
            float outv = 0.0;
            for (int i = 0; i < 32; ++i) if (i < n_per_t) { st[v][i] += k[i] * delta; outv += st[v][i] * q[i]; }
            outv = lane_sum(v < vn ? outv : 0.0, local.x);
            if (v < vn && row_lead) core_out[row_t + hv_idx * p.Dv + dv_base + v] = PIE_STORE(outv);
        }
    }
    for (int v = 0; v < VROWS; ++v)
        for (int i = 0; i < 32; ++i)
            if (v < vn && i < n_per_t) rstate[state_base + uint(v * p.Dk + n_per_t * dk_idx + i)] = st[v][i];
}

#else

float convsilu_token(int slot, int b_idx, int c) {
    float acc = PIE_LOAD(conv_b[c]);
    for (int j = 0; j < p.Kc - 1; ++j)
        acc += conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c] * PIE_LOAD(conv_w[c * p.Kc + j]);
    acc += PIE_LOAD(mixed[b_idx * p.conv_dim + c]) * PIE_LOAD(conv_w[c * p.Kc + (p.Kc - 1)]);
    return silu(acc);
}

void write_conv_token(int slot, int b_idx, int c) {
    for (int j = 0; j < p.Kc - 1; ++j)
        new_conv_state[(slot * p.Kc + j) * p.conv_dim + c] = conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c];
    new_conv_state[(slot * p.Kc + (p.Kc - 1)) * p.conv_dim + c] = PIE_LOAD(mixed[b_idx * p.conv_dim + c]);
}

#if defined(PIE_PREFILL)
float tap(int slot, int t, int j, int c) {
    int idx = t - (p.Kc - 1) + j;
    return idx >= 0 ? PIE_LOAD(mixed[idx * pc.row_pitch + c])
                    : conv_state[(slot * p.Kc + p.Kc + idx) * p.conv_dim + c];
}
float convsilu_prefill(int slot, int t, int c) {
    float acc = PIE_LOAD(conv_b[c]);
    for (int j = 0; j < p.Kc - 1; ++j) acc += tap(slot, t, j, c) * PIE_LOAD(conv_w[c * p.Kc + j]);
    acc += PIE_LOAD(mixed[t * pc.row_pitch + c]) * PIE_LOAD(conv_w[c * p.Kc + (p.Kc - 1)]);
    return silu(acc);
}
void write_conv_prefill(int slot, int t, int c) {
    for (int j = 0; j < p.Kc; ++j) {
        int idx = t - (p.Kc - 1) + j;
        new_conv_state[(slot * p.Kc + j) * p.conv_dim + c] =
            idx >= 0 ? PIE_LOAD(mixed[idx * pc.row_pitch + c]) : conv_state[(slot * p.Kc + p.Kc + idx) * p.conv_dim + c];
    }
}
#endif

PIE_NUMTHREADS
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint3 tid : SV_DispatchThreadID) {
#if defined(PIE_RECURRENT)
    int n = int(group.z);
    int b_idx = n / p.Hv;
    int hv_idx = n % p.Hv;
    int dk_idx = int(local.x);
    int dv_idx = int(tid.y);
    int n_per_t = p.Dk / 32;
    // The workgroup is 4 deep on y and Metal's `dispatchThreads` grid y is `Dv` exactly,
    // so any `Dv` not a multiple of 4 launches lanes with no value channel to
    // own. Every GDN checkpoint the tree has seen states `Dv = 128`, which is
    // why this has never fired.
    //
    // The guard is on the STORES rather than an early return, because
    // `row_sum32` barriers below and `barrier()` must sit in control flow
    // uniform across the workgroup. Returning here would leave the surviving
    // lanes waiting at a barrier their neighbours will never reach, which is
    // undefined and, on some drivers, a hang rather than a wrong number.
    // The loads are harmless: they read a neighbouring head, and the
    // arithmetic they feed is thrown away.
    bool mine = dv_idx < p.Dv;
#if defined(PIE_SLOTTED)
    int slot = int(slot_ids[b_idx]);
#else
    int slot = b_idx;
#endif
    float vval = convsilu_token(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
    uint state_base = uint(((slot * p.Hv + hv_idx) * p.Dv + dv_idx) * p.Dk);
    uint pk_base = uint(n * p.Dk);
    float kv = 0.0;
    float st[8];
    float q[8];
    float k[8];
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        q[i] = pre_q[pk_base + uint(d)];
        k[i] = pre_k[pk_base + uint(d)];
        st[i] = rstate[state_base + uint(d)] * pre_gate[2 * n + 0];
        kv += st[i] * k[i];
    }
    kv = row_sum32(kv, local.x, local.y);
    float delta = (vval - kv) * pre_gate[2 * n + 1];
    float outv = 0.0;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        st[i] += k[i] * delta;
        outv += st[i] * q[i];
        if (mine) rstate[state_base + uint(d)] = st[i];
    }
    outv = row_sum32(outv, local.x, local.y);
    if (dk_idx == 0 && mine) core_out[(b_idx * p.Hv + hv_idx) * p.Dv + dv_idx] = PIE_STORE(outv);
    if (mine) write_conv_token(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
#elif defined(PIE_PREFILL)
    int n = int(group.z), t = n / p.Hv, hv_idx = n % p.Hv;
    int rep = p.Hv / p.Hk, hk_idx = hv_idx / rep;
    bool hk_first = (hv_idx % rep) == 0;
    int slot = int(slot_ids[0]);
    int dk_idx = int(local.x);
    int n_per_t = p.Dk / 32;
    int pitch_f = pc.row_pitch / 2;
    float qraw[8], kraw[8], qsq = 0.0, ksq = 0.0;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        qraw[i] = convsilu_prefill(slot, t, p.q_off + hk_idx * p.Dk + d);
        kraw[i] = convsilu_prefill(slot, t, p.k_off + hk_idx * p.Dk + d);
        qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i];
    }
    float qinv = p.inv_sqrt_dk / sqrt(row_sum32(qsq, local.x, local.y) + p.eps);
    float kinv = 1.0 / sqrt(row_sum32(ksq, local.x, local.y) + p.eps);
    int row_f = t * pitch_f;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        pre_q[row_f + hv_idx * p.Dk + d] = qraw[i] * qinv;
        pre_k[row_f + hv_idx * p.Dk + d] = kraw[i] * kinv;
    }
    if (dk_idx == 0) {
        int row_t = t * pc.row_pitch;
        float ad = PIE_LOAD(a_gate[row_t + hv_idx]) + PIE_LOAD(dt_bias[hv_idx]);
        float sp = max(ad, 0.0) + log(1.0 + exp(-abs(ad)));
        pre_gate[row_f + 2 * hv_idx + 0] = exp(-exp(A_log[hv_idx]) * sp);
        pre_gate[row_f + 2 * hv_idx + 1] = 1.0 / (1.0 + exp(-PIE_LOAD(b_gate[row_t + hv_idx])));
    }
    for (int dv = dk_idx; dv < p.Dv; dv += 32)
        pre_gate[row_f + 2 * p.Hv + hv_idx * p.Dv + dv] = convsilu_prefill(slot, t, p.v_off + hv_idx * p.Dv + dv);
    if (t != pc.n_scan - 1) return;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        if (hk_first) { write_conv_prefill(slot, t, p.q_off + hk_idx * p.Dk + d); write_conv_prefill(slot, t, p.k_off + hk_idx * p.Dk + d); }
    }
    for (int dv = dk_idx; dv < p.Dv; dv += 32) write_conv_prefill(slot, t, p.v_off + hv_idx * p.Dv + dv);
#else
    int n = int(group.z);
    int b_idx = n / p.Hv;
    int hv_idx = n % p.Hv;
    int rep = p.Hv / p.Hk;
    int hk_idx = hv_idx / rep;
    bool hk_first = (hv_idx % rep) == 0;
    int dk_idx = int(local.x);
    int n_per_t = p.Dk / 32;
#if defined(PIE_SLOTTED)
    int slot = int(slot_ids[b_idx]);
#else
    int slot = b_idx;
#endif
    float qraw[8]; float kraw[8]; float qsq = 0.0; float ksq = 0.0;
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        qraw[i] = convsilu_token(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
        kraw[i] = convsilu_token(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
        qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i];
    }
    float qinv = p.inv_sqrt_dk / sqrt(row_sum32(qsq, local.x, local.y) + p.eps);
    float kinv = 1.0 / sqrt(row_sum32(ksq, local.x, local.y) + p.eps);
    uint pk_base = uint(n * p.Dk);
    for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        pre_q[pk_base + uint(d)] = qraw[i] * qinv;
        pre_k[pk_base + uint(d)] = kraw[i] * kinv;
    }
    if (dk_idx == 0) {
        float ad = PIE_LOAD(a_gate[b_idx * p.Hv + hv_idx]) + PIE_LOAD(dt_bias[hv_idx]);
        float sp = max(ad, 0.0) + log(1.0 + exp(-abs(ad)));
        pre_gate[2 * n + 0] = exp(-exp(A_log[hv_idx]) * sp);
        pre_gate[2 * n + 1] = 1.0 / (1.0 + exp(-PIE_LOAD(b_gate[b_idx * p.Hv + hv_idx])));
    }
    if (hk_first) for (int i = 0; i < n_per_t; ++i) {
        int d = n_per_t * dk_idx + i;
        write_conv_token(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
        write_conv_token(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
    }
#endif
}
#endif

// pie:instantiate gdn_core_recurrent_bfloat16 PIE_RECURRENT=1
// pie:instantiate gdn_core_recurrent_slotted_bfloat16 PIE_RECURRENT=1 PIE_SLOTTED=1
// pie:instantiate gdn_prep_bfloat16
// pie:instantiate gdn_prep_slotted_bfloat16 PIE_SLOTTED=1
// pie:instantiate gdn_prep_prefill_bfloat16 PIE_PREFILL=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_16_v_1 PIE_SCAN=1 PIE_LANES=16 PIE_VROWS=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_16_v_2 PIE_SCAN=1 PIE_LANES=16 PIE_VROWS=2
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_16_v_4 PIE_SCAN=1 PIE_LANES=16 PIE_VROWS=4
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_32_v_2 PIE_SCAN=1 PIE_LANES=32 PIE_VROWS=2
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_32_v_4 PIE_SCAN=1 PIE_LANES=32 PIE_VROWS=4
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_32_v_8 PIE_SCAN=1 PIE_LANES=32 PIE_VROWS=8
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_4_v_1 PIE_SCAN=1 PIE_LANES=4 PIE_VROWS=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_8_v_1 PIE_SCAN=1 PIE_LANES=8 PIE_VROWS=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_8_v_2 PIE_SCAN=1 PIE_LANES=8 PIE_VROWS=2
