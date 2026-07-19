// Raw-Metal GDN-core PREP-DISPATCH split (decisive 128x->1x de-redundancy).
//
// The single-dispatch gdn_core.metal recomputes the dv-INDEPENDENT q/k path
// (conv1d+silu + l2norm + q-prescale + gating decay/beta) once per dv tile.
// The in-kernel threadgroup-share caps that at 4x redundancy (a tg of 1024
// threads = 32 simdgroups can only span 32 of Vd=128 dv). To kill the residual
// 4x and reach the full 1x, split into two dispatches bridged by global scratch:
//
//   (1) gdn_prep<T>        grid {32, 1, R*Hv}  tg {32,1,1}
//         one simdgroup per (req, v-head); 32 dk-lanes x n_per_t=4 cover Dk=128.
//         Computes the q/k path EXACTLY ONCE per head (zero redundancy) +
//         writes the q/k channels of new_conv_state. Emits to scratch:
//           pre_q   [R, Hv, Dk]  fp32   (normalized + 1/sqrt(Dk)-prescaled q)
//           pre_k   [R, Hv, Dk]  fp32   (normalized k)
//           pre_gate[R, Hv, 2 ]  fp32   ({decay, beta} per head)
//
//   (2) gdn_core_recurrent<T>  grid {32, Vd, R*Hv}  tg {32,4,1}
//         one simdgroup per (req, v-head, v-dim); reads pre_q/pre_k/pre_gate
//         from scratch (NO recompute), does its own per-dv v-channel conv+silu,
//         the recurrent state RMW (+2 simd_sums), and the per-dv v-channel
//         new_conv_state writeback.
//
// Bit-exactness vs the single fused gdn_core: pre_q/pre_k are stored fp32 with
// the SAME 32-lane simd_sum reduction over Dk=128 (=32 lanes x 4), so the values
// the recurrent kernel reads are IDENTICAL to the in-kernel `sh_q/sh_k` floats.
// new_conv_state coverage is the same per-channel-once invariant: prep writes
// every q/k channel exactly once per head, recurrent writes every v channel
// exactly once per (head,dv). gating (decay/beta) computed once per head in fp32.
//
// Dependency: gdn_core_recurrent reads prep's scratch outputs -> prep must
// barrier-complete before recurrent (host DAG inserts the GdnPrep edge before
// GdnCore; hazard coloring enforces the scratch RAW). rstate is untouched by
// prep; recurrent reads+writes it in-place (each (req,hv,dv) row owned by one
// threadgroup, race-free, same as the fused kernel).

#include <metal_stdlib>
using namespace metal;

struct GdnCoreParams {
  int   Dk;            // k/q head dim (128)
  int   Dv;            // v head dim (128)
  int   Hk;            // k/q heads (16)
  int   Hv;            // v heads (16)
  int   conv_dim;      // 2*Hk*Dk + Hv*Dv (6144)
  int   Kc;            // conv kernel width (4)
  int   q_off;         // q channel offset within conv_dim (0)
  int   k_off;         // k channel offset (Hk*Dk)
  int   v_off;         // v channel offset (2*Hk*Dk)
  float eps;           // l2norm eps (1e-6)
  float inv_sqrt_dk;   // 1/sqrt(Dk) q pre-scale (0.0883883)
};

// ---------------------------------------------------------------------------
// (1) Prep: the dv-independent q/k path, computed ONCE per (req, v-head).
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void gdn_prep(
    const device T*     mixed          [[buffer(0)]],   // [R, conv_dim] this token's in-proj mixed_qkv
    const device float* conv_state     [[buffer(1)]],   // [R, Kc, conv_dim] READ-ONLY conv history
    const device T*     conv_w         [[buffer(2)]],   // [conv_dim, Kc]
    const device T*     conv_b         [[buffer(3)]],   // [conv_dim]
    const device float* A_log          [[buffer(4)]],   // [Hv]  F32 in ckpt
    const device T*     dt_bias        [[buffer(5)]],   // [Hv]
    const device T*     a_gate         [[buffer(6)]],   // [R, Hv]
    const device T*     b_gate         [[buffer(7)]],   // [R, Hv]
    device float*       pre_q          [[buffer(8)]],   // [R, Hv, Dk] fp32 OUT
    device float*       pre_k          [[buffer(9)]],   // [R, Hv, Dk] fp32 OUT
    device float*       pre_gate       [[buffer(10)]],  // [R, Hv, 2]  fp32 OUT {decay, beta}
    device float*       new_conv_state [[buffer(11)]],  // [R, Kc, conv_dim] writes q/k channels (ping-pong, != conv_state)
    constant GdnCoreParams& p          [[buffer(12)]],
    uint3 tpig                         [[thread_position_in_grid]],
    uint  simd_lane                    [[thread_index_in_simdgroup]]) {
  const int Dk = p.Dk, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n        = int(tpig.z);          // 0 .. R*Hv-1
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;
  const int dk_idx   = int(tpig.x);          // 0..31 (== simd_lane; one simdgroup per head)
  const int n_per_t  = Dk / 32;              // 4
  const int q_off = p.q_off, k_off = p.k_off;

  // conv1d(+silu) over a channel's Kc-tap window (history + this token).
  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(b_idx * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));          // silu
  };

  // q/k conv/silu + l2norm + q-prescale (identical to gdn_core L100-116).
  float qraw[8], kraw[8];                      // n_per_t<=8
  for (int i = 0; i < n_per_t; ++i) {
    int d = n_per_t * dk_idx + i;              // 0..Dk-1
    qraw[i] = convsilu(q_off + hv_idx * Dk + d);
    kraw[i] = convsilu(k_off + hv_idx * Dk + d);
  }
  float qsq = 0.0f, ksq = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i]; }
  qsq = simd_sum(qsq); ksq = simd_sum(ksq);
  float qinv = p.inv_sqrt_dk / sqrt(qsq + p.eps);   // q also pre-scaled 1/sqrt(Dk)
  float kinv = 1.0f / sqrt(ksq + p.eps);
  device float* oq = pre_q + size_t(n) * Dk;
  device float* ok = pre_k + size_t(n) * Dk;
  for (int i = 0; i < n_per_t; ++i) {
    int d = n_per_t * dk_idx + i;
    oq[d] = qraw[i] * qinv;
    ok[d] = kraw[i] * kinv;
  }

  // gating: g = -exp(A_log)*softplus(a+dt_bias); decay = exp(g); beta = sigmoid(b).
  if (dk_idx == 0) {
    float ad = float(a_gate[b_idx * Hv + hv_idx]) + float(dt_bias[hv_idx]);
    float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));   // softplus
    pre_gate[2 * n + 0] = exp(-exp(float(A_log[hv_idx])) * sp);
    pre_gate[2 * n + 1] = 1.0f / (1.0f + exp(-float(b_gate[b_idx * Hv + hv_idx])));
  }

  // q/k conv_state writeback (shift + append) -> ping-pong new_conv_state.
  // Each q/k channel written exactly once per head (was dv_idx==0-guarded in the
  // fused kernel; here it is naturally once since prep runs once per head).
  auto wb = [&](int c) {
    for (int j = 0; j < Kc - 1; ++j)
      new_conv_state[(b_idx * Kc + j) * CDIM + c] = conv_state[(b_idx * Kc + (j + 1)) * CDIM + c];
    new_conv_state[(b_idx * Kc + (Kc - 1)) * CDIM + c] = float(mixed[b_idx * CDIM + c]);
  };
  for (int i = 0; i < n_per_t; ++i) {
    int d = n_per_t * dk_idx + i;
    wb(q_off + hv_idx * Dk + d);
    wb(k_off + hv_idx * Dk + d);
  }
}

// ---------------------------------------------------------------------------
// (2) Recurrent: per (req, v-head, v-dim); reads prep's q/k/gating from scratch.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void gdn_core_recurrent(
    const device T*     mixed          [[buffer(0)]],   // [R, conv_dim] (this token's mixed; v conv)
    const device float* conv_state     [[buffer(1)]],   // [R, Kc, conv_dim] READ-ONLY (v history)
    device float*       rstate         [[buffer(2)]],   // [R, Hv, Vd, Dk] in-place native recurrent state
    device T*           core_out       [[buffer(3)]],   // [R, Hv, Vd] pre-GatedRms output
    const device T*     conv_w         [[buffer(4)]],   // [conv_dim, Kc]
    const device T*     conv_b         [[buffer(5)]],   // [conv_dim]
    const device float* pre_q          [[buffer(6)]],   // [R, Hv, Dk] fp32 (from gdn_prep)
    const device float* pre_k          [[buffer(7)]],   // [R, Hv, Dk] fp32 (from gdn_prep)
    const device float* pre_gate       [[buffer(8)]],   // [R, Hv, 2]  fp32 {decay, beta}
    device float*       new_conv_state [[buffer(9)]],   // [R, Kc, conv_dim] writes v channels (ping-pong)
    constant GdnCoreParams& p          [[buffer(10)]],
    uint3 tpig                         [[thread_position_in_grid]],
    uint  simd_lane                    [[thread_index_in_simdgroup]]) {
  const int Dk = p.Dk, Dv = p.Dv, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n        = int(tpig.z);          // 0 .. R*Hv-1
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;
  const int dk_idx   = int(tpig.x);          // 0..31 (== simd_lane)
  const int dv_idx   = int(tpig.y);          // 0..Vd-1
  const int n_per_t  = Dk / 32;              // 4
  const int v_off = p.v_off;

  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(b_idx * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));          // silu
  };

  // this lane's v channel (unique per dv) + read the shared q/k/gating from scratch.
  float vval = convsilu(v_off + hv_idx * Dv + dv_idx);
  const device float* iq = pre_q + size_t(n) * Dk;
  const device float* ik = pre_k + size_t(n) * Dk;
  float q[8], k[8];
  for (int i = 0; i < n_per_t; ++i) { int d = n_per_t * dk_idx + i; q[i] = iq[d]; k[i] = ik[d]; }
  float gdecay = pre_gate[2 * n + 0], beta = pre_gate[2 * n + 1];

  // --- recurrent step (register state + simd_sum), native [Hv,Vd,Dk] ---
  device float* i_state = rstate + (size_t(n * Dv + dv_idx) * Dk);
  float st[8];
  for (int i = 0; i < n_per_t; ++i) st[i] = i_state[n_per_t * dk_idx + i];

  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { st[i] = st[i] * gdecay; kv_mem += st[i] * k[i]; }
  kv_mem = simd_sum(kv_mem);
  float delta = (vval - kv_mem) * beta;
  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { st[i] = st[i] + k[i] * delta; out += st[i] * q[i]; }
  out = simd_sum(out);
  if (simd_lane == 0)
    core_out[(b_idx * Hv + hv_idx) * Dv + dv_idx] = static_cast<T>(out);
  for (int i = 0; i < n_per_t; ++i) i_state[n_per_t * dk_idx + i] = st[i];

  // v-channel conv_state writeback (per dv; q/k channels written by gdn_prep).
  auto wb = [&](int c) {
    for (int j = 0; j < Kc - 1; ++j)
      new_conv_state[(b_idx * Kc + j) * CDIM + c] = conv_state[(b_idx * Kc + (j + 1)) * CDIM + c];
    new_conv_state[(b_idx * Kc + (Kc - 1)) * CDIM + c] = float(mixed[b_idx * CDIM + c]);
  };
  wb(v_off + hv_idx * Dv + dv_idx);
}

// Paged/multi-batch counterparts retain the prep/recurrent split (and thus the
// exact DAG dependency) while redirecting only persistent state through the
// per-token slot map.  Activations and prep scratch remain token-major.
template <typename T>
[[kernel]] void gdn_prep_slotted(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    const device T* conv_w [[buffer(2)]], const device T* conv_b [[buffer(3)]],
    const device float* A_log [[buffer(4)]], const device T* dt_bias [[buffer(5)]],
    const device T* a_gate [[buffer(6)]], const device T* b_gate [[buffer(7)]],
    device float* pre_q [[buffer(8)]], device float* pre_k [[buffer(9)]],
    device float* pre_gate [[buffer(10)]], device float* new_conv_state [[buffer(11)]],
    constant GdnCoreParams& p [[buffer(12)]], const device uint* slot_ids [[buffer(13)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  const int Dk = p.Dk, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n = int(tpig.z), b_idx = n / Hv, hv_idx = n % Hv;
  const int slot = int(slot_ids[b_idx]), dk_idx = int(tpig.x), n_per_t = Dk / 32;
  const int q_off = p.q_off, k_off = p.k_off;
  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));
  };
  float qraw[8], kraw[8];
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    qraw[i] = convsilu(q_off + hv_idx * Dk + d);
    kraw[i] = convsilu(k_off + hv_idx * Dk + d);
  }
  float qsq = 0.0f, ksq = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i]; }
  const float qinv = p.inv_sqrt_dk / sqrt(simd_sum(qsq) + p.eps);
  const float kinv = 1.0f / sqrt(simd_sum(ksq) + p.eps);
  device float* oq = pre_q + size_t(n) * Dk;
  device float* ok = pre_k + size_t(n) * Dk;
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    oq[d] = qraw[i] * qinv; ok[d] = kraw[i] * kinv;
  }
  if (dk_idx == 0) {
    const float ad = float(a_gate[b_idx * Hv + hv_idx]) + float(dt_bias[hv_idx]);
    const float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));
    pre_gate[2 * n] = exp(-exp(float(A_log[hv_idx])) * sp);
    pre_gate[2 * n + 1] = 1.0f / (1.0f + exp(-float(b_gate[b_idx * Hv + hv_idx])));
  }
  auto wb = [&](int c) {
    for (int j = 0; j < Kc - 1; ++j)
      new_conv_state[(slot * Kc + j) * CDIM + c] =
          conv_state[(slot * Kc + (j + 1)) * CDIM + c];
    new_conv_state[(slot * Kc + (Kc - 1)) * CDIM + c] = float(mixed[b_idx * CDIM + c]);
  };
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    wb(q_off + hv_idx * Dk + d); wb(k_off + hv_idx * Dk + d);
  }
}

template <typename T>
[[kernel]] void gdn_core_recurrent_slotted(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    device float* rstate [[buffer(2)]], device T* core_out [[buffer(3)]],
    const device T* conv_w [[buffer(4)]], const device T* conv_b [[buffer(5)]],
    const device float* pre_q [[buffer(6)]], const device float* pre_k [[buffer(7)]],
    const device float* pre_gate [[buffer(8)]], device float* new_conv_state [[buffer(9)]],
    constant GdnCoreParams& p [[buffer(10)]], const device uint* slot_ids [[buffer(11)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  const int Dk = p.Dk, Dv = p.Dv, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n = int(tpig.z), b_idx = n / Hv, hv_idx = n % Hv, dv_idx = int(tpig.y);
  const int slot = int(slot_ids[b_idx]), dk_idx = int(tpig.x), n_per_t = Dk / 32;
  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));
  };
  const float vval = convsilu(p.v_off + hv_idx * Dv + dv_idx);
  const device float* iq = pre_q + size_t(n) * Dk;
  const device float* ik = pre_k + size_t(n) * Dk;
  float q[8], k[8], st[8];
  device float* i_state = rstate + (size_t((slot * Hv + hv_idx) * Dv + dv_idx) * Dk);
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    q[i] = iq[d]; k[i] = ik[d]; st[i] = i_state[d];
    st[i] *= pre_gate[2 * n]; kv_mem += st[i] * k[i];
  }
  kv_mem = simd_sum(kv_mem);
  const float delta = (vval - kv_mem) * pre_gate[2 * n + 1];
  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { st[i] += k[i] * delta; out += st[i] * q[i]; }
  out = simd_sum(out);
  if (simd_lane == 0) core_out[(b_idx * Hv + hv_idx) * Dv + dv_idx] = static_cast<T>(out);
  for (int i = 0; i < n_per_t; ++i) i_state[n_per_t * dk_idx + i] = st[i];
  const int c = p.v_off + hv_idx * Dv + dv_idx;
  for (int j = 0; j < Kc - 1; ++j)
    new_conv_state[(slot * Kc + j) * CDIM + c] =
        conv_state[(slot * Kc + (j + 1)) * CDIM + c];
  new_conv_state[(slot * Kc + (Kc - 1)) * CDIM + c] = float(mixed[b_idx * CDIM + c]);
}

#define instantiate_gdn_prep(name, itype)                                  \
  template [[host_name("gdn_prep_" #name)]] [[kernel]] void                \
  gdn_prep<itype>(                                                         \
      const device itype*, const device float*, const device itype*,       \
      const device itype*, const device float*, const device itype*,       \
      const device itype*, const device itype*, device float*,             \
      device float*, device float*, device float*,                         \
      constant GdnCoreParams&, uint3, uint);                               \
  template [[host_name("gdn_core_recurrent_" #name)]] [[kernel]] void      \
  gdn_core_recurrent<itype>(                                               \
      const device itype*, const device float*, device float*,             \
      device itype*, const device itype*, const device itype*,             \
      const device float*, const device float*, const device float*,       \
      device float*, constant GdnCoreParams&, uint3, uint);

#define instantiate_gdn_prep_slotted(name, itype)                           \
  template [[host_name("gdn_prep_slotted_" #name)]] [[kernel]] void         \
  gdn_prep_slotted<itype>(                                                  \
      const device itype*, const device float*, const device itype*,        \
      const device itype*, const device float*, const device itype*,        \
      const device itype*, const device itype*, device float*,              \
      device float*, device float*, device float*, constant GdnCoreParams&, \
      const device uint*, uint3, uint);                                     \
  template [[host_name("gdn_core_recurrent_slotted_" #name)]] [[kernel]] void \
  gdn_core_recurrent_slotted<itype>(                                        \
      const device itype*, const device float*, device float*, device itype*, \
      const device itype*, const device itype*, const device float*,        \
      const device float*, const device float*, device float*,              \
      constant GdnCoreParams&, const device uint*, uint3, uint);

instantiate_gdn_prep(float32, float)
instantiate_gdn_prep(float16, half)
instantiate_gdn_prep(bfloat16, bfloat)
instantiate_gdn_prep_slotted(float32, float)
instantiate_gdn_prep_slotted(float16, half)
instantiate_gdn_prep_slotted(bfloat16, bfloat)
