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
//           pre_gate[R, 2*Hv + Hv*Dv] fp32 ({decay,beta} then precomputed V)
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

#include "gdn_params.h"

// ---------------------------------------------------------------------------
// (1) Prep: the dv-independent q/k path, computed ONCE per (req, v-head).
// ---------------------------------------------------------------------------
template <typename T, bool SLOTTED>
METAL_FUNC void gdn_prep_body(
    const device T* mixed, const device float* conv_state,
    const device T* conv_w, const device T* conv_b,
    const device float* A_log, const device T* dt_bias,
    const device T* a_gate, const device T* b_gate,
    device float* pre_q, device float* pre_k, device float* pre_gate,
    device float* new_conv_state, constant GdnCoreParams& p,
    const device uint* slot_ids, uint3 tpig, uint simd_lane) {
  const int Dk = p.Dk, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n        = int(tpig.z);          // 0 .. R*Hv-1
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;
  // See gdn_core.metal: q/k carry Hk heads and are repeated to Hv, so a value
  // head reads key head hv/rep. rep==1 makes this the identity.
  const int rep      = Hv / p.Hk;
  const int hk_idx   = hv_idx / rep;
  const bool hk_first = (hv_idx % rep) == 0;
  const int slot     = SLOTTED ? int(slot_ids[b_idx]) : b_idx;
  const int dk_idx   = int(tpig.x);          // 0..31 (== simd_lane; one simdgroup per head)
  const int n_per_t  = Dk / 32;              // 4
  const int q_off = p.q_off, k_off = p.k_off;

  // conv1d(+silu) over a channel's Kc-tap window (history + this token).
  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));          // silu
  };

  // q/k conv/silu + l2norm + q-prescale (identical to gdn_core L100-116).
  float qraw[8], kraw[8];                      // n_per_t<=8
  for (int i = 0; i < n_per_t; ++i) {
    int d = n_per_t * dk_idx + i;              // 0..Dk-1
    qraw[i] = convsilu(q_off + hk_idx * Dk + d);
    kraw[i] = convsilu(k_off + hk_idx * Dk + d);
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
  // Each q/k channel written exactly once: prep runs once per VALUE head, and
  // `rep` value heads share a key head, so only the group's first writes.
  auto wb = [&](int c) {
    for (int j = 0; j < Kc - 1; ++j)
      new_conv_state[(slot * Kc + j) * CDIM + c] =
          conv_state[(slot * Kc + (j + 1)) * CDIM + c];
    new_conv_state[(slot * Kc + (Kc - 1)) * CDIM + c] =
        float(mixed[b_idx * CDIM + c]);
  };
  if (hk_first) {
    for (int i = 0; i < n_per_t; ++i) {
      int d = n_per_t * dk_idx + i;
      wb(q_off + hk_idx * Dk + d);
      wb(k_off + hk_idx * Dk + d);
    }
  }
}

template <typename T>
[[kernel]] void gdn_prep(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    const device T* conv_w [[buffer(2)]], const device T* conv_b [[buffer(3)]],
    const device float* A_log [[buffer(4)]], const device T* dt_bias [[buffer(5)]],
    const device T* a_gate [[buffer(6)]], const device T* b_gate [[buffer(7)]],
    device float* pre_q [[buffer(8)]], device float* pre_k [[buffer(9)]],
    device float* pre_gate [[buffer(10)]], device float* new_conv_state [[buffer(11)]],
    constant GdnCoreParams& p [[buffer(12)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  gdn_prep_body<T, false>(
      mixed, conv_state, conv_w, conv_b, A_log, dt_bias, a_gate, b_gate,
      pre_q, pre_k, pre_gate, new_conv_state, p,
      (const device uint*)nullptr, tpig, simd_lane);
}

// ---------------------------------------------------------------------------
// (2) Recurrent: per (req, v-head, v-dim); reads prep's q/k/gating from scratch.
// ---------------------------------------------------------------------------
template <typename T, bool SLOTTED>
METAL_FUNC void gdn_core_recurrent_body(
    const device T* mixed, const device float* conv_state,
    device float* rstate, device T* core_out,
    const device T* conv_w, const device T* conv_b,
    const device float* pre_q, const device float* pre_k,
    const device float* pre_gate, device float* new_conv_state,
    constant GdnCoreParams& p, const device uint* slot_ids,
    uint3 tpig, uint simd_lane) {
  const int Dk = p.Dk, Dv = p.Dv, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n        = int(tpig.z);          // 0 .. R*Hv-1
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;
  const int slot     = SLOTTED ? int(slot_ids[b_idx]) : b_idx;
  const int dk_idx   = int(tpig.x);          // 0..31 (== simd_lane)
  const int dv_idx   = int(tpig.y);          // 0..Vd-1
  const int n_per_t  = Dk / 32;              // 4
  const int v_off = p.v_off;

  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
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
  device float* i_state =
      rstate + (size_t((slot * Hv + hv_idx) * Dv + dv_idx) * Dk);
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
      new_conv_state[(slot * Kc + j) * CDIM + c] =
          conv_state[(slot * Kc + (j + 1)) * CDIM + c];
    new_conv_state[(slot * Kc + (Kc - 1)) * CDIM + c] =
        float(mixed[b_idx * CDIM + c]);
  };
  wb(v_off + hv_idx * Dv + dv_idx);
}

template <typename T>
[[kernel]] void gdn_core_recurrent(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    device float* rstate [[buffer(2)]], device T* core_out [[buffer(3)]],
    const device T* conv_w [[buffer(4)]], const device T* conv_b [[buffer(5)]],
    const device float* pre_q [[buffer(6)]], const device float* pre_k [[buffer(7)]],
    const device float* pre_gate [[buffer(8)]], device float* new_conv_state [[buffer(9)]],
    constant GdnCoreParams& p [[buffer(10)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  gdn_core_recurrent_body<T, false>(
      mixed, conv_state, rstate, core_out, conv_w, conv_b,
      pre_q, pre_k, pre_gate, new_conv_state, p,
      (const device uint*)nullptr, tpig, simd_lane);
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
  gdn_prep_body<T, true>(
      mixed, conv_state, conv_w, conv_b, A_log, dt_bias, a_gate, b_gate,
      pre_q, pre_k, pre_gate, new_conv_state, p, slot_ids, tpig, simd_lane);
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
  gdn_core_recurrent_body<T, true>(
      mixed, conv_state, rstate, core_out, conv_w, conv_b,
      pre_q, pre_k, pre_gate, new_conv_state, p, slot_ids, tpig, simd_lane);
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

instantiate_gdn_prep(bfloat16, bfloat)
instantiate_gdn_prep_slotted(bfloat16, bfloat)

// ---------------------------------------------------------------------------
// Prefill variants: one dispatch for a whole prompt instead of one per token.
//
// A prompt walked token by token serializes the GDN pair behind a barrier per
// token -- 34 tokens x 18 layers x 2 kernels = 1224 dispatches in a strict
// chain, measured at 25ms of a 60ms prefill.  Two observations remove it:
//
//   * The conv1d is a Kc-tap FIR over the prompt, so token t's window is just
//     `mixed[t-Kc+1 .. t]`, falling back to the persisted history only while
//     `t < Kc-1`.  No ping-pong is needed *within* a prompt, which makes the
//     whole prep pass token-parallel.
//   * The recurrent state row for (slot, hv, dv) is owned by exactly one
//     simdgroup, so the scan over tokens can run in registers inside the kernel
//     with no cross-threadgroup ordering at all.
//
// Both read the prompt through a uniform `row_pitch` (the prefill scratch
// layout's widest tensor, in activation elements; fp32 scratch is half that)
// off token 0's argument table, and only the last scanned token writes the
// conv history forward -- which is why the caller keeps an odd trailing token
// on the per-token path, so this always reads `conv_state` and writes
// `new_conv_state` and never races the two.
//
// The arithmetic is unchanged: the same taps in the same order, the same
// simd_sum reductions, the same fp32 scratch round-trip.
template <typename T>
[[kernel]] void gdn_prep_prefill(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    const device T* conv_w [[buffer(2)]], const device T* conv_b [[buffer(3)]],
    const device float* A_log [[buffer(4)]], const device T* dt_bias [[buffer(5)]],
    const device T* a_gate [[buffer(6)]], const device T* b_gate [[buffer(7)]],
    device float* pre_q [[buffer(8)]], device float* pre_k [[buffer(9)]],
    device float* pre_gate [[buffer(10)]], device float* new_conv_state [[buffer(11)]],
    constant GdnCoreParams& p [[buffer(12)]], const device uint* slot_ids [[buffer(13)]],
    constant int& row_pitch [[buffer(14)]], constant int& n_scan [[buffer(15)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  const int Dk = p.Dk, Dv = p.Dv, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n = int(tpig.z), t = n / Hv, hv_idx = n % Hv;
  // See gdn_core.metal: q/k carry Hk heads repeated to Hv; rep==1 is identity.
  const int rep = Hv / p.Hk;
  const int hk_idx = hv_idx / rep;
  const bool hk_first = (hv_idx % rep) == 0;
  const int slot = int(slot_ids[0]), dk_idx = int(tpig.x), n_per_t = Dk / 32;
  const int q_off = p.q_off, k_off = p.k_off, v_off = p.v_off;
  const size_t pitch_t = size_t(row_pitch);
  const size_t pitch_f = size_t(row_pitch) / 2;   // fp32 scratch shares the byte pitch
  const size_t row_t = size_t(t) * pitch_t;

  // Tap j of token t's conv window: prompt row `t-(Kc-1)+j` when it exists,
  // otherwise row `Kc + idx` of the history this prompt started from.
  auto tap = [&](int j, int c) -> float {
    const int idx = t - (Kc - 1) + j;
    return idx >= 0 ? float(mixed[size_t(idx) * pitch_t + c])
                    : conv_state[(slot * Kc + Kc + idx) * CDIM + c];
  };
  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j) acc += tap(j, c) * float(conv_w[c * Kc + j]);
    acc += float(mixed[row_t + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));
  };

  float qraw[8], kraw[8];
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    qraw[i] = convsilu(q_off + hk_idx * Dk + d);
    kraw[i] = convsilu(k_off + hk_idx * Dk + d);
  }
  float qsq = 0.0f, ksq = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i]; }
  qsq = simd_sum(qsq); ksq = simd_sum(ksq);
  const float qinv = p.inv_sqrt_dk / sqrt(qsq + p.eps);
  const float kinv = 1.0f / sqrt(ksq + p.eps);
  device float* oq = pre_q + size_t(t) * pitch_f + size_t(hv_idx) * Dk;
  device float* ok = pre_k + size_t(t) * pitch_f + size_t(hv_idx) * Dk;
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    oq[d] = qraw[i] * qinv;
    ok[d] = kraw[i] * kinv;
  }
  if (dk_idx == 0) {
    const float ad = float(a_gate[row_t + hv_idx]) + float(dt_bias[hv_idx]);
    const float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));
    device float* g = pre_gate + size_t(t) * pitch_f + 2 * size_t(hv_idx);
    g[0] = exp(-exp(float(A_log[hv_idx])) * sp);
    g[1] = 1.0f / (1.0f + exp(-float(b_gate[row_t + hv_idx])));
  }
  // The recurrent kernel's thirty-two Dk lanes all consume the same v scalar.
  // Compute each value-channel convolution once here, while tokens are
  // parallel, and place it after the two gate scalars in this fp32 scratch row.
  device float* pv = pre_gate + size_t(t) * pitch_f + 2 * size_t(Hv);
  for (int dv = dk_idx; dv < Dv; dv += 32)
    pv[size_t(hv_idx) * Dv + dv] =
        convsilu(v_off + hv_idx * Dv + dv);

  // Only the last scanned token carries the history forward.
  if (t != n_scan - 1) return;
  auto wb = [&](int c) {
    for (int j = 0; j < Kc; ++j) {
      const int idx = t - (Kc - 1) + j;
      new_conv_state[(slot * Kc + j) * CDIM + c] =
          idx >= 0 ? float(mixed[size_t(idx) * pitch_t + c])
                   : conv_state[(slot * Kc + Kc + idx) * CDIM + c];
    }
  };
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    if (hk_first) {
      wb(q_off + hk_idx * Dk + d);
      wb(k_off + hk_idx * Dk + d);
    }
  }
  for (int dv = dk_idx; dv < Dv; dv += 32)
    wb(v_off + hv_idx * Dv + dv);
}

template <typename T>
[[kernel]] void gdn_core_recurrent_prefill(
    device float* rstate [[buffer(2)]], device T* core_out [[buffer(3)]],
    const device float* pre_q [[buffer(6)]], const device float* pre_k [[buffer(7)]],
    const device float* pre_gate [[buffer(8)]],
    constant GdnCoreParams& p [[buffer(10)]], const device uint* slot_ids [[buffer(11)]],
    constant int& row_pitch [[buffer(12)]], constant int& n_scan [[buffer(13)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  const int Dk = p.Dk, Dv = p.Dv, Hv = p.Hv;
  const int hv_idx = int(tpig.z), dv_idx = int(tpig.y);
  const int slot = int(slot_ids[0]), dk_idx = int(tpig.x), n_per_t = Dk / 32;
  const size_t pitch_f = size_t(row_pitch) / 2;

  // This simdgroup owns the whole (slot, hv, dv) state row, so the scan runs in
  // registers with no ordering beyond the loop itself.
  device float* i_state = rstate + (size_t((slot * Hv + hv_idx) * Dv + dv_idx) * Dk);
  float st[8];
  for (int i = 0; i < n_per_t; ++i) st[i] = i_state[n_per_t * dk_idx + i];

  for (int t = 0; t < n_scan; ++t) {
    const size_t row_t = size_t(t) * size_t(row_pitch);
    const float vval =
        pre_gate[size_t(t) * pitch_f + 2 * size_t(Hv) +
                 size_t(hv_idx) * Dv + dv_idx];

    const device float* iq = pre_q + size_t(t) * pitch_f + size_t(hv_idx) * Dk;
    const device float* ik = pre_k + size_t(t) * pitch_f + size_t(hv_idx) * Dk;
    const device float* g = pre_gate + size_t(t) * pitch_f + 2 * size_t(hv_idx);
    float q[8], k[8];
    for (int i = 0; i < n_per_t; ++i) {
      const int d = n_per_t * dk_idx + i;
      q[i] = iq[d];
      k[i] = ik[d];
    }
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) { st[i] *= g[0]; kv_mem += st[i] * k[i]; }
    kv_mem = simd_sum(kv_mem);
    const float delta = (vval - kv_mem) * g[1];
    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) { st[i] += k[i] * delta; out += st[i] * q[i]; }
    out = simd_sum(out);
    if (simd_lane == 0)
      core_out[row_t + size_t(hv_idx) * Dv + dv_idx] = static_cast<T>(out);
  }
  for (int i = 0; i < n_per_t; ++i) i_state[n_per_t * dk_idx + i] = st[i];
}

#define instantiate_gdn_prefill(name, itype)                                    \
  template [[host_name("gdn_prep_prefill_" #name)]] [[kernel]] void             \
  gdn_prep_prefill<itype>(                                                      \
      const device itype*, const device float*, const device itype*,            \
      const device itype*, const device float*, const device itype*,            \
      const device itype*, const device itype*, device float*, device float*,   \
      device float*, device float*, constant GdnCoreParams&, const device uint*,\
      constant int&, constant int&, uint3, uint);                               \
  template [[host_name("gdn_core_recurrent_prefill_" #name)]] [[kernel]] void   \
  gdn_core_recurrent_prefill<itype>(                                            \
      device float*, device itype*, const device float*, const device float*,   \
      const device float*,                                                     \
      constant GdnCoreParams&, const device uint*, constant int&, constant int&,\
      uint3, uint);

instantiate_gdn_prefill(bfloat16, bfloat)
