#include <metal_stdlib>
using namespace metal;

template <typename T, bool SLOTTED>
METAL_FUNC void gdn_prep_body(
    const device T* mixed, const device float* conv_state,
    const device T* conv_w, const device T* conv_b,
    const device float* A_log, const device T* dt_bias,
    const device T* a_gate, const device T* b_gate,
    device float* pre_q, device float* pre_k, device float* pre_gate,
    device float* new_conv_state, const device uint* slot_ids,
    int Dk, int Dv, int Hk, int Hv, int conv_dim, int Kc,
    int q_off, int k_off, int v_off, float eps, float inv_sqrt_dk,
    uint3 tpig, uint simd_lane) {

  const int CDIM = conv_dim;
  const int n        = int(tpig.z);
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;

  const int rep      = Hv / Hk;
  const int hk_idx   = hv_idx / rep;
  const bool hk_first = (hv_idx % rep) == 0;
  const int slot     = SLOTTED ? int(slot_ids[b_idx]) : b_idx;
  const int dk_idx   = int(tpig.x);
  const int n_per_t  = Dk / 32;

  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));
  };

  float qraw[8], kraw[8];
  for (int i = 0; i < n_per_t; ++i) {
    int d = n_per_t * dk_idx + i;
    qraw[i] = convsilu(q_off + hk_idx * Dk + d);
    kraw[i] = convsilu(k_off + hk_idx * Dk + d);
  }
  float qsq = 0.0f, ksq = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i]; }
  qsq = simd_sum(qsq); ksq = simd_sum(ksq);
  float qinv = inv_sqrt_dk / sqrt(qsq + eps);
  float kinv = 1.0f / sqrt(ksq + eps);
  device float* oq = pre_q + size_t(n) * Dk;
  device float* ok = pre_k + size_t(n) * Dk;
  for (int i = 0; i < n_per_t; ++i) {
    int d = n_per_t * dk_idx + i;
    oq[d] = qraw[i] * qinv;
    ok[d] = kraw[i] * kinv;
  }

  if (dk_idx == 0) {
    float ad = float(a_gate[b_idx * Hv + hv_idx]) + float(dt_bias[hv_idx]);
    float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));
    pre_gate[2 * n + 0] = exp(-exp(float(A_log[hv_idx])) * sp);
    pre_gate[2 * n + 1] = 1.0f / (1.0f + exp(-float(b_gate[b_idx * Hv + hv_idx])));
  }

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
    const constant int& Dk           [[buffer(12)]],
    const constant int& Dv           [[buffer(13)]],
    const constant int& Hk           [[buffer(14)]],
    const constant int& Hv           [[buffer(15)]],
    const constant int& conv_dim     [[buffer(16)]],
    const constant int& Kc           [[buffer(17)]],
    const constant int& q_off        [[buffer(18)]],
    const constant int& k_off        [[buffer(19)]],
    const constant int& v_off        [[buffer(20)]],
    const constant float& eps          [[buffer(21)]],
    const constant float& inv_sqrt_dk  [[buffer(22)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  gdn_prep_body<T, false>(
      mixed, conv_state, conv_w, conv_b, A_log, dt_bias, a_gate, b_gate,
      pre_q, pre_k, pre_gate, new_conv_state,
      (const device uint*)nullptr,
      Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps, inv_sqrt_dk, tpig, simd_lane);
}

template <typename T, bool SLOTTED>
METAL_FUNC void gdn_core_recurrent_body(
    const device T* mixed, const device float* conv_state,
    device float* rstate, device T* core_out,
    const device T* conv_w, const device T* conv_b,
    const device float* pre_q, const device float* pre_k,
    const device float* pre_gate, device float* new_conv_state,
    const device uint* slot_ids,
    int Dk, int Dv, int Hk, int Hv, int conv_dim, int Kc,
    int q_off, int k_off, int v_off, float eps, float inv_sqrt_dk,
    uint3 tpig, uint simd_lane) {

  const int CDIM = conv_dim;
  const int n        = int(tpig.z);
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;
  const int slot     = SLOTTED ? int(slot_ids[b_idx]) : b_idx;
  const int dk_idx   = int(tpig.x);
  const int dv_idx   = int(tpig.y);
  const int n_per_t  = Dk / 32;

  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));
  };

  float vval = convsilu(v_off + hv_idx * Dv + dv_idx);
  const device float* iq = pre_q + size_t(n) * Dk;
  const device float* ik = pre_k + size_t(n) * Dk;
  float q[8], k[8];
  for (int i = 0; i < n_per_t; ++i) { int d = n_per_t * dk_idx + i; q[i] = iq[d]; k[i] = ik[d]; }
  float gdecay = pre_gate[2 * n + 0], beta = pre_gate[2 * n + 1];

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
    const constant int& Dk           [[buffer(10)]],
    const constant int& Dv           [[buffer(11)]],
    const constant int& Hk           [[buffer(12)]],
    const constant int& Hv           [[buffer(13)]],
    const constant int& conv_dim     [[buffer(14)]],
    const constant int& Kc           [[buffer(15)]],
    const constant int& q_off        [[buffer(16)]],
    const constant int& k_off        [[buffer(17)]],
    const constant int& v_off        [[buffer(18)]],
    const constant float& eps          [[buffer(19)]],
    const constant float& inv_sqrt_dk  [[buffer(20)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  gdn_core_recurrent_body<T, false>(
      mixed, conv_state, rstate, core_out, conv_w, conv_b,
      pre_q, pre_k, pre_gate, new_conv_state,
      (const device uint*)nullptr,
      Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps, inv_sqrt_dk, tpig, simd_lane);
}

template <typename T>
[[kernel]] void gdn_prep_slotted(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    const device T* conv_w [[buffer(2)]], const device T* conv_b [[buffer(3)]],
    const device float* A_log [[buffer(4)]], const device T* dt_bias [[buffer(5)]],
    const device T* a_gate [[buffer(6)]], const device T* b_gate [[buffer(7)]],
    device float* pre_q [[buffer(8)]], device float* pre_k [[buffer(9)]],
    device float* pre_gate [[buffer(10)]], device float* new_conv_state [[buffer(11)]],
    const device uint* slot_ids [[buffer(12)]],
    const constant int& Dk           [[buffer(13)]],
    const constant int& Dv           [[buffer(14)]],
    const constant int& Hk           [[buffer(15)]],
    const constant int& Hv           [[buffer(16)]],
    const constant int& conv_dim     [[buffer(17)]],
    const constant int& Kc           [[buffer(18)]],
    const constant int& q_off        [[buffer(19)]],
    const constant int& k_off        [[buffer(20)]],
    const constant int& v_off        [[buffer(21)]],
    const constant float& eps          [[buffer(22)]],
    const constant float& inv_sqrt_dk  [[buffer(23)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  gdn_prep_body<T, true>(
      mixed, conv_state, conv_w, conv_b, A_log, dt_bias, a_gate, b_gate,
      pre_q, pre_k, pre_gate, new_conv_state, slot_ids,
      Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps, inv_sqrt_dk, tpig, simd_lane);
}

template <typename T>
[[kernel]] void gdn_core_recurrent_slotted(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    device float* rstate [[buffer(2)]], device T* core_out [[buffer(3)]],
    const device T* conv_w [[buffer(4)]], const device T* conv_b [[buffer(5)]],
    const device float* pre_q [[buffer(6)]], const device float* pre_k [[buffer(7)]],
    const device float* pre_gate [[buffer(8)]], device float* new_conv_state [[buffer(9)]],
    const device uint* slot_ids [[buffer(10)]],

    const constant int& Dk           [[buffer(11)]],
    const constant int& Dv           [[buffer(12)]],
    const constant int& Hk           [[buffer(13)]],
    const constant int& Hv           [[buffer(14)]],
    const constant int& conv_dim     [[buffer(15)]],
    const constant int& Kc           [[buffer(16)]],
    const constant int& q_off        [[buffer(17)]],
    const constant int& k_off        [[buffer(18)]],
    const constant int& v_off        [[buffer(19)]],
    const constant float& eps          [[buffer(20)]],
    const constant float& inv_sqrt_dk  [[buffer(21)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  gdn_core_recurrent_body<T, true>(
      mixed, conv_state, rstate, core_out, conv_w, conv_b,
      pre_q, pre_k, pre_gate, new_conv_state, slot_ids,
      Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps, inv_sqrt_dk, tpig, simd_lane);
}

#define instantiate_gdn_prep(name, itype)                             \
  template [[host_name("gdn_prep_" #name)]] [[kernel]] void           \
  gdn_prep<itype>(                                                    \
      const device itype*, const device float*, const device itype*,  \
      const device itype*, const device float*, const device itype*,  \
      const device itype*, const device itype*, device float*,        \
      device float*, device float*, device float*,                    \
      const constant int&, const constant int&, const constant int&,  \
      const constant int&, const constant int&, const constant int&,  \
      const constant int&, const constant int&, const constant int&,  \
      const constant float&, const constant float&, uint3, uint);     \
  template [[host_name("gdn_core_recurrent_" #name)]] [[kernel]] void \
  gdn_core_recurrent<itype>(                                          \
      const device itype*, const device float*, device float*,        \
      device itype*, const device itype*, const device itype*,        \
      const device float*, const device float*, const device float*,  \
      device float*,                                                  \
      const constant int&, const constant int&, const constant int&,  \
      const constant int&, const constant int&, const constant int&,  \
      const constant int&, const constant int&, const constant int&,  \
      const constant float&, const constant float&, uint3, uint);

#define instantiate_gdn_prep_slotted(name, itype)                             \
  template [[host_name("gdn_prep_slotted_" #name)]] [[kernel]] void           \
  gdn_prep_slotted<itype>(                                                    \
      const device itype*, const device float*, const device itype*,          \
      const device itype*, const device float*, const device itype*,          \
      const device itype*, const device itype*, device float*,                \
      device float*, device float*, device float*, const device uint*,        \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant float&, const constant float&, uint3, uint);             \
  template [[host_name("gdn_core_recurrent_slotted_" #name)]] [[kernel]] void \
  gdn_core_recurrent_slotted<itype>(                                          \
      const device itype*, const device float*, device float*, device itype*, \
      const device itype*, const device itype*, const device float*,          \
      const device float*, const device float*, device float*,                \
      const device uint*,                                                     \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant float&, const constant float&, uint3, uint);

instantiate_gdn_prep(bfloat16, bfloat)
instantiate_gdn_prep_slotted(bfloat16, bfloat)

template <typename T>
[[kernel]] void gdn_prep_prefill(
    const device T* mixed [[buffer(0)]], const device float* conv_state [[buffer(1)]],
    const device T* conv_w [[buffer(2)]], const device T* conv_b [[buffer(3)]],
    const device float* A_log [[buffer(4)]], const device T* dt_bias [[buffer(5)]],
    const device T* a_gate [[buffer(6)]], const device T* b_gate [[buffer(7)]],
    device float* pre_q [[buffer(8)]], device float* pre_k [[buffer(9)]],
    device float* pre_gate [[buffer(10)]], device float* new_conv_state [[buffer(11)]],
    const device uint* slot_ids [[buffer(12)]],
    const constant int& Dk           [[buffer(13)]],
    const constant int& Dv           [[buffer(14)]],
    const constant int& Hk           [[buffer(15)]],
    const constant int& Hv           [[buffer(16)]],
    const constant int& conv_dim     [[buffer(17)]],
    const constant int& Kc           [[buffer(18)]],
    const constant int& q_off        [[buffer(19)]],
    const constant int& k_off        [[buffer(20)]],
    const constant int& v_off        [[buffer(21)]],
    const constant float& eps          [[buffer(22)]],
    const constant float& inv_sqrt_dk  [[buffer(23)]],
    const constant int& row_pitch      [[buffer(24)]],
    const constant int& n_scan         [[buffer(25)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {

  const int CDIM = conv_dim;
  const int n = int(tpig.z), t = n / Hv, hv_idx = n % Hv;

  const int rep = Hv / Hk;
  const int hk_idx = hv_idx / rep;
  const bool hk_first = (hv_idx % rep) == 0;

  const int slot = int(slot_ids[t]), dk_idx = int(tpig.x), n_per_t = Dk / 32;
  int start = t;
  for (int back = 0; back < Kc - 1 && start > 0; ++back) {
    if (int(slot_ids[start - 1]) != slot) break;
    --start;
  }
  const size_t pitch_t = size_t(row_pitch);

  const size_t qk_pitch = size_t(Hv) * size_t(Dk);
  const size_t g_pitch = 2 * size_t(Hv) + size_t(Hv) * size_t(Dv);
  const size_t row_t = size_t(t) * pitch_t;

  auto tap = [&](int j, int c) -> float {
    const int idx = t - (Kc - 1) + j;
    const int local = idx - start;
    return idx >= start ? float(mixed[size_t(idx) * pitch_t + c])
                        : conv_state[(slot * Kc + Kc + local) * CDIM + c];
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
  const float qinv = inv_sqrt_dk / sqrt(qsq + eps);
  const float kinv = 1.0f / sqrt(ksq + eps);
  device float* oq = pre_q + size_t(t) * qk_pitch + size_t(hv_idx) * Dk;
  device float* ok = pre_k + size_t(t) * qk_pitch + size_t(hv_idx) * Dk;
  for (int i = 0; i < n_per_t; ++i) {
    const int d = n_per_t * dk_idx + i;
    oq[d] = qraw[i] * qinv;
    ok[d] = kraw[i] * kinv;
  }
  if (dk_idx == 0) {

    const size_t gate_at = size_t(t) * size_t(Hv) + size_t(hv_idx);
    const float ad = float(a_gate[gate_at]) + float(dt_bias[hv_idx]);
    const float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));
    device float* g = pre_gate + size_t(t) * g_pitch + 2 * size_t(hv_idx);
    g[0] = exp(-exp(float(A_log[hv_idx])) * sp);
    g[1] = 1.0f / (1.0f + exp(-float(b_gate[gate_at])));
  }

  device float* pv = pre_gate + size_t(t) * g_pitch + 2 * size_t(Hv);
  for (int dv = dk_idx; dv < Dv; dv += 32)
    pv[size_t(hv_idx) * Dv + dv] =
        convsilu(v_off + hv_idx * Dv + dv);

  if (t != n_scan - 1 && int(slot_ids[t + 1]) == slot) return;
  auto wb = [&](int c) {
    for (int j = 0; j < Kc; ++j) {
      const int idx = t - (Kc - 1) + j;
      const int local = idx - start;
      new_conv_state[(slot * Kc + j) * CDIM + c] =
          idx >= start ? float(mixed[size_t(idx) * pitch_t + c])
                       : conv_state[(slot * Kc + Kc + local) * CDIM + c];
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

template <int LANES>
METAL_FUNC float gdn_row_sum(float v) {
  v += simd_shuffle_xor(v, 1u);
  if (LANES >= 4) v += simd_shuffle_xor(v, 2u);
  if (LANES >= 8) v += simd_shuffle_xor(v, 4u);
  if (LANES >= 16) v += simd_shuffle_xor(v, 8u);
  if (LANES >= 32) v += simd_shuffle_xor(v, 16u);
  return v;
}

template <typename T, int LANES, int VROWS = 1>
[[kernel]] void gdn_core_recurrent_prefill(
    device float* rstate [[buffer(2)]], device T* core_out [[buffer(3)]],
    const device float* pre_q [[buffer(6)]], const device float* pre_k [[buffer(7)]],
    const device float* pre_gate [[buffer(8)]],
    const device uint* slot_ids [[buffer(10)]],
    const constant int& Dk           [[buffer(11)]],
    const constant int& Dv           [[buffer(12)]],
    const constant int& Hk           [[buffer(13)]],
    const constant int& Hv           [[buffer(14)]],
    const constant int& conv_dim     [[buffer(15)]],
    const constant int& Kc           [[buffer(16)]],
    const constant int& q_off        [[buffer(17)]],
    const constant int& k_off        [[buffer(18)]],
    const constant int& v_off        [[buffer(19)]],
    const constant float& eps          [[buffer(20)]],
    const constant float& inv_sqrt_dk  [[buffer(21)]],
    const constant int& row_pitch      [[buffer(22)]],
    const constant int& n_scan         [[buffer(23)]],
    uint3 tpig [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {

  constexpr int ROWS = 32 / LANES;
  constexpr int MAX_PER_T = 128 / LANES;
  const int hv_idx = int(tpig.z);
  const int dv_base = (int(tpig.y) * ROWS + (int(tpig.x) / LANES)) * VROWS;
  const int dk_idx = int(tpig.x) % LANES;
  const int n_per_t = Dk / LANES;

  const size_t qk_pitch = size_t(Hv) * size_t(Dk);
  const size_t g_pitch = 2 * size_t(Hv) + size_t(Hv) * size_t(Dv);
  const size_t o_pitch = size_t(Hv) * size_t(Dv);
  const bool row_lead = dk_idx == 0;
  if (dv_base >= Dv) return;

  const int vn = min(VROWS, Dv - dv_base);

  int slot = int(slot_ids[0]);
  device float* i_state = rstate + (size_t((slot * Hv + hv_idx) * Dv + dv_base) * Dk);
  float st[VROWS][MAX_PER_T];
  auto load = [&]() {
    i_state = rstate + (size_t((slot * Hv + hv_idx) * Dv + dv_base) * Dk);
    for (int v = 0; v < vn; ++v)
      for (int i = 0; i < n_per_t; ++i)
        st[v][i] = i_state[size_t(v) * Dk + n_per_t * dk_idx + i];
  };
  auto store = [&]() {
    for (int v = 0; v < vn; ++v)
      for (int i = 0; i < n_per_t; ++i)
        i_state[size_t(v) * Dk + n_per_t * dk_idx + i] = st[v][i];
  };
  load();

  for (int t = 0; t < n_scan; ++t) {
    const int seat = int(slot_ids[t]);
    if (seat != slot) {
      store();
      slot = seat;
      load();
    }
    const size_t row_t = size_t(t) * o_pitch;
    const device float* iv = pre_gate + size_t(t) * g_pitch + 2 * size_t(Hv) +
                             size_t(hv_idx) * Dv + dv_base;

    const device float* iq = pre_q + size_t(t) * qk_pitch + size_t(hv_idx) * Dk;
    const device float* ik = pre_k + size_t(t) * qk_pitch + size_t(hv_idx) * Dk;
    const device float* g = pre_gate + size_t(t) * g_pitch + 2 * size_t(hv_idx);
    float q[MAX_PER_T], k[MAX_PER_T];
    for (int i = 0; i < n_per_t; ++i) {
      const int d = n_per_t * dk_idx + i;
      q[i] = iq[d];
      k[i] = ik[d];
    }
    const float ga = g[0], gb = g[1];
    float kv_mem[VROWS];
    for (int v = 0; v < vn; ++v) {
      float acc = 0.0f;
      for (int i = 0; i < n_per_t; ++i) { st[v][i] *= ga; acc += st[v][i] * k[i]; }
      kv_mem[v] = gdn_row_sum<LANES>(acc);
    }
    for (int v = 0; v < vn; ++v) {
      const float delta = (iv[v] - kv_mem[v]) * gb;
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) { st[v][i] += k[i] * delta; out += st[v][i] * q[i]; }
      out = gdn_row_sum<LANES>(out);
      if (row_lead)
        core_out[row_t + size_t(hv_idx) * Dv + dv_base + v] = static_cast<T>(out);
    }
  }
  store();
}

#define instantiate_gdn_prefill(name, itype)                                  \
  template [[host_name("gdn_prep_prefill_" #name)]] [[kernel]] void           \
  gdn_prep_prefill<itype>(                                                    \
      const device itype*, const device float*, const device itype*,          \
      const device itype*, const device float*, const device itype*,          \
      const device itype*, const device itype*, device float*, device float*, \
      device float*, device float*, const device uint*,                       \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant float&, const constant float&,                           \
      const constant int&, const constant int&, uint3, uint);

#define instantiate_gdn_scan(name, itype, lanes, vrows)                       \
  template [[host_name("gdn_core_recurrent_prefill_" #name "_l_" #lanes       \
                       "_v_" #vrows)]]                                        \
  [[kernel]] void gdn_core_recurrent_prefill<itype, lanes, vrows>(            \
      device float*, device itype*, const device float*, const device float*, \
      const device float*, const device uint*,                                \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const constant int&,          \
      const constant float&, const constant float&,                           \
      const constant int&, const constant int&, uint3, uint);

instantiate_gdn_prefill(bfloat16, bfloat)
instantiate_gdn_scan(bfloat16, bfloat, 16, 1)
instantiate_gdn_scan(bfloat16, bfloat, 16, 2)
instantiate_gdn_scan(bfloat16, bfloat, 16, 4)
instantiate_gdn_scan(bfloat16, bfloat, 8, 1)
instantiate_gdn_scan(bfloat16, bfloat, 8, 2)
instantiate_gdn_scan(bfloat16, bfloat, 4, 1)
instantiate_gdn_scan(bfloat16, bfloat, 32, 2)
instantiate_gdn_scan(bfloat16, bfloat, 32, 4)
instantiate_gdn_scan(bfloat16, bfloat, 32, 8)

template <typename T>
[[kernel]] void qwen_gdn_ba_gates(
    const device T* ba          [[buffer(0)]],
    const device float* a_log   [[buffer(1)]],
    const device T* dt_bias     [[buffer(2)]],
    device float* gates         [[buffer(3)]],
    const constant int& v_heads [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  const uint h = pos.x;
  const uint t = pos.y;
  const size_t vh = size_t(v_heads);
  const size_t row = size_t(t) * 2 * vh;

  const float bv = float(ba[row + size_t(h)]);
  const float av = float(ba[row + vh + size_t(h)]);

  const float z = av + float(dt_bias[h]);
  const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::fast::exp(z));

  gates[row + size_t(h)] = -metal::fast::exp(a_log[h]) * sp;
  gates[row + vh + size_t(h)] = 1.0f / (1.0f + metal::fast::exp(-bv));
}

#define instantiate_qwen_gdn_ba_gates(name, itype)                            \
  template [[host_name("qwen_gdn_ba_gates_" #name)]]                          \
  [[kernel]] void qwen_gdn_ba_gates<itype>(                                   \
      const device itype*, const device float*, const device itype*,          \
      device float*, const constant int&, uint2);

instantiate_qwen_gdn_ba_gates(bfloat16, bfloat)
