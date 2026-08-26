#include <metal_stdlib>
using namespace metal;

template <typename T, bool SLOTTED>
METAL_FUNC void gdn_core_body(
    const device T*     mixed,
    const device float* conv_state,
    device float*       rstate,
    device T*           core_out,
    const device T*     conv_w,
    const device T*     conv_b,
    const device float* A_log,
    const device T*     dt_bias,
    const device T*     a_gate,
    const device T*     b_gate,
    device float*       new_conv_state,
    const device uint*  slot_ids,
    int Dk, int Dv, int Hk, int Hv, int conv_dim, int Kc,
    int q_off, int k_off, int v_off, float eps, float inv_sqrt_dk,
    threadgroup float*  sh_q,
    threadgroup float*  sh_k,
    threadgroup float*  sh_decay,
    threadgroup float*  sh_beta,
    uint3 tpig, uint3 tpit, uint simd_lane) {

  const int CDIM = conv_dim;
  const int n        = int(tpig.z);
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;

  const int rep      = Hv / Hk;
  const int hk_idx   = hv_idx / rep;
  const bool hk_first = (hv_idx % rep) == 0;
  const int dk_idx   = int(tpit.x);
  const int dv_idx   = int(tpig.y);
  const int n_per_t  = Dk / 32;

  const int slot = SLOTTED ? int(slot_ids[b_idx]) : b_idx;

  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));
  };

  if (tpit.y == 0) {
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
    for (int i = 0; i < n_per_t; ++i) {
      int d = n_per_t * dk_idx + i;
      sh_q[d] = qraw[i] * qinv;
      sh_k[d] = kraw[i] * kinv;
    }
    if (dk_idx == 0) {

      float ad = float(a_gate[b_idx * Hv + hv_idx]) + float(dt_bias[hv_idx]);
      float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));
      sh_decay[0] = exp(-exp(float(A_log[hv_idx])) * sp);
      sh_beta[0]  = 1.0f / (1.0f + exp(-float(b_gate[b_idx * Hv + hv_idx])));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float vval = convsilu(v_off + hv_idx * Dv + dv_idx);
  float q[8], k[8];
  for (int i = 0; i < n_per_t; ++i) { int d = n_per_t * dk_idx + i; q[i] = sh_q[d]; k[i] = sh_k[d]; }
  float gdecay = sh_decay[0], beta = sh_beta[0];

  device float* i_state = rstate + (size_t((slot * Hv + hv_idx) * Dv + dv_idx) * Dk);
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
      new_conv_state[(slot * Kc + j) * CDIM + c] = conv_state[(slot * Kc + (j + 1)) * CDIM + c];
    new_conv_state[(slot * Kc + (Kc - 1)) * CDIM + c] = float(mixed[b_idx * CDIM + c]);
  };
  if (dv_idx == 0 && hk_first) {
    for (int i = 0; i < n_per_t; ++i) {
      int d = n_per_t * dk_idx + i;
      wb(q_off + hk_idx * Dk + d);
      wb(k_off + hk_idx * Dk + d);
    }
  }
  wb(v_off + hv_idx * Dv + dv_idx);
}

template <typename T>
[[kernel]] void gdn_core(
    const device T*     mixed          [[buffer(0)]],
    const device float* conv_state     [[buffer(1)]],
    device float*       rstate         [[buffer(2)]],
    device T*           core_out       [[buffer(3)]],
    const device T*     conv_w         [[buffer(4)]],
    const device T*     conv_b         [[buffer(5)]],
    const device float* A_log          [[buffer(6)]],
    const device T*     dt_bias        [[buffer(7)]],
    const device T*     a_gate         [[buffer(8)]],
    const device T*     b_gate         [[buffer(9)]],
    device float*       new_conv_state [[buffer(10)]],
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
    uint3 tpig                         [[thread_position_in_grid]],
    uint3 tpit                         [[thread_position_in_threadgroup]],
    uint  simd_lane                    [[thread_index_in_simdgroup]]) {
  threadgroup float sh_q[256], sh_k[256], sh_decay[1], sh_beta[1];
  gdn_core_body<T, false>(mixed, conv_state, rstate, core_out, conv_w, conv_b,
                          A_log, dt_bias, a_gate, b_gate, new_conv_state,
                          (const device uint*)nullptr,
                          Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps, inv_sqrt_dk,
                          sh_q, sh_k, sh_decay, sh_beta, tpig, tpit, simd_lane);
}

template <typename T>
[[kernel]] void gdn_core_slotted(
    const device T*     mixed          [[buffer(0)]],
    const device float* conv_state     [[buffer(1)]],
    device float*       rstate         [[buffer(2)]],
    device T*           core_out       [[buffer(3)]],
    const device T*     conv_w         [[buffer(4)]],
    const device T*     conv_b         [[buffer(5)]],
    const device float* A_log          [[buffer(6)]],
    const device T*     dt_bias        [[buffer(7)]],
    const device T*     a_gate         [[buffer(8)]],
    const device T*     b_gate         [[buffer(9)]],
    device float*       new_conv_state [[buffer(10)]],
    const device uint*  slot_ids       [[buffer(11)]],
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
    uint3 tpig                         [[thread_position_in_grid]],
    uint3 tpit                         [[thread_position_in_threadgroup]],
    uint  simd_lane                    [[thread_index_in_simdgroup]]) {
  threadgroup float sh_q[256], sh_k[256], sh_decay[1], sh_beta[1];
  gdn_core_body<T, true>(mixed, conv_state, rstate, core_out, conv_w, conv_b,
                         A_log, dt_bias, a_gate, b_gate, new_conv_state,
                         slot_ids,
                         Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps, inv_sqrt_dk,
                         sh_q, sh_k, sh_decay, sh_beta, tpig, tpit, simd_lane);
}

#define instantiate_gdn_core(name, itype)                            \
  template [[host_name("gdn_core_" #name)]] [[kernel]] void          \
  gdn_core<itype>(                                                   \
      const device itype*, const device float*, device float*,       \
      device itype*, const device itype*, const device itype*,       \
      const device float*, const device itype*, const device itype*, \
      const device itype*, device float*,                            \
      const constant int&, const constant int&, const constant int&, \
      const constant int&, const constant int&, const constant int&, \
      const constant int&, const constant int&, const constant int&, \
      const constant float&, const constant float&,                  \
      uint3, uint3, uint);                                           \
  template [[host_name("gdn_core_slotted_" #name)]] [[kernel]] void  \
  gdn_core_slotted<itype>(                                           \
      const device itype*, const device float*, device float*,       \
      device itype*, const device itype*, const device itype*,       \
      const device float*, const device itype*, const device itype*, \
      const device itype*, device float*, const device uint*,        \
      const constant int&, const constant int&, const constant int&, \
      const constant int&, const constant int&, const constant int&, \
      const constant int&, const constant int&, const constant int&, \
      const constant float&, const constant float&,                  \
      uint3, uint3, uint);

instantiate_gdn_core(bfloat16, bfloat)
