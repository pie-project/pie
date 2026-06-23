// Raw-Metal fused GDN-core for Phase-0 decode (M=1, one token row).
//
// Fuses the ENTIRE gated-delta-net core into ONE dispatch (decode_abi GDN-core=1):
//   conv1d(+silu) + split(q,k,v) + l2norm(q,k) + q-scale(1/sqrt(Dk))
//   + gating(g = -exp(A_log)*softplus(a+dt_bias); decay=exp(g); beta=sigmoid(b))
//   + recurrent step (register state + simd_sum) + conv_state shift/append.
//
// Port of beta's validated MLX metal_kernel (gdn_core_fused.cpp), bit-exact
// (core_out 4.5e-8 / state 7.5e-7 / conv_state 0.0) vs the decomposed MLX op
// (gated_delta.cpp::gdn_decode_region, pre-GatedRms = charlie's `gdn_core_pre`).
//
// Threadgroup design (R=1, T=1 decode):
//   grid = {32, Vd, R*Hv}   tg = {32, 4, 1}
//   one simdgroup per (req,v-head,v-dim) row; its 32 dk-lanes cooperatively
//   compute conv+l2norm+recurrent. l2norm/kv/out reductions via simd_sum over
//   the head's Dk dims (Dk=128 = 32 lanes x n_per_t=4). No threadgroup barrier.
//   The conv/l2norm/gating prologue is recomputed per v-dim tile (Vd-fold
//   redundant) but trivial vs the recurrent step -> 1 dispatch beats a split
//   prep dispatch (saves a barrier + the q/k/v/g/beta global round-trip).
//
// Port notes (per decode_abi invariants):
//   * Geometry (Dk/Dv/Hk/Hv/conv_dim/Kc/offsets/eps/inv_sqrt_dk) is STATIC model
//     geometry -> constant GdnCoreParams& buffer, NOT per-token IO (I1 safe).
//   * recurrent_state is consumed/produced NATIVE [R,Hv,Vd,Dk] (prong-1, no
//     swapaxes) and rebound in-place to the same heap slot (I4): each (req,v-head,
//     v-dim) row is owned by exactly one threadgroup, so read-then-write is race-free.
//   * conv_state CANNOT be in-place: convsilu reads the Kc-tap history while wb()
//     shifts it, and the redundant v-dim threadgroups interleave those reads/writes.
//     So conv_state is READ-ONLY input + a SEPARATE new_conv_state output (ping-pong,
//     delta swaps the two heap slots per token). ABI co-fix: bind::GdnCore needs a
//     ConvStateOut slot (was "ConvState in-place").
//   * GQA: this model (qwen3.5-0.8B) has Hk==Hv (rep=1), q/k index by hv directly.
//     For rep>1, index hk = hv/(Hv/Hk); see KSTRIDE note below.
//   * bfloat native on Metal 4; recurrent state stays fp32 for accuracy.

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

// T  = core_out dtype (bf16/half/float); recurrent state is always fp32.
template <typename T>
[[kernel]] void gdn_core(
    const device T*     mixed          [[buffer(0)]],   // [R, conv_dim]  this token's in-proj mixed_qkv
    const device float* conv_state     [[buffer(1)]],   // [R, Kc, conv_dim]  READ-ONLY conv history
    device float*       rstate         [[buffer(2)]],   // [R, Hv, Vd, Dk]  in-place native recurrent state
    device T*           core_out       [[buffer(3)]],   // [R, Hv, Vd]  pre-GatedRms output
    const device T*     conv_w         [[buffer(4)]],   // [conv_dim, Kc]
    const device T*     conv_b         [[buffer(5)]],   // [conv_dim]
    const device float* A_log          [[buffer(6)]],   // [Hv]  F32 in ckpt — typed float* (bit-exact decay; raw zero-copy stage)
    const device T*     dt_bias        [[buffer(7)]],   // [Hv]
    const device T*     a_gate         [[buffer(8)]],   // [R, Hv]
    const device T*     b_gate         [[buffer(9)]],   // [R, Hv]
    device float*       new_conv_state [[buffer(10)]],  // [R, Kc, conv_dim]  shifted history (ping-pong, != conv_state)
    constant GdnCoreParams& p          [[buffer(11)]],
    uint3 tpig                         [[thread_position_in_grid]],
    uint3 tpit                         [[thread_position_in_threadgroup]],
    uint  simd_lane                    [[thread_index_in_simdgroup]]) {
  const int Dk = p.Dk, Dv = p.Dv, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n        = int(tpig.z);          // 0 .. R*Hv-1
  const int b_idx    = n / Hv;
  const int hv_idx   = n % Hv;
  const int dk_idx   = int(tpit.x);          // 0..31
  const int dv_idx   = int(tpig.y);          // 0..Vd-1
  const int n_per_t  = Dk / 32;              // 4
  const int q_off = p.q_off, k_off = p.k_off, v_off = p.v_off;

  // --- conv1d(+silu) over a channel's Kc-tap window (history + this token) ---
  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(b_idx * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));          // silu
  };

  // this lane owns dk channels [n_per_t*dk_idx, +n_per_t) of head hv_idx (q and k),
  // and the single v channel (hv_idx, dv_idx).
  float qraw[8], kraw[8];                      // n_per_t<=8
  for (int i = 0; i < n_per_t; ++i) {
    int d = n_per_t * dk_idx + i;              // 0..Dk-1
    qraw[i] = convsilu(q_off + hv_idx * Dk + d);
    kraw[i] = convsilu(k_off + hv_idx * Dk + d);
  }
  float vval = convsilu(v_off + hv_idx * Dv + dv_idx);

  // --- l2norm(q,k) over the head's Dk dims (simd_sum across the 32 dk-lanes) ---
  float qsq = 0.0f, ksq = 0.0f;
  for (int i = 0; i < n_per_t; ++i) { qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i]; }
  qsq = simd_sum(qsq); ksq = simd_sum(ksq);
  float qinv = p.inv_sqrt_dk / sqrt(qsq + p.eps);   // q also pre-scaled 1/sqrt(Dk)
  float kinv = 1.0f / sqrt(ksq + p.eps);
  float q[8], k[8];
  for (int i = 0; i < n_per_t; ++i) { q[i] = qraw[i] * qinv; k[i] = kraw[i] * kinv; }

  // --- gating: g = -exp(A_log)*softplus(a+dt_bias); decay = exp(g); beta = sigmoid(b) ---
  float ad = float(a_gate[b_idx * Hv + hv_idx]) + float(dt_bias[hv_idx]);
  float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));   // softplus
  float gdecay = exp(-exp(float(A_log[hv_idx])) * sp);
  float beta = 1.0f / (1.0f + exp(-float(b_gate[b_idx * Hv + hv_idx])));

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

  // --- conv_state writeback (shift + append) to a SEPARATE ping-pong slot. ---
  // Reading conv_state (read-only) and writing new_conv_state avoids the
  // read/write race the redundant v-dim threadgroups would hit if in-place.
  // Idempotent across redundant lanes (all write identical values from stable input).
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
  wb(v_off + hv_idx * Dv + dv_idx);
}

#define instantiate_gdn_core(name, itype)                                  \
  template [[host_name("gdn_core_" #name)]] [[kernel]] void                \
  gdn_core<itype>(                                                         \
      const device itype*, const device float*, device float*,             \
      device itype*, const device itype*, const device itype*,             \
      const device float*, const device itype*, const device itype*,       \
      const device itype*, device float*,                                  \
      constant GdnCoreParams&, uint3, uint3, uint);

instantiate_gdn_core(float32, float)
instantiate_gdn_core(float16, half)
instantiate_gdn_core(bfloat16, bfloat)
