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
// ── M>1 slot-indexed state seam (delta's GDN bridge, ckpt 030) ───────────────
// For S>1 the conv/recurrent state slabs are keyed by rs_slot_ids[r], NOT by the
// token's contiguous row. ONLY state accesses remap b_idx→slot; activations
// (mixed/core_out/a_gate/b_gate) stay token-major b_idx. `SLOTTED` selects:
//   false  →  slot = b_idx  (the sealed M=1 path; slot_ids never read → byte-identical 264).
//   true   →  slot = slot_ids[b_idx]  (S>1; state slabs sized max_slots, delta owns host side).
// Both slab strides derive from GdnCoreParams (Hv/Dv/Dk/Kc/CDIM) — slot_ids alone suffices.
template <typename T, bool SLOTTED>
METAL_FUNC void gdn_core_body(
    const device T*     mixed,          // [N, conv_dim]  this token's in-proj mixed_qkv
    const device float* conv_state,     // [max_slots, Kc, conv_dim]  READ-ONLY conv history
    device float*       rstate,         // [max_slots, Hv, Vd, Dk]  in-place native recurrent state
    device T*           core_out,       // [N, Hv, Vd]  pre-GatedRms output
    const device T*     conv_w,         // [conv_dim, Kc]
    const device T*     conv_b,         // [conv_dim]
    const device float* A_log,          // [Hv]  F32 in ckpt
    const device T*     dt_bias,        // [Hv]
    const device T*     a_gate,         // [N, Hv]
    const device T*     b_gate,         // [N, Hv]
    device float*       new_conv_state, // [max_slots, Kc, conv_dim]  shifted history (ping-pong)
    constant GdnCoreParams& p,
    const device uint*  slot_ids,       // [N]  rs_slot per token row (read only when SLOTTED)
    threadgroup float*  sh_q,           // [Dk]  shared normalized+prescaled q (tg-allocated by wrapper)
    threadgroup float*  sh_k,           // [Dk]  shared normalized k
    threadgroup float*  sh_decay,       // [1]   shared per-head decay
    threadgroup float*  sh_beta,        // [1]   shared per-head beta
    uint3 tpig, uint3 tpit, uint simd_lane) {
  const int Dk = p.Dk, Dv = p.Dv, Hv = p.Hv, CDIM = p.conv_dim, Kc = p.Kc;
  const int n        = int(tpig.z);          // 0 .. N*Hv-1
  const int b_idx    = n / Hv;               // token row (activation index)
  const int hv_idx   = n % Hv;
  const int dk_idx   = int(tpit.x);          // 0..31
  const int dv_idx   = int(tpig.y);          // 0..Vd-1
  const int n_per_t  = Dk / 32;              // 4
  const int q_off = p.q_off, k_off = p.k_off, v_off = p.v_off;
  // STATE slab index: remapped to the request's persistent slot when SLOTTED.
  const int slot = SLOTTED ? int(slot_ids[b_idx]) : b_idx;

  // --- conv1d(+silu) over a channel's Kc-tap window (history + this token) ---
  // history reads from the SLOT's conv slab; the new token (`mixed`) is the token row.
  auto convsilu = [&](int c) -> float {
    float acc = float(conv_b[c]);
    for (int j = 0; j < Kc - 1; ++j)
      acc += conv_state[(slot * Kc + (j + 1)) * CDIM + c] * float(conv_w[c * Kc + j]);
    acc += float(mixed[b_idx * CDIM + c]) * float(conv_w[c * Kc + (Kc - 1)]);
    return acc / (1.0f + exp(-acc));          // silu
  };

  // --- q/k conv/silu + l2norm + gating: depend only on (hv,dk), NOT dv ---------
  // Identical for every dv of the head, so compute ONCE per threadgroup (the
  // simdgroup tpit.y==0) into threadgroup memory and share across the tile's dv
  // simdgroups — kills the Vd-fold redundancy (was recomputed in all 128 dv
  // threadgroups). The threadgroup spans a tile of dv (tg.y simdgroups) so this
  // dedups tg.y× while keeping full occupancy. Bit-identical: same simd_sum
  // reduction, same values, just computed once and broadcast.
  // The shared q/k/gating threadgroup buffers (sh_q/sh_k/sh_decay/sh_beta) are
  // declared in the [[kernel]] wrapper and passed in (Metal forbids threadgroup
  // address-space decls inside a non-kernel helper).
  if (tpit.y == 0) {
    float qraw[8], kraw[8];                    // n_per_t<=8
    for (int i = 0; i < n_per_t; ++i) {
      int d = n_per_t * dk_idx + i;            // 0..Dk-1
      qraw[i] = convsilu(q_off + hv_idx * Dk + d);
      kraw[i] = convsilu(k_off + hv_idx * Dk + d);
    }
    float qsq = 0.0f, ksq = 0.0f;
    for (int i = 0; i < n_per_t; ++i) { qsq += qraw[i] * qraw[i]; ksq += kraw[i] * kraw[i]; }
    qsq = simd_sum(qsq); ksq = simd_sum(ksq);
    float qinv = p.inv_sqrt_dk / sqrt(qsq + p.eps);   // q also pre-scaled 1/sqrt(Dk)
    float kinv = 1.0f / sqrt(ksq + p.eps);
    for (int i = 0; i < n_per_t; ++i) {
      int d = n_per_t * dk_idx + i;
      sh_q[d] = qraw[i] * qinv;
      sh_k[d] = kraw[i] * kinv;
    }
    if (dk_idx == 0) {
      // gating: g = -exp(A_log)*softplus(a+dt_bias); decay = exp(g); beta = sigmoid(b)
      float ad = float(a_gate[b_idx * Hv + hv_idx]) + float(dt_bias[hv_idx]);
      float sp = max(ad, 0.0f) + log(1.0f + exp(-fabs(ad)));   // softplus
      sh_decay[0] = exp(-exp(float(A_log[hv_idx])) * sp);
      sh_beta[0]  = 1.0f / (1.0f + exp(-float(b_gate[b_idx * Hv + hv_idx])));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // this lane's v channel (unique per dv) + read the shared q/k/gating.
  float vval = convsilu(v_off + hv_idx * Dv + dv_idx);
  float q[8], k[8];
  for (int i = 0; i < n_per_t; ++i) { int d = n_per_t * dk_idx + i; q[i] = sh_q[d]; k[i] = sh_k[d]; }
  float gdecay = sh_decay[0], beta = sh_beta[0];

  // --- recurrent step (register state + simd_sum), native [Hv,Vd,Dk] ---
  // State row is keyed by SLOT (persistent per-request slab), not the token row.
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

  // --- conv_state writeback (shift + append) to a SEPARATE ping-pong slot. ---
  // Reading conv_state (read-only) and writing new_conv_state avoids the
  // read/write race the redundant v-dim threadgroups would hit if in-place.
  // q/k channels depend only on (hv,dk) — identical across all Vd threadgroups of
  // a head — so only dv_idx==0 writes them (was Vd=128-fold redundant write
  // traffic, ~8MB/token). The v channel is unique per (hv,dv): every dv writes its
  // own. Full coverage, each channel written exactly once. Output bit-identical.
  auto wb = [&](int c) {
    for (int j = 0; j < Kc - 1; ++j)
      new_conv_state[(slot * Kc + j) * CDIM + c] = conv_state[(slot * Kc + (j + 1)) * CDIM + c];
    new_conv_state[(slot * Kc + (Kc - 1)) * CDIM + c] = float(mixed[b_idx * CDIM + c]);
  };
  if (dv_idx == 0) {
    for (int i = 0; i < n_per_t; ++i) {
      int d = n_per_t * dk_idx + i;
      wb(q_off + hv_idx * Dk + d);
      wb(k_off + hv_idx * Dk + d);
    }
  }
  wb(v_off + hv_idx * Dv + dv_idx);
}

// ── M=1 sealed entry: 12 buffers, NO slot_ids — byte-identical PSO (264 holds). ──
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
    constant GdnCoreParams& p          [[buffer(11)]],
    uint3 tpig                         [[thread_position_in_grid]],
    uint3 tpit                         [[thread_position_in_threadgroup]],
    uint  simd_lane                    [[thread_index_in_simdgroup]]) {
  threadgroup float sh_q[256], sh_k[256], sh_decay[1], sh_beta[1];
  gdn_core_body<T, false>(mixed, conv_state, rstate, core_out, conv_w, conv_b,
                          A_log, dt_bias, a_gate, b_gate, new_conv_state, p,
                          (const device uint*)nullptr,
                          sh_q, sh_k, sh_decay, sh_beta, tpig, tpit, simd_lane);
}

// ── M>1 slotted entry: +slot_ids[buffer(12)] (alpha GdnCore::SlotIds=12). ──
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
    constant GdnCoreParams& p          [[buffer(11)]],
    const device uint*  slot_ids       [[buffer(12)]],
    uint3 tpig                         [[thread_position_in_grid]],
    uint3 tpit                         [[thread_position_in_threadgroup]],
    uint  simd_lane                    [[thread_index_in_simdgroup]]) {
  threadgroup float sh_q[256], sh_k[256], sh_decay[1], sh_beta[1];
  gdn_core_body<T, true>(mixed, conv_state, rstate, core_out, conv_w, conv_b,
                         A_log, dt_bias, a_gate, b_gate, new_conv_state, p,
                         slot_ids,
                         sh_q, sh_k, sh_decay, sh_beta, tpig, tpit, simd_lane);
}

#define instantiate_gdn_core(name, itype)                                  \
  template [[host_name("gdn_core_" #name)]] [[kernel]] void                \
  gdn_core<itype>(                                                         \
      const device itype*, const device float*, device float*,             \
      device itype*, const device itype*, const device itype*,             \
      const device float*, const device itype*, const device itype*,       \
      const device itype*, device float*,                                  \
      constant GdnCoreParams&, uint3, uint3, uint);                        \
  template [[host_name("gdn_core_slotted_" #name)]] [[kernel]] void        \
  gdn_core_slotted<itype>(                                                 \
      const device itype*, const device float*, device float*,             \
      device itype*, const device itype*, const device itype*,             \
      const device float*, const device itype*, const device itype*,       \
      const device itype*, device float*,                                  \
      constant GdnCoreParams&, const device uint*, uint3, uint3, uint);

instantiate_gdn_core(float32, float)
instantiate_gdn_core(float16, half)
instantiate_gdn_core(bfloat16, bfloat)
