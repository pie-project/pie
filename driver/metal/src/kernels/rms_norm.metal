// Raw-Metal port of MLX rms_single_row, scoped to Phase-0 decode (M=1, one token row).
//
// Source: mlx/backend/metal/kernels/rms_norm.metal (rms_single_row).
// Port notes:
//   * Decode is single-row -> only the single_row variant is needed (no rms_looped,
//     no vjp/backward kernels). gid selects the row; for B=1 decode gid==0.
//   * eps/axis_size/w_stride are STATIC geometry (hidden=1024, w_stride=1, eps=1e-6),
//     NOT per-token IO scalars, so they may stay as a constant params buffer with no
//     impact on the byte-identical-CB / encode-overlap invariant (decode_abi I1).
//   * bfloat is native on Metal 4 (macOS 26) -> MLX's bf16.h emulation is dropped.
//   * Output is RMSNorm(x) * w, matching MLX semantics (weight always applied here).
//   * qwen3.5/qwen36 uses the Gemma `(1 + weight)` convention for EVERY RMSNorm
//     (attn_norm, q_norm, k_norm, ffn_norm, final_norm — all plus_one=true; see
//     model/qwen36.cpp + ops/norm.cpp). MLX folds the +1 by materializing
//     `mx::add(weight, 1.0f)` (float) before fast::rms_norm, so the effective gain
//     is float (1.0f + weight); we apply it in float here when p.plus_one != 0.
//     The gated-RMSNorm (gated_rms.metal) does NOT use +1 (raw gate_norm weight).

#include <metal_stdlib>
using namespace metal;

#include "rms_params.h"
#include "rms_reduce.h"

template <typename T, int N_READS>
METAL_FUNC void rms_row_body(
    const device T* x, const device T* w, device T* out,
    constant RmsParams& p, size_t row_base,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint lid, uint simd_lane, uint simd_group) {
  const uint axis_size = p.axis_size;
  const uint w_stride = p.w_stride;

  x += row_base + lid * N_READS;
  w += w_stride * lid * N_READS;
  const float inv = rms_inv_from_lane_sum(
      rms_lane_square_sum<T, N_READS>(x, axis_size, lid),
      axis_size, p.eps, inv_rms, partials, simd_lane, simd_group);

  out += row_base + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      T wv = T(p.gain * (p.plus_one ? (1.0f + float(w[w_stride * i]))
                                    : float(w[w_stride * i])));
      out[i] = wv * static_cast<T>(x[i] * inv);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        T wv = T(p.gain * (p.plus_one ? (1.0f + float(w[w_stride * i]))
                                      : float(w[w_stride * i])));
        out[i] = wv * static_cast<T>(x[i] * inv);
      }
    }
  }
}

template <typename T, int N_READS>
[[kernel]] void rms_single_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    constant RmsParams& p      [[buffer(3)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, p, size_t(gid) * p.axis_size,
      inv_rms, partials, lid, simd_lane, simd_group);
}

// Prefill variant: the prompt's scratch rows are a uniform `row_pitch` elements
// apart (the widest tensor in the layout), not `axis_size`, so a whole prompt can
// run as one dispatch instead of one per token.  Arithmetic is byte-identical to
// `rms_single_row` -- only the row's base address is computed differently.
template <typename T, int N_READS>
[[kernel]] void rms_strided_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    constant RmsParams& p      [[buffer(3)]],
    constant int& row_pitch    [[buffer(4)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, p, size_t(gid) * size_t(row_pitch),
      inv_rms, partials, lid, simd_lane, simd_group);
}

#define instantiate_rms_strided_row(name, itype, n_reads)              \
  template [[host_name("rms_strided_row_" #name)]] [[kernel]] void      \
  rms_strided_row<itype, n_reads>(                                      \
      const device itype*, const device itype*, device itype*,          \
      constant RmsParams&, constant int&, uint, uint, uint, uint);

instantiate_rms_strided_row(bfloat16, bfloat, 4)

#define instantiate_rms_single_row(name, itype, n_reads)               \
  template [[host_name("rms_single_row_" #name)]] [[kernel]] void       \
  rms_single_row<itype, n_reads>(                                       \
      const device itype*, const device itype*, device itype*,          \
      constant RmsParams&, uint, uint, uint, uint);

instantiate_rms_single_row(bfloat16, bfloat, 4)

// ── Fused norm + residual (+ optional layer scalar) — gemma4 ─────────────────
//
// gemma4 wraps each sublayer in a norm sandwich: the BLOCK's output is
// normalised before it rejoins the residual stream. So `rms_single_row` is
// immediately followed by `residual_add` three times a layer, and the per-layer
// embedding path adds a learned scalar after its own add. Each of those is a
// separate dispatch, and a dispatch that is barrier-separated costs ~5.8 us on
// this machine against a step of 8.46 ms -- measured, by running the same DAG
// with 723, 833 and 1 barriers.
//
// The pair fuses for free: the threadgroup that computes the row's inverse RMS
// already holds every element of the row, so adding the residual in its
// write-back costs one extra load and no synchronisation at all.
//
//   out = rms_norm(x) * w + r            (`rms_residual`)
//   out = (rms_norm(x) * w + r) * s[0]   (`rms_residual_scaled`)
//
// Arithmetic is otherwise identical to `rms_single_row` followed by
// `residual_add`: the same float accumulate, the same `precise::rsqrt`, and the
// same single bf16 round on the way out. It is NOT bit-identical to the
// two-dispatch form, which rounds the norm to bf16 before the add reads it back;
// this keeps that intermediate in float, so it is strictly closer to the
// reference. The parity walk is what says so.
template <typename T, int N_READS, bool SCALED>
METAL_FUNC void rms_residual_impl(
    const device T* x,
    const device T* w,
    const device T* r,
    const device T* s,
    device T* out,
    constant RmsParams& p,
    threadgroup float* local_inv_mean,
    threadgroup float* local_sums,
    uint gid,
    uint lid,
    uint simd_lane_id,
    uint simd_group_id) {
  const uint axis_size = p.axis_size;
  const uint w_stride = p.w_stride;

  const size_t row = size_t(gid) * size_t(axis_size);
  x += row + lid * N_READS;
  w += w_stride * lid * N_READS;
  const float inv = rms_inv_from_lane_sum(
      rms_lane_square_sum<T, N_READS>(x, axis_size, lid),
      axis_size, p.eps, local_inv_mean, local_sums,
      simd_lane_id, simd_group_id);

  const float scale = SCALED ? float(s[0]) : 1.0f;
  r += row + lid * N_READS;
  out += row + lid * N_READS;
  for (int i = 0; i < N_READS; i++) {
    if (lid * N_READS + N_READS <= axis_size || (lid * N_READS + i) < axis_size) {
      const float wv = p.gain * (p.plus_one ? (1.0f + float(w[w_stride * i]))
                                         : float(w[w_stride * i]));
      const float normed = wv * (float(x[i]) * inv);
      out[i] = static_cast<T>((normed + float(r[i])) * scale);
    }
  }
}

template <typename T, int N_READS>
[[kernel]] void rms_residual(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    constant RmsParams& p      [[buffer(3)]],
    const device T* r          [[buffer(4)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane_id          [[thread_index_in_simdgroup]],
    uint simd_group_id         [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  rms_residual_impl<T, N_READS, false>(x, w, r, nullptr, out, p, local_inv_mean, local_sums,
                                       gid, lid, simd_lane_id, simd_group_id);
}

template <typename T, int N_READS>
[[kernel]] void rms_residual_scaled(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    constant RmsParams& p      [[buffer(3)]],
    const device T* r          [[buffer(4)]],
    const device T* s          [[buffer(5)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane_id          [[thread_index_in_simdgroup]],
    uint simd_group_id         [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  rms_residual_impl<T, N_READS, true>(x, w, r, s, out, p, local_inv_mean, local_sums,
                                      gid, lid, simd_lane_id, simd_group_id);
}

#define instantiate_rms_residual(name, itype, nreads)                    \
  template [[host_name("rms_residual_" #name)]]                          \
  [[kernel]] void rms_residual<itype, nreads>(                           \
      const device itype*, const device itype*, device itype*,           \
      constant RmsParams&, const device itype*,                          \
      uint, uint, uint, uint);                                           \
  template [[host_name("rms_residual_scaled_" #name)]]                   \
  [[kernel]] void rms_residual_scaled<itype, nreads>(                    \
      const device itype*, const device itype*, device itype*,           \
      constant RmsParams&, const device itype*, const device itype*,     \
      uint, uint, uint, uint);

instantiate_rms_residual(bfloat16, bfloat, 4)
