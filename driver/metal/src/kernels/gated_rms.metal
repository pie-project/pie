// Raw-Metal gated RMSNorm for Phase-0 decode (GDN layers; golden tag `gdn_core`).
//
// RMSNormGated(out, z) = weight * rmsnorm(out) * silu(z), normalized over the
// value-head dim (V_d=128), per value-head (V_h=16). Source of truth:
// ops/gated_delta.cpp::rmsnorm_gated —
//   ms     = mean(out^2, axis=-1)           # over V_d
//   outhat = out * rsqrt(ms + eps)
//   normed = (outhat * gate_norm_w) * (z * sigmoid(z))
// All math in float32 (MLX upcasts), but gate_norm_w is read at the checkpoint's
// own width: `linear_attn.norm.weight` ships bf16, and the loader stages weights
// byte-for-byte, so a float* parameter here reads element 2i+1 in place of i (and
// runs off the end of the tensor past V_d/2). gate_norm_w is the RAW weight (NO
// (1+w) — unlike the standard rms_norm). `out` (core_out) arrives as beta's
// GdnCore output (T); we read it, recompute in float, store T.
// Gates against golden `gdn_core` (gated-RMSNorm is folded into that tag).
//
// Launch: dispatchThreads grid=(V_d, V_h, 1), tg=(V_d, 1, 1) -> one threadgroup
// per value-head, V_d lanes cooperatively reduce the sum-of-squares.

#include <metal_stdlib>
using namespace metal;

#include "rms_params.h"
#include "rms_reduce.h"

template <typename T>
METAL_FUNC void gated_rms_body(
    const device T* x, const device T* z, const device T* w, device T* out,
    constant GatedRmsParams& p, size_t idx, uint lid,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint simd_lane, uint simd_group) {
  const float xi = float(x[idx]);
  const float inv = rms_inv_from_lane_sum(
      xi * xi, p.vd, p.eps, inv_rms, partials, simd_lane, simd_group);
  const float zr = float(z[idx]);
  const float y = 1.0f / (1.0f + metal::exp(-metal::fabs(zr)));
  const float sig = zr < 0.0f ? 1.0f - y : y;
  const float siluz = zr * sig;
  const float outhat = xi * inv;
  out[idx] = T((outhat * float(w[lid])) * siluz);
}

template <typename T>
[[kernel]] void gated_rms(
    const device T* x        [[buffer(0)]],   // core_out [V_h, V_d]
    const device T* z        [[buffer(1)]],   // gate     [V_h, V_d]
    const device T* w        [[buffer(2)]],   // gate_norm_w [V_d] (raw, act dtype)
    device T* out            [[buffer(3)]],   // [V_h, V_d]
    constant GatedRmsParams& p [[buffer(4)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 tpg         [[threadgroups_per_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const uint lid = lid3.x;
  const size_t idx =
      size_t(tgpos.z * tpg.y + tgpos.y) * p.vd + lid;
  gated_rms_body(
      x, z, w, out, p, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
}

template <typename T>
[[kernel]] void gated_rms_strided(
    const device T* x        [[buffer(0)]],
    const device T* z        [[buffer(1)]],
    const device T* w        [[buffer(2)]],
    device T* out            [[buffer(3)]],
    constant GatedRmsParams& p [[buffer(4)]],
    constant int& row_pitch    [[buffer(5)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 tpg         [[threadgroups_per_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const uint lid = lid3.x;
  const size_t idx =
      size_t(tgpos.z) * size_t(row_pitch) + size_t(tgpos.y) * p.vd + lid;
  (void)tpg;
  gated_rms_body(
      x, z, w, out, p, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
}

#define instantiate_gated_rms_strided(name, itype)                \
  template [[host_name("gated_rms_strided_" #name)]] [[kernel]] void \
  gated_rms_strided<itype>(                                       \
      const device itype*, const device itype*, const device itype*, \
      device itype*, constant GatedRmsParams&, constant int&, uint3, uint3, uint3, uint, uint);

instantiate_gated_rms_strided(bfloat16, bfloat)

#define instantiate_gated_rms(name, itype)                        \
  template [[host_name("gated_rms_" #name)]]                      \
  [[kernel]] void gated_rms<itype>(                               \
      const device itype*, const device itype*, const device itype*, \
      device itype*, constant GatedRmsParams&, uint3, uint3, uint3, uint, uint);

instantiate_gated_rms(bfloat16, bfloat)
