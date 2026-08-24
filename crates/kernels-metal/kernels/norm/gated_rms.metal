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
//
// THE TWO SCALARS ARE SCALARS, NOT A STRUCT.
//
// `eps` and `vd` took `constant GatedRmsParams& p [[buffer(4)]]` -- MLX's
// layout out of `norm/rms_params.h`, and the shape `kernels-vulkan` and
// `kernels-wgpu` then copied. Both words are live, so nothing is deleted here:
// `gated_rms` states them as `eps: Const<f32>` and `vd: Const<i32>` rather
// than forwarding `ctx.params()` whole, and binds two `setBytes` where it
// bound a staged block. Words 0 and 1 of the statement's run are the same two
// numbers they always were, reached by index instead of by struct field.
//
// The buffer indices ASCEND IN THE STRUCT'S ORDER, because that order is the
// statement's, and `row_pitch` keeps the END of the strided form's list: it
// was buffer 5 behind a block of two words and it is buffer 6 in front of two
// scalars, which is the same place. A stride folded in ahead of the pair would
// renumber what the two entrypoints share, and the pair is what they share.
//
// `vd` is `uint` here while the mark is `Const<i32>`, exactly as
// `RmsParams.axis_size` is against `rms_strided_head_row`'s `axis`: the run is
// a `Vec<u32>` and the bits are the value, and the mark's Rust type is about
// what the BODY may do with the number -- this one hands it to `head_width`,
// which refuses a non-positive extent.

#include <metal_stdlib>
using namespace metal;

#include "rms_reduce.h"

// ── TWO ELEMENTS, BECAUSE THE DECLARATION STATES TWO ────────────────────────
//
// `kernels::points::Norm::rmsnorm_gated` declares `x: In<Tensor<f32>>` and
// `weight: Const<Tensor<f32>>` beside a `gate` and a result that ride the
// statement's `T`: the gated-delta core leaves its output in float, and the
// norm weight this fold reads is the one the recurrence staged, not the
// checkpoint's bf16 row. This file used to be one template over ONE `T` for
// all four buffers, so the point could not be claimed -- a float plane read
// through a `bfloat*` reads element 2i+1 in place of i and runs off the end
// of the tensor past V_d/2, which the header above already said in as many
// words.
//
// `X` is the element of the two planes the declaration pins to float; `T` is
// the element of the two that ride the statement. `<bfloat, bfloat>` keeps
// the name and the ABI the legacy driver has always fired; `<float, bfloat>`
// is the arm `Norm::rmsnorm_gated` claims.
template <typename X, typename T, bool SILU>
METAL_FUNC void gated_rms_body(
    const device X* x, const device T* z, const device X* w, device T* out,
    float eps, uint vd, size_t idx, uint lid,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint simd_lane, uint simd_group) {
  const float xi = float(x[idx]);
  const float inv = rms_inv_from_lane_sum(
      xi * xi, vd, eps, inv_rms, partials, simd_lane, simd_group);
  const float zr = float(z[idx]);
  const float y = 1.0f / (1.0f + metal::exp(-metal::fabs(zr)));
  const float sig = zr < 0.0f ? 1.0f - y : y;
  const float gate = SILU ? zr * sig : sig;
  const float outhat = xi * inv;
  out[idx] = T((outhat * float(w[lid])) * gate);
}

template <typename X, typename T, bool SILU>
[[kernel]] void gated_rms(
    const device X* x        [[buffer(0)]],   // core_out [V_h, V_d]
    const device T* z        [[buffer(1)]],   // gate     [V_h, V_d]
    const device X* w        [[buffer(2)]],   // gate_norm_w [V_d] (raw)
    device T* out            [[buffer(3)]],   // [V_h, V_d]
    const constant float& eps  [[buffer(4)]],
    const constant uint& vd    [[buffer(5)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 tpg         [[threadgroups_per_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const uint lid = lid3.x;
  const size_t idx =
      size_t(tgpos.z * tpg.y + tgpos.y) * vd + lid;
  gated_rms_body<X, T, SILU>(
      x, z, w, out, eps, vd, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
}

template <typename X, typename T>
[[kernel]] void gated_rms_strided(
    const device X* x        [[buffer(0)]],
    const device T* z        [[buffer(1)]],
    const device X* w        [[buffer(2)]],
    device T* out            [[buffer(3)]],
    const constant float& eps  [[buffer(4)]],
    const constant uint& vd    [[buffer(5)]],
    const constant int& row_pitch [[buffer(6)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 tpg         [[threadgroups_per_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const uint lid = lid3.x;
  const size_t idx =
      size_t(tgpos.z) * size_t(row_pitch) + size_t(tgpos.y) * vd + lid;
  (void)tpg;
  gated_rms_body<X, T, true>(
      x, z, w, out, eps, vd, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
}

#define instantiate_gated_rms_strided(name, xtype, itype)         \
  template [[host_name("gated_rms_strided_" #name)]] [[kernel]] void \
  gated_rms_strided<xtype, itype>(                                \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      const constant int&, uint3, uint3, uint3, uint, uint);

instantiate_gated_rms_strided(bfloat16, bfloat, bfloat)

#define instantiate_gated_rms(name, xtype, itype)                 \
  template [[host_name("gated_rms_" #name)]]                      \
  [[kernel]] void gated_rms<xtype, itype, true>(                  \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

#define instantiate_gated_rms_by(name, xtype, itype)              \
  template [[host_name("gated_rms_by_" #name)]]                   \
  [[kernel]] void gated_rms<xtype, itype, false>(                 \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

instantiate_gated_rms(bfloat16, bfloat, bfloat)

// The claimed arm: the normed plane and its weight in float, the gate and the
// result at the statement's element. `Norm::rmsnorm_gated` fires this one.
instantiate_gated_rms(f32_bfloat16, float, bfloat)

instantiate_gated_rms_by(f32_bfloat16, float, bfloat)
