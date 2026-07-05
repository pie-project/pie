// Raw-Metal gated RMSNorm for Phase-0 decode (GDN layers; golden tag `gdn_core`).
//
// RMSNormGated(out, z) = weight * rmsnorm(out) * silu(z), normalized over the
// value-head dim (V_d=128), per value-head (V_h=16). Source of truth:
// ops/gated_delta.cpp::rmsnorm_gated —
//   ms     = mean(out^2, axis=-1)           # over V_d
//   outhat = out * rsqrt(ms + eps)
//   normed = (outhat * gate_norm_w) * (z * sigmoid(z))
// All math in float32 (MLX upcasts; gate_norm_w is stored F32). gate_norm_w is the
// RAW weight (NO (1+w) — unlike the standard rms_norm). `out` (core_out) arrives
// as beta's GdnCore output (T); we read it, recompute in float, store T.
// Gates against golden `gdn_core` (gated-RMSNorm is folded into that tag).
//
// Launch: dispatchThreads grid=(V_d, V_h, 1), tg=(V_d, 1, 1) -> one threadgroup
// per value-head, V_d lanes cooperatively reduce the sum-of-squares.

#include <metal_stdlib>
using namespace metal;

struct GatedRmsParams {
  float eps;    // norm eps (1e-6)
  uint  vd;     // value-head dim (reduction axis), e.g. 128
};

template <typename T>
[[kernel]] void gated_rms(
    const device T* x        [[buffer(0)]],   // core_out [V_h, V_d]
    const device T* z        [[buffer(1)]],   // gate     [V_h, V_d]
    const device float* w    [[buffer(2)]],   // gate_norm_w [V_d] (F32, raw)
    device T* out            [[buffer(3)]],   // [V_h, V_d]
    constant GatedRmsParams& p [[buffer(4)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  const uint vd   = p.vd;
  const uint head = tgpos.y;          // value-head index
  const uint lid  = lid3.x;
  const uint idx  = head * vd + lid;  // lid in [0, vd)

  float xi  = float(x[idx]);
  float acc = simd_sum(xi * xi);

  threadgroup float partials[32];     // <=32 simdgroups (vd<=1024)
  if (simd_group == 0) {
    partials[simd_lane] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) {
    partials[simd_group] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float inv_rms[1];
  if (simd_group == 0) {
    float s = simd_sum(partials[simd_lane]);
    if (simd_lane == 0) {
      inv_rms[0] = precise::rsqrt(s / float(vd) + p.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float outhat = xi * inv_rms[0];
  float zr  = float(z[idx]);
  float y   = 1.0f / (1.0f + metal::exp(-metal::fabs(zr)));
  float sig = (zr < 0.0f) ? (1.0f - y) : y;       // MLX stable sigmoid
  float siluz = zr * sig;                          // silu(z) = z*sigmoid(z)
  out[idx] = T((outhat * w[lid]) * siluz);
}

#define instantiate_gated_rms(name, itype)                        \
  template [[host_name("gated_rms_" #name)]]                      \
  [[kernel]] void gated_rms<itype>(                               \
      const device itype*, const device itype*, const device float*, \
      device itype*, constant GatedRmsParams&, uint3, uint3, uint, uint);

instantiate_gated_rms(float32, float)
instantiate_gated_rms(float16, half)
instantiate_gated_rms(bfloat16, bfloat)
