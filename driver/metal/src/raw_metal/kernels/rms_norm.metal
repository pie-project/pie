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

#include <metal_stdlib>
using namespace metal;

struct RmsParams {
  float eps;
  uint axis_size;   // feature dim (hidden), e.g. 1024
  uint w_stride;    // weight stride along axis (1 for contiguous)
};

template <typename T, int N_READS>
[[kernel]] void rms_single_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    constant RmsParams& p      [[buffer(3)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane_id          [[thread_index_in_simdgroup]],
    uint simd_group_id         [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  const uint axis_size = p.axis_size;
  const uint w_stride = p.w_stride;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      float xi = x[i];
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = precise::rsqrt(acc / axis_size + p.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
      }
    }
  }
}

#define instantiate_rms_single_row(name, itype, n_reads)               \
  template [[host_name("rms_single_row_" #name)]] [[kernel]] void       \
  rms_single_row<itype, n_reads>(                                       \
      const device itype*, const device itype*, device itype*,          \
      constant RmsParams&, uint, uint, uint, uint);

instantiate_rms_single_row(float32, float, 4)
instantiate_rms_single_row(float16, half, 4)
instantiate_rms_single_row(bfloat16, bfloat, 4)
