// Raw-Metal gemma4 weightless RMSNorm (V-norm before the KV write, decode M=1).
//
//   out[i] = x[i] / sqrt(mean(x^2) + eps)        (NO learnable weight)
//
// gemma4 normalizes V per-head over head_dim with a *weightless* RMSNorm before
// appending to the KV cache (`ops::rms_norm(V, eps)` in gemma4.cpp). Single row
// per kv-head; for E2B n_kv=1 so one [head_dim]=[256] row. Same single-row
// reduction shape as delta's rms_single_row, minus the weight multiply.
// bind::VNorm = { X=0, Out=1 }; Axis/Eps are static geometry (VNormParams).

#include <metal_stdlib>
using namespace metal;

#include "rms_params.h"
#include "rms_reduce.h"

template <typename T, int N_READS>
[[kernel]] void vnorm_single_row(
    const device T* x        [[buffer(0)]],
    device T* out            [[buffer(1)]],
    constant VNormParams& p  [[buffer(2)]],
    uint gid                 [[threadgroup_position_in_grid]],
    uint lid                 [[thread_position_in_threadgroup]],
    uint simd_lane_id        [[thread_index_in_simdgroup]],
    uint simd_group_id       [[simdgroup_index_in_threadgroup]],
    uint tg_size             [[threads_per_threadgroup]]) {
  const uint axis_size = p.axis_size;
  const uint span = tg_size * uint(N_READS);

  threadgroup float local_inv_rms[1];
  threadgroup float local_sums[32];

  // Strided for the reason `rms_row_body` is: one chunk per thread makes the
  // threadgroup `ceil(axis_size / N_READS)`, which passes what Metal allows
  // once a row is wider than `N_READS * 1024`.
  const device T* xr = x + gid * size_t(axis_size);
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    acc += rms_lane_square_sum_at<T, N_READS>(xr + start, axis_size, start);
  }
  const float inv_rms = rms_inv_from_lane_sum(
      acc, axis_size, p.eps, local_inv_rms, local_sums,
      simd_lane_id, simd_group_id);

  device T* outr = out + gid * size_t(axis_size);
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        outr[start + uint(i)] = static_cast<T>(float(xr[start + uint(i)]) * inv_rms);
      }
    }
  }
}

#define instantiate_vnorm(name, itype, nreads)                         \
  template [[host_name("vnorm_single_row_" #name)]]                    \
  [[kernel]] void vnorm_single_row<itype, nreads>(                     \
      const device itype*, device itype*, constant VNormParams&,       \
      uint, uint, uint, uint, uint);

instantiate_vnorm(bfloat16, bfloat, 4)
