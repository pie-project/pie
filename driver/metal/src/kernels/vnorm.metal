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
    uint simd_group_id       [[simdgroup_index_in_threadgroup]]) {
  const uint axis_size = p.axis_size;

  threadgroup float local_inv_rms[1];
  threadgroup float local_sums[32];

  x += gid * size_t(axis_size) + lid * N_READS;
  const float inv_rms = rms_inv_from_lane_sum(
      rms_lane_square_sum<T, N_READS>(x, axis_size, lid),
      axis_size, p.eps, local_inv_rms, local_sums,
      simd_lane_id, simd_group_id);

  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = static_cast<T>(float(x[i]) * inv_rms);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = static_cast<T>(float(x[i]) * inv_rms);
      }
    }
  }
}

#define instantiate_vnorm(name, itype, nreads)                         \
  template [[host_name("vnorm_single_row_" #name)]]                    \
  [[kernel]] void vnorm_single_row<itype, nreads>(                     \
      const device itype*, device itype*, constant VNormParams&,       \
      uint, uint, uint, uint);

instantiate_vnorm(bfloat16, bfloat, 4)
