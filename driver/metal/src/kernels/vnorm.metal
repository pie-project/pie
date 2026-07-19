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

struct VNormParams {
  float eps;
  uint  axis_size;  // head_dim (e.g. 256)
};

template <typename T, int N_READS>
[[kernel]] void vnorm_single_row(
    const device T* x        [[buffer(0)]],
    device T* out            [[buffer(1)]],
    constant VNormParams& p  [[buffer(2)]],
    uint gid                 [[threadgroup_position_in_grid]],
    uint lid                 [[thread_position_in_threadgroup]],
    uint simd_lane_id        [[thread_index_in_simdgroup]],
    uint simd_group_id       [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  const uint axis_size = p.axis_size;

  threadgroup float local_inv_rms[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
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
  if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_rms[0] = precise::rsqrt(acc / float(axis_size) + p.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_rms = local_inv_rms[0];

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

instantiate_vnorm(float32, float, 4)
instantiate_vnorm(float16, half, 4)
instantiate_vnorm(bfloat16, bfloat, 4)
