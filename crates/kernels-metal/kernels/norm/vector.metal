// Raw-Metal gemma4 weightless RMSNorm (V-norm before the KV write, decode M=1).
//
//   out[i] = x[i] / sqrt(mean(x^2) + eps)        (NO learnable weight)
//
// gemma4 normalizes V per-head over head_dim with a *weightless* RMSNorm before
// appending to the KV cache (`ops::rms_norm(V, eps)` in gemma4.cpp). Single row
// per kv-head; for E2B n_kv=1 so one [head_dim]=[256] row. Same single-row
// reduction shape as delta's rms_single_row, minus the weight multiply.
// bind::VNorm = { X=0, Out=1 }; the epsilon is buffer 2 and the axis buffer 3,
// each a scalar of its own.
//
// THE TWO SCALARS ARE SCALARS, NOT A STRUCT.
//
// This took `constant VNormParams& p [[buffer(2)]]` with `{ float eps; uint
// axis_size; }` in it -- MLX's layout, and the shape `kernels-vulkan` and
// `kernels-wgpu` then copied out of `norm/rms_params.h`. Both words were live,
// so nothing here is deleted: what changes is that `vnorm_single_row` states
// them as `eps: Const<f32>` and `axis_size: Const<i32>` rather than forwarding
// `ctx.params()` whole, and the routine binds two `setBytes` where it bound a
// staged block. Words 0 and 1 of the statement's run are the same two numbers
// they always were, reached by index instead of by struct field, and the
// buffer indices ASCEND IN THE STRUCT'S ORDER because that order is the
// statement's.
//
// The axis stays `uint` here while the mark is `Const<i32>`, exactly as
// `RmsParams.axis_size` does against `rms_strided_head_row`'s `axis`: the run
// is a `Vec<u32>` and the bits are the value, and what the mark's Rust type
// decides is what the BODY may do with the number -- and this body hands it to
// `rms_grid`, which refuses a non-positive extent.

#include <metal_stdlib>
using namespace metal;

#include "rms_reduce.h"

template <typename T, int N_READS>
[[kernel]] void vnorm_single_row(
    const device T* x        [[buffer(0)]],
    device T* out            [[buffer(1)]],
    const constant float& eps      [[buffer(2)]],
    const constant uint& axis_size [[buffer(3)]],
    uint gid                 [[threadgroup_position_in_grid]],
    uint lid                 [[thread_position_in_threadgroup]],
    uint simd_lane_id        [[thread_index_in_simdgroup]],
    uint simd_group_id       [[simdgroup_index_in_threadgroup]],
    uint tg_size             [[threads_per_threadgroup]]) {
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
      acc, axis_size, eps, local_inv_rms, local_sums,
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
      const device itype*, device itype*, const constant float&,       \
      const constant uint&, uint, uint, uint, uint, uint);

instantiate_vnorm(bfloat16, bfloat, 4)
