// Raw-Metal dense bf16 GEMV — gemma4's M=1 decode linear kernel.
//
// out[N] = W[N, K] @ x[K]   (row-major W, contiguous x/y)
//
// gemma4-E2B ships dense bf16 (no 4-bit checkpoint exists), so the linear layers
// (q/k/v/o/gate/up/down + tied embed^T logits) are a plain matvec — simpler than
// qwen3.6's affine_qmv (no dequant). Mirrors qmv_fast's launch geometry so the
// two share a dispatch shape:
//   * group = (SIMD_SIZE=32, num_simdgroups=2, 1)
//   * grid  = (1, ceil(N / (num_simdgroups*results_per_simdgroup)) , 1)
//   * each threadgroup owns 2*4 = 8 output rows; each simd lane strides K by 32,
//     accumulates in float, reduces with simd_sum -> bit-exact-friendly vs an
//     mlx float-accumulated reference (charlie's cosine gate >= 0.99999).
//   * K (in_vec_size) / N (out_vec_size) are STATIC per weight (geometry, not IO
//     scalars) -> safe as constant buffers under the decode I1 invariant.
//   * bfloat is native on Metal 4 (no bf16.h vendoring).

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

static constant constexpr const int SIMD_SIZE = 32;

template <typename T>
[[kernel]] void bf16_gemv(
    const device T* w               [[buffer(0)]],  // [N, K] row-major
    const device T* x               [[buffer(1)]],  // [K]
    device T* y                     [[buffer(2)]],  // [N]
    const constant int& in_vec_size [[buffer(3)]],  // K
    const constant int& out_vec_size[[buffer(4)]],  // N
    uint3 tid     [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;

  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  // Each simd lane reads x[k] for k = simd_lid, simd_lid+32, ...
  float result[results_per_simdgroup] = {0.0f};
  for (int k = simd_lid; k < in_vec_size; k += SIMD_SIZE) {
    const float xk = static_cast<float>(x[k]);
    for (int row = 0; row < results_per_simdgroup; row++) {
      const int orow = out_row + row;
      if (orow < out_vec_size) {
        result[row] += static_cast<float>(w[orow * in_vec_size + k]) * xk;
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    const float s = simd_sum(result[row]);
    const int orow = out_row + row;
    if (simd_lid == 0 && orow < out_vec_size) {
      y[orow] = static_cast<T>(s);
    }
  }
}

#define instantiate_bf16_gemv(name, itype)                              \
  template [[host_name("bf16_gemv_" #name)]]                            \
  [[kernel]] void bf16_gemv<itype>(                                     \
      const device itype*, const device itype*, device itype*,          \
      const constant int&, const constant int&, uint3, uint, uint);

instantiate_bf16_gemv(float32, float)
instantiate_bf16_gemv(float16, half)
instantiate_bf16_gemv(bfloat16, bfloat)
