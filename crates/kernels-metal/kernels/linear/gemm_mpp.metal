#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#include "../prim/mpp.h"

// `act . w^T` for bf16 activations and a bf16 weight of `N` rows of `K`, on the Metal 4
// tensor-op matmul: qmm_mpp's core with the weight read as it lies in place of the dequantize.
// A threadgroup lands an `M` x `N` tile, its float sum walked over the contraction 64 at a time;
// a row tail past `rows` is not written.
template <int M, int N, int S>
[[kernel]] void dense_gemm_mpp(const device bfloat *act [[buffer(0)]],
                               const device bfloat *w [[buffer(1)]],
                               device bfloat *y [[buffer(2)]], constant int &rows [[buffer(3)]],
                               constant int &columns [[buffer(4)]],
                               constant int &width [[buffer(5)]],
                               uint3 position [[threadgroup_position_in_grid]]) {
  const uint row = position.y * M, column = position.x * N;
  const int held = min(M, rows - int(row));
  auto activations = tensor(const_cast<device bfloat *>(act) + ulong(row) * width,
                            dextents<int, 2>{width, held}, array<int, 2>{1, width});
  auto weights = tensor(const_cast<device bfloat *>(w) + ulong(column) * width,
                        dextents<int, 2>{width, N}, array<int, 2>{1, width});
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<S>> multiply;
  auto a = activations.template slice<64, M>(0, 0);
  auto b = weights.template slice<64, N>(0, 0);
  auto total =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  const ulong valid_mask = mpp_valid_mask(total);
  mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) { total[i] = 0; });
  for (int k = 0; k < width; k += 64) {
    auto lhs = activations.template slice<64, M>(k, 0);
    auto rhs = weights.template slice<64, N>(k, 0);
    multiply.run(lhs, rhs, total);
  }
  if (held < M) {
    // The tensor store lands the whole tile whatever the extents say: a tail lands its rows one
    // element at a time.
    mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
      auto at = total.get_multidimensional_index(i);
      if (at[1] < held)
        y[ulong(row + at[1]) * columns + column + at[0]] = bfloat(total[i]);
    });
    return;
  }
  auto converted =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), bfloat>();
  mpp_for_each(total, valid_mask,
               [&](ushort i) __attribute__((always_inline)) { converted[i] = bfloat(total[i]); });
  auto destination = tensor(y + ulong(row) * columns + column, dextents<int, 2>{columns, M},
                            array<int, 2>{1, columns});
  converted.store(destination.template slice<N, M>(0, 0));
}

#define PIE_DENSE_MPP_POINT(name, m, n, s)                                                         \
  template [[host_name(name)]] [[kernel]] void dense_gemm_mpp<m, n, s>(                            \
      const device bfloat *, const device bfloat *, device bfloat *, constant int &,               \
      constant int &, constant int &, uint3);
