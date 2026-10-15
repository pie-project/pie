#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

inline void staged_dequant(uint4 codes, float scale, float bias, threadgroup half *into) {
  const uint words[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
  for (ushort i = 0; i < 4; ++i) {
    const uint w = words[i];
    const float4 low = float4(w & 15, (w >> 4) & 15, (w >> 8) & 15, (w >> 12) & 15);
    const float4 high = float4((w >> 16) & 15, (w >> 20) & 15, (w >> 24) & 15, w >> 28);
    *reinterpret_cast<threadgroup half4 *>(into + 8 * i) = half4(fma(low, scale, bias));
    *reinterpret_cast<threadgroup half4 *>(into + 8 * i + 4) = half4(fma(high, scale, bias));
  }
}

template <int M, int SPLIT>
[[kernel]] void
affine_qmm_staged(const device uint *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
                  const device bfloat *biases [[buffer(2)]], const device bfloat *input [[buffer(3)]],
                  device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
                  constant int &columns [[buffer(6)]], device float *partials [[buffer(7)]],
                  constant int &rows [[buffer(8)]], constant int &stride [[buffer(9)]],
                  uint2 position [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]],
                  uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int N = 32, K = 32;
  threadgroup half stages[2][2][K * N];
  threadgroup uint4 pairs[2][2 * N];
  threadgroup half *stage = &stages[simd][0][0];
  threadgroup uint4 *pair = &pairs[simd][0];
  const uint column = position.x * 2 * N + simd * N;
  const int blocks = width / (2 * K) / SPLIT;
  const int first = int(position.y) * blocks, last = first + blocks;
  const device uint4 *codes =
      reinterpret_cast<const device uint4 *>(weights + ulong(column + lane / 2) * (stride / 8)) +
      lane % 2;
  const ulong half_rows = ulong(N / 2) * (stride / 32);
  const device bfloat *scale = scales + ulong(column + lane) * (stride / 64);
  const device bfloat *bias = biases + ulong(column + lane) * (stride / 64);
  auto activations = tensor(const_cast<device bfloat *>(input), dextents<int, 2>{width, M},
                            array<int, 2>{1, width});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> front(stage, dextents<int, 2>{K, N},
                                                                 array<int, 2>{1, K});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> back(stage + K * N, dextents<int, 2>{K, N},
                                                                array<int, 2>{1, K});
  auto b0 = front.slice<K, N>(0, 0);
  auto b1 = back.slice<K, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(M, N, K, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> multiply;
  auto a = activations.template slice<K, M>(0, 0);
  auto total =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i)
    total[i] = 0.0f;
  uint4 low = codes[2 * first], high = codes[half_rows + 2 * first];
  for (int block = first; block < last; ++block) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    pair[lane] = low;
    pair[N + lane] = high;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (block + 1 < last) {
      low = codes[2 * block + 2];
      high = codes[half_rows + 2 * block + 2];
    }
    const float s = scale[block], b = bias[block];
    staged_dequant(pair[2 * lane], s, b, stage + lane * K);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto even = activations.template slice<K, M>(2 * block * K, 0);
    multiply.run(even, b0, total);
    staged_dequant(pair[2 * lane + 1], s, b, stage + K * N + lane * K);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto odd = activations.template slice<K, M>((2 * block + 1) * K, 0);
    multiply.run(odd, b1, total);
  }
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i) {
    if (!total.is_valid_element(i))
      continue;
    const auto at = total.get_multidimensional_index(i);
    const uint row = at[1];
    if (row >= uint(rows))
      continue;
    const ulong o = ulong(row) * columns + column + at[0];
    if constexpr (SPLIT > 1)
      partials[ulong(position.y) * rows * columns + o] = total[i];
    else
      output[o] = bfloat(total[i]);
  }
}

#define PIE_STAGED_POINT(name, m, split)                                                           \
  template [[host_name(name)]] [[kernel]] void affine_qmm_staged<m, split>(                        \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);
