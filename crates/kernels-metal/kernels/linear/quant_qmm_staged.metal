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
METAL_FUNC void staged_tile(const device uint *weights, const device bfloat *scales,
                            const device bfloat *biases, const device bfloat *input,
                            device bfloat *output, constant int &width, constant int &columns,
                            device float *partials, constant int &rows, constant int &stride,
                            uint2 position, uint lane, uint simd, threadgroup half *stages,
                            threadgroup uint4 *pairs) {
  constexpr int N = 32, K = 32;
  threadgroup half *stage = stages + simd * 2 * K * N;
  threadgroup uint4 *pair = pairs + simd * 2 * N;
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
  threadgroup half stages[2 * 2 * 32 * 32];
  threadgroup uint4 pairs[2 * 2 * 32];
  staged_tile<M, SPLIT>(weights, scales, biases, input, output, width, columns, partials, rows,
                        stride, position, lane, simd, stages, pairs);
}

#define PIE_STAGED_POINT(name, m, split)                                                           \
  template [[host_name(name)]] [[kernel]] void affine_qmm_staged<m, split>(                        \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);

// The same eight rows, the dequantized tile handed to the MPP op from registers: no
// threadgroup stage, and a lane reads 32 contiguous bytes of a column a step, four
// lanes a whole 128-byte line. Inside each 256-wide span of the contraction the order
// of k is permuted alike for both operands, so a lane's share of a column is one whole
// quant group (one scale, one bias): logical k = 4q + 16t + r of run `cc` is physical
// 64q + 8cc + 4t + r. Cooperative inputs need the 26.2 tensor ops; an older system
// compiles these points as the staged tile they replace.
#if __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
#define PIE_COOP_MULTIPLY                                                                          \
  constexpr auto descriptor = matmul2d_descriptor(16, 32, 32, false, true, false,                 \
                                                  matmul2d_descriptor::mode::multiply_accumulate); \
  matmul2d<descriptor, execution_simdgroup> multiply;                                              \
  auto a = multiply.get_left_input_cooperative_tensor<bfloat, half, float>();                      \
  auto b = multiply.get_right_input_cooperative_tensor<bfloat, half, float>()

// A simdgroup's 32 columns from `column`, zeroed: the destination `coop_walk` adds to.
inline auto coop_destination() {
  PIE_COOP_MULTIPLY;
  auto total = multiply.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i)
    total[i] = 0.0f;
  return total;
}

// Adds the products of spans [first, last) of the contraction, for the bank's 32 columns from
// `column`, to `total`.
template <class Acc>
METAL_FUNC void coop_walk(const device uint *weights, const device bfloat *scales,
                          const device bfloat *biases, const device bfloat *input, int width,
                          int rows, int stride, uint column, int first, int last, uint lane,
                          thread Acc &total) {
  PIE_COOP_MULTIPLY;
#pragma unroll
  for (ushort i = 0; i < a.get_capacity(); ++i)
    a[i] = bfloat(0.0f);
  // The operand layouts put lane `lane` on group `q` of each span and on row (of the
  // activations) or column (of the weights, plus 8j) `sub`, both as they lay it out.
  const ushort q = (lane & 1) + 2 * ((lane >> 3) & 1);
  const ushort sub = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  const ulong words = ulong(stride) / 32, groups = ulong(stride) / 64;
  const bool live = int(sub) < rows;
  const device uint4 *codes =
      reinterpret_cast<const device uint4 *>(weights) + ulong(column + sub) * words + 2 * q;
  const device bfloat *scale = scales + ulong(column + sub) * groups + q;
  const device bfloat *bias = biases + ulong(column + sub) * groups + q;
  const device uint4 *x = reinterpret_cast<const device uint4 *>(
                              input + ulong(min(int(sub), rows - 1)) * width) + 8 * q;
  for (int span = first; span < last; ++span) {
    half2 s[4], z[4];
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      s[j] = half2(half(float(scale[8 * j * groups + span * 4])));
      z[j] = half2(half(float(bias[8 * j * groups + span * 4])));
    }
#pragma unroll
    for (ushort h = 0; h < 2; ++h) {
      uint4 w[4];
#pragma unroll
      for (ushort j = 0; j < 4; ++j)
        w[j] = codes[8 * j * words + span * 8 + h];
#pragma unroll
      for (ushort cc = 0; cc < 4; ++cc) {
        const uint4 xv = x[span * 32 + h * 4 + cc];
#pragma unroll
        for (ushort t = 0; t < 2; ++t)
#pragma unroll
          for (ushort r = 0; r < 4; ++r)
            a[8 * t + r] = live ? as_type<bfloat2>(xv[2 * t + r / 2])[r % 2] : bfloat(0.0f);
        // Two codes a step: the half 1024 + code, less 1024, is the code exactly.
#pragma unroll
        for (ushort j = 0; j < 4; ++j) {
          const uint word = w[j][cc];
#pragma unroll
          for (ushort r = 0; r < 4; ++r) {
            half2 v = as_type<half2>(((word >> (4 * r)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
            v = fma(v, s[j], z[j]);
            b[4 * j + r] = v.x;
            b[16 + 4 * j + r] = v.y;
          }
        }
        multiply.run(a, b, total);
      }
    }
  }
}
#endif

// A threadgroup holds 64 columns as two simdgroups, and `SPLIT` partitions of the contraction
// as that many such pairs: simdgroup `2p + c` walks partition `p` (its share of the 256-input
// spans, as the register tile cuts them) of columns `32c` on. Each partition past the first
// hands its sums over in threadgroup memory, and the first adds them in partition order from
// zero, as the split reduction of the partials would.
template <int SPLIT>
[[kernel]] void
affine_qmm_coop(const device uint *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
                const device bfloat *biases [[buffer(2)]], const device bfloat *input [[buffer(3)]],
                device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
                constant int &columns [[buffer(6)]], constant int &rows [[buffer(8)]],
                constant int &stride [[buffer(9)]],
                uint2 position [[threadgroup_position_in_grid]],
                uint lane [[thread_index_in_simdgroup]],
                uint simd [[simdgroup_index_in_threadgroup]]) {
#if __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
  const uint part = simd / 2, column = position.x * 64 + (simd % 2) * 32;
  const int spans = width / 256;
  const int first = int(part) * spans / SPLIT, last = (int(part) + 1) * spans / SPLIT;
  auto total = coop_destination();
  coop_walk(weights, scales, biases, input, width, rows, stride, column, first, last, lane,
            total);
  // The eight live rows of a lane's elements, in element order.
  threadgroup float handed[SPLIT > 1 ? (SPLIT - 1) * 2 * 8 * 32 : 1];
  threadgroup float *mine = handed + (simd % 2) * 8 * 32 + lane;
  if constexpr (SPLIT > 1) {
    if (part > 0) {
      ushort slot = 0;
#pragma unroll
      for (ushort i = 0; i < total.get_capacity(); ++i) {
        if (!total.is_valid_element(i) || total.get_multidimensional_index(i)[1] >= 8)
          continue;
        mine[((part - 1) * 2 * 8 + slot++) * 32] = total[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part > 0)
      return;
  }
  ushort slot = 0;
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i) {
    if (!total.is_valid_element(i))
      continue;
    const auto at = total.get_multidimensional_index(i);
    const uint row = at[1];
    if (row >= 8)
      continue;
    float sum = total[i];
    if constexpr (SPLIT > 1) {
      sum = 0.0f + sum;
      for (int p = 1; p < SPLIT; ++p)
        sum += mine[((p - 1) * 2 * 8 + slot) * 32];
    }
    ++slot;
    if (row < uint(rows))
      output[ulong(row) * columns + column + at[0]] = bfloat(sum);
  }
#else
  // The first pair walks the whole contraction alone.
  threadgroup half stages[2 * 2 * 32 * 32];
  threadgroup uint4 pairs[2 * 2 * 32];
  if (simd >= 2)
    return;
  staged_tile<8, 1>(weights, scales, biases, input, output, width, columns, nullptr, rows,
                    stride, position, lane, simd, stages, pairs);
#endif
}

// A packed `[gate | up]` bank's gate and up columns into their swiglu, `output` rows of
// `intermediate`: a threadgroup's first simdgroup walks the gate's 32 columns from 32x, its
// second the up's same columns, which it hands over rounded to bf16, so each output is
// `packed_swiglu` of the gate and up the plain tile lands, bit for bit.
[[kernel]] void
affine_qmm_coop_swiglu(const device uint *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
                       const device bfloat *biases [[buffer(2)]], const device bfloat *input [[buffer(3)]],
                       device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
                       constant int &intermediate [[buffer(6)]], constant int &rows [[buffer(8)]],
                       constant int &stride [[buffer(9)]],
                       uint2 position [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
#if __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
  threadgroup bfloat up[16 * 32];
  const uint column = position.x * 32;
  auto total = coop_destination();
  coop_walk(weights, scales, biases, input, width, rows, stride, column + simd * intermediate, 0,
            width / 256, lane, total);
  if (simd == 1) {
#pragma unroll
    for (ushort i = 0; i < total.get_capacity(); ++i)
      up[i * 32 + lane] = bfloat(total[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 1)
    return;
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i) {
    if (!total.is_valid_element(i))
      continue;
    const auto at = total.get_multidimensional_index(i);
    const uint row = at[1];
    if (row >= uint(rows) || row >= 8)
      continue;
    const float g = float(bfloat(total[i]));
    const float u = float(up[i * 32 + lane]);
    output[ulong(row) * intermediate + column + at[0]] = bfloat((g / (1.0f + metal::exp(-g))) * u);
  }
#else
  // Without the tensor ops a thread sums its outputs' dot products alone: correct, not quick.
  const uint column = position.x * 32;
  const ulong groups = ulong(stride) / 64;
  for (uint at = simd * 32 + lane; at < 8 * 32; at += 64) {
    const uint row = at / 32, c = column + at % 32;
    if (row >= uint(rows))
      continue;
    float sums[2];
    for (uint half_ = 0; half_ < 2; ++half_) {
      const uint n = c + half_ * intermediate;
      const device uint *codes = weights + ulong(n) * (stride / 8);
      float sum = 0.0f;
      for (int k = 0; k < width; ++k) {
        const float code = float((codes[k / 8] >> (4 * (k % 8))) & 15);
        const float w = float(half(fma(half(code), half(float(scales[n * groups + k / 64])),
                                       half(float(biases[n * groups + k / 64])))));
        sum += float(input[ulong(row) * width + k]) * w;
      }
      sums[half_] = sum;
    }
    const float g = float(bfloat(sums[0]));
    const float u = float(bfloat(sums[1]));
    output[ulong(row) * intermediate + c] = bfloat((g / (1.0f + metal::exp(-g))) * u);
  }
#endif
}

#define PIE_COOP_POINT(name, split)                                                                \
  template [[host_name(name)]] [[kernel]] void affine_qmm_coop<split>(                             \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, constant int &, constant int &, uint2,      \
      uint, uint);
