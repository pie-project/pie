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

#if __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
// The register-fed tile's walk over spans [first, last) of one simdgroup's 32 columns.
template <class Multiply, class A, class B, class Total>
METAL_FUNC void coop_walk(thread Multiply &multiply, thread A &a, thread B &b, thread Total &total,
                          const device uint4 *codes, const device bfloat *scale,
                          const device bfloat *bias, const device uint4 *x, ulong words,
                          ulong groups, int first, int last, bool live, uint tile) {
  // Each tile starts its walk (tile * 40503 mod 2^16) / 2^16 of the way in and wraps, so the
  // tiles of a launch do not all fetch the same slice of the input at once.
  const int count = last - first;
  const int lead = int((ulong((tile * 40503u) & 0xFFFFu) * uint(count)) >> 16);
  for (int step = 0; step < count; ++step) {
    const int span = first + (lead + step) % count;
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

// The same eight rows, the dequantized tile handed to the MPP op from registers: no
// threadgroup stage, and a lane reads 32 contiguous bytes of a column a step, four
// lanes a whole 128-byte line. Inside each 256-wide span of the contraction the order
// of k is permuted alike for both operands, so a lane's share of a column is one whole
// quant group (one scale, one bias): logical k = 4q + 16t + r of run `cc` is physical
// 64q + 8cc + 4t + r. Cooperative inputs need the 26.2 tensor ops; an older system
// compiles this point as the staged tile it replaces. `SPLIT` is the grid's y extent.
template <int SPLIT, int INNER = 1, bool GLU = false>
[[kernel]] void
affine_qmm_coop(const device uint *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
                const device bfloat *biases [[buffer(2)]], const device bfloat *input [[buffer(3)]],
                device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
                constant int &columns [[buffer(6)]], device float *partials [[buffer(7)]],
                constant int &rows [[buffer(8)]], constant int &stride [[buffer(9)]],
                uint2 position [[threadgroup_position_in_grid]],
                uint lane [[thread_index_in_simdgroup]],
                uint simd [[simdgroup_index_in_threadgroup]]) {
#if __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
  constexpr int N = 32;
  constexpr auto descriptor = matmul2d_descriptor(16, N, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> multiply;
  threadgroup float parts[INNER > 1 ? (INNER - 1) * 8 * 2 * N : GLU ? 8 * N : 1];
  auto a = multiply.get_left_input_cooperative_tensor<bfloat, half, float>();
  auto b = multiply.get_right_input_cooperative_tensor<bfloat, half, float>();
  auto total = multiply.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i)
    total[i] = 0.0f;
#pragma unroll
  for (ushort i = 0; i < a.get_capacity(); ++i)
    a[i] = bfloat(0.0f);
  // The operand layouts put lane `lane` on group `q` of each span and on row (of the
  // activations) or column (of the weights, plus 8j) `sub`, both as they lay it out.
  const ushort q = (lane & 1) + 2 * ((lane >> 3) & 1);
  const ushort sub = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  // INNER > 1 cuts the contraction across the threadgroup's simdgroups instead of the grid:
  // simdgroup `simd` takes half simd % 2 of the 64 columns and part simd / 2 of k.
  // GLU: simdgroup 0 takes the gate's 32 columns of `columns` and 1 the up's, packed past them.
  const uint column = GLU ? simd * columns + position.x * N
                          : position.x * 2 * N + (INNER > 1 ? simd % 2 : simd) * N;
  const int spans = width / 256 / SPLIT / INNER;
  const int first = (INNER > 1 ? int(simd / 2) : int(position.y)) * spans, last = first + spans;
  const ulong words = ulong(stride) / 32, groups = ulong(stride) / 64;
  const bool live = int(sub) < rows;
  const device uint4 *codes =
      reinterpret_cast<const device uint4 *>(weights) + ulong(column + sub) * words + 2 * q;
  const device bfloat *scale = scales + ulong(column + sub) * groups + q;
  const device bfloat *bias = biases + ulong(column + sub) * groups + q;
  const device uint4 *x = reinterpret_cast<const device uint4 *>(
                              input + ulong(min(int(sub), rows - 1)) * width) + 8 * q;
  coop_walk(multiply, a, b, total, codes, scale, bias, x, words, groups, first, last, live,
            position.x);
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i) {
    if (!total.is_valid_element(i))
      continue;
    const auto at = total.get_multidimensional_index(i);
    const uint row = at[1];
    if (row >= uint(rows) || row >= 8)
      continue;
    const ulong o = ulong(row) * columns + column + at[0];
    if constexpr (GLU) {
      if (simd == 1)
        parts[row * N + at[0]] = total[i];
    } else if constexpr (INNER > 1) {
      if (simd >= 2)
        parts[(simd / 2 - 1) * 8 * 2 * N + row * 2 * N + (simd % 2) * N + at[0]] = total[i];
    } else if constexpr (SPLIT > 1)
      partials[ulong(position.y) * rows * columns + o] = total[i];
    else
      output[o] = bfloat(total[i]);
  }
  if constexpr (GLU) {
    // swiglu of the rounded gate and up, as packed_swiglu takes them.
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
      const float u = float(bfloat(parts[row * N + at[0]]));
      output[ulong(row) * columns + column + at[0]] = bfloat((g / (1.0f + metal::exp(-g))) * u);
    }
  }
  if constexpr (INNER > 1) {
    // The parts summed in k order from zero, as qmm_splitk_reduce sums a grid's.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd >= 2)
      return;
#pragma unroll
    for (ushort i = 0; i < total.get_capacity(); ++i) {
      if (!total.is_valid_element(i))
        continue;
      const auto at = total.get_multidimensional_index(i);
      const uint row = at[1];
      if (row >= uint(rows) || row >= 8)
        continue;
      float acc = 0.0f;
      acc += total[i];
      for (int part = 0; part < INNER - 1; ++part)
        acc += parts[part * 8 * 2 * N + row * 2 * N + simd * N + at[0]];
      output[ulong(row) * columns + column + at[0]] = bfloat(acc);
    }
  }
#else
  threadgroup half stages[2 * 2 * 32 * 32];
  threadgroup uint4 pairs[2 * 2 * 32];
  if constexpr (GLU) {
    // Lane by lane: each thread one gate and up dot of a row.
    const uint t = simd * 32 + lane;
    const ulong words = ulong(stride) / 8, groups = ulong(stride) / 64;
    for (uint at = t; at < 8 * 32; at += 64) {
      const uint row = at / 32, col = position.x * 32 + at % 32;
      if (row >= uint(rows))
        continue;
      float g = 0.0f, u = 0.0f;
      for (int i = 0; i < width; ++i) {
        const float xv = float(input[ulong(row) * width + i]);
        const ulong gr = col, ur = ulong(columns) + col;
        const float gq = float((weights[gr * words + i / 8] >> (4 * (i % 8))) & 15);
        const float uq = float((weights[ur * words + i / 8] >> (4 * (i % 8))) & 15);
        g += xv * (gq * float(scales[gr * groups + i / 64]) + float(biases[gr * groups + i / 64]));
        u += xv * (uq * float(scales[ur * groups + i / 64]) + float(biases[ur * groups + i / 64]));
      }
      g = float(bfloat(g));
      u = float(bfloat(u));
      output[ulong(row) * columns + col] = bfloat((g / (1.0f + metal::exp(-g))) * u);
    }
    return;
  }
  // An inner cut runs whole on the first two simdgroups here.
  if (INNER > 1 && simd >= 2)
    return;
  staged_tile<8, SPLIT>(weights, scales, biases, input, output, width, columns, partials, rows,
                        stride, position, lane, simd, stages, pairs);
#endif
}

#define PIE_COOP_POINT(name, split)                                                                \
  template [[host_name(name)]] [[kernel]] void affine_qmm_coop<split>(                             \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);

#define PIE_COOP_GLU_POINT(name)                                                                   \
  template [[host_name(name)]] [[kernel]] void affine_qmm_coop<1, 1, true>(                        \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);

#define PIE_COOP_INNER_POINT(name, inner)                                                          \
  template [[host_name(name)]] [[kernel]] void affine_qmm_coop<1, inner>(                          \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);

// Two or three projections of one activation in one launch of the register-fed tile, each
// segment's columns from row `first` of its bank landed in its own output: simdgroup tile `g` (32
// columns) of the segments' tiles in order, every one over the whole contraction.
[[kernel]] void
affine_qmm_coop_segments(const device uint *w0 [[buffer(0)]], const device bfloat *s0 [[buffer(1)]],
                         const device bfloat *b0 [[buffer(2)]], device bfloat *y0 [[buffer(3)]],
                         const device uint *w1 [[buffer(4)]], const device bfloat *s1 [[buffer(5)]],
                         const device bfloat *b1 [[buffer(6)]], device bfloat *y1 [[buffer(7)]],
                         const device uint *w2 [[buffer(8)]], const device bfloat *s2 [[buffer(9)]],
                         const device bfloat *b2 [[buffer(10)]], device bfloat *y2 [[buffer(11)]],
                         const device bfloat *input [[buffer(12)]], constant int &width [[buffer(13)]],
                         constant int &rows [[buffer(14)]], constant int &c0 [[buffer(15)]],
                         constant int &c1 [[buffer(16)]], constant int &c2 [[buffer(17)]],
                         constant int &ld0 [[buffer(18)]], constant int &ld1 [[buffer(19)]],
                         constant int &ld2 [[buffer(20)]], constant int &f0 [[buffer(21)]],
                         constant int &f1 [[buffer(22)]], constant int &f2 [[buffer(23)]],
                         uint2 position [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int N = 32;
  const uint g = position.x * 2 + simd, t0 = c0 / N, t1 = t0 + c1 / N, t2 = t1 + c2 / N;
  if (g >= t2)
    return;
  const int seg = g < t0 ? 0 : g < t1 ? 1 : 2;
  const uint local = g - (seg == 0 ? 0 : seg == 1 ? t0 : t1);
  const device uint *weights = seg == 0 ? w0 : seg == 1 ? w1 : w2;
  const device bfloat *scales = seg == 0 ? s0 : seg == 1 ? s1 : s2;
  const device bfloat *biases = seg == 0 ? b0 : seg == 1 ? b1 : b2;
  device bfloat *output = seg == 0 ? y0 : seg == 1 ? y1 : y2;
  const int columns = seg == 0 ? c0 : seg == 1 ? c1 : c2;
  const int stride = seg == 0 ? ld0 : seg == 1 ? ld1 : ld2;
  const uint column = uint(seg == 0 ? f0 : seg == 1 ? f1 : f2) + local * N;
#if __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
  constexpr auto descriptor = matmul2d_descriptor(16, N, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> multiply;
  auto a = multiply.get_left_input_cooperative_tensor<bfloat, half, float>();
  auto b = multiply.get_right_input_cooperative_tensor<bfloat, half, float>();
  auto total = multiply.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i)
    total[i] = 0.0f;
#pragma unroll
  for (ushort i = 0; i < a.get_capacity(); ++i)
    a[i] = bfloat(0.0f);
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
  coop_walk(multiply, a, b, total, codes, scale, bias, x, words, groups, 0, width / 256, live,
            g / 2);
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i) {
    if (!total.is_valid_element(i))
      continue;
    const auto at = total.get_multidimensional_index(i);
    const uint row = at[1];
    if (row >= uint(rows) || row >= 8)
      continue;
    output[ulong(row) * columns + local * N + at[0]] = bfloat(total[i]);
  }
#else
  // Lane by lane: each lane one column's dots.
  const ulong words = ulong(stride) / 8, groups = ulong(stride) / 64;
  const ulong r = column + lane;
  for (int row = 0; row < rows; ++row) {
    float acc = 0.0f;
    for (int i = 0; i < width; ++i) {
      const float q = float((weights[r * words + i / 8] >> (4 * (i % 8))) & 15);
      acc += float(input[ulong(row) * width + i]) *
             (q * float(scales[r * groups + i / 64]) + float(biases[r * groups + i / 64]));
    }
    output[ulong(row) * columns + local * N + lane] = bfloat(acc);
  }
#endif
}
