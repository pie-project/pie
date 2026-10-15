#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void qmm_mpp_input_sums(const device bfloat *input [[buffer(0)]],
                               device float *sums [[buffer(1)]], constant int &width [[buffer(2)]],
                               constant uint &tile_rows [[buffer(3)]],
                               uint2 tile [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint group [[simdgroup_index_in_threadgroup]]) {
  uint row = tile.y * 4 + group;
  ulong offset = ulong(row) * width + tile.x * 64 + lane;
  float sum = simd_sum(float(input[offset]) + float(input[offset + 32]));
  if (lane == 0)
    sums[ulong(row / tile_rows) * tile_rows * (width / 64) + tile.x * tile_rows + row % tile_rows] =
        sum;
}

kernel void qmm_mpp_pack_bank(const device uint *codes [[buffer(0)]],
                              const device bfloat *scales [[buffer(1)]],
                              const device bfloat *biases [[buffer(2)]],
                              device uint *packed [[buffer(3)]], constant uint &width [[buffer(4)]],
                              constant uint &columns [[buffer(5)]],
                              uint word [[thread_position_in_grid]]) {
  ulong words = ulong(columns) * width / 8;
  if (word >= words)
    return;
  uint column = word / (width / 8);
  uint group = (word % (width / 8)) / 8;
  ulong parameter = (ulong(column / 256) * (width / 64) + group) * 256 + column % 256;
  packed[parameter * 8 + word % 8] = codes[word];
  if (word % 8 == 0) {
    device bfloat *factors = reinterpret_cast<device bfloat *>(packed + words);
    ulong source = ulong(column) * (width / 64) + group;
    factors[parameter] = scales[source];
    factors[ulong(columns) * (width / 64) + parameter] = biases[source];
  }
}

#include "../common/mpp.metal"

template <int M, int N, int S, bool PACKED, bool RELAXED, int PARTS, bool PAIRED, bool LOCAL,
          int ROW_GROUP, class Index>
[[kernel]] void
affine_qmm_mpp(const device uchar *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
               const device bfloat *biases [[buffer(2)]], const device bfloat *input [[buffer(3)]],
               device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
               constant int &columns [[buffer(6)]], const device float *sums [[buffer(7)]],
               constant int &rows [[buffer(8)]], constant int &stride [[buffer(9)]],
               uint3 position [[threadgroup_position_in_grid]],
               uint thread_id [[thread_index_in_threadgroup]]) {
  uint row_tile = position.y, column_tile = position.x;
  if constexpr (ROW_GROUP == 0) {
    row_tile = position.x / (columns / N);
    column_tile = position.x % (columns / N);
  } else if constexpr (ROW_GROUP > 0) {
    uint span = ROW_GROUP * (columns / N);
    uint first = (position.x / span) * ROW_GROUP;
    uint count = min(uint(ROW_GROUP), uint(rows / M) - first);
    row_tile = first + (position.x % span) % count;
    column_tile = (position.x % span) / count;
  }
  const int groups = width / 64, groups_total = stride / 64;
  const uint row = row_tile * M, column = column_tile * N;
  const uint part = LOCAL ? thread_id / (S * 32) : position.z;
  if constexpr (PACKED) {
    scales = reinterpret_cast<const device bfloat *>(weights + Index(columns) * stride / 2);
    biases = scales + Index(columns) * groups_total;
  }
  const Index weight_base = PACKED ? Index(column / 256) * groups_total * 8192 + (column % 256) * 32
                                   : Index(column) * (stride / 2);
  auto activations = tensor(const_cast<device bfloat *>(input) + Index(row) * width,
                            dextents<int, 2>{width, M}, array<int, 2>{1, width});
  auto weight_view = [&](uint group) {
    Index offset = weight_base + Index(group) * (PACKED ? 8192 : 32);
    return tensor<device uint4b_format, dextents<int, 2>, tensor_inline>(
        const_cast<device uchar *>(weights) + offset, dextents<int, 2>{64, N},
        array<int, 2>{1, PACKED ? 64 : stride});
  };
  constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, RELAXED);
  matmul2d<descriptor, execution_simdgroups<S>> multiply;
  auto a = activations.template slice<64, M>(0, 0);
  auto b = weight_view(0).template slice<64, N>(0, 0);
  auto total =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  const ulong valid_mask = mpp_valid_mask(total);
  mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) { total[i] = 0; });
  auto dot = [&](uint group, thread decltype(total) &result) {
    auto lhs = activations.template slice<64, M>(group * 64, 0);
    auto rhs = weight_view(group).template slice<64, N>(0, 0);
    multiply.run(lhs, rhs, result);
  };
  auto accumulate = [&](uint group, const thread decltype(total) &result) {
    mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
      auto at = total.get_multidimensional_index(i);
      Index parameter = PACKED ? (Index(column / 256) * groups_total + group) * 256 + column % 256 + at[0]
                               : Index(column + at[0]) * groups_total + group;
      Index sum = Index(row_tile) * M * groups + group * M + at[1];
      total[i] += result[i] * float(scales[parameter]) + sums[sum] * float(biases[parameter]);
    });
  };
  uint group = part * (groups / PARTS);
  const uint end = (part + 1) * (groups / PARTS);
  if constexpr (PAIRED) {
    for (; group + 1 < end; group += 2) {
      decltype(total) first, second;
      dot(group, first);
      dot(group + 1, second);
      accumulate(group, first);
      accumulate(group + 1, second);
    }
  }
  for (; group < end; ++group) {
    decltype(total) result;
    dot(group, result);
    accumulate(group, result);
  }
  if constexpr (LOCAL) {
    threadgroup float partials[PARTS * M * N];
    mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
      auto at = total.get_multidimensional_index(i);
      partials[(part * M + at[1]) * N + at[0]] = total[i];
    });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = thread_id; i < M * N; i += S * 32 * PARTS) {
      float sum = 0;
      for (uint p = 0; p < PARTS; ++p)
        sum += partials[p * M * N + i];
      if (row + i / N < uint(rows))
        output[Index(row + i / N) * columns + column + i % N] = bfloat(sum);
    }
  } else if constexpr (PARTS > 1) {
    auto destination = tensor(reinterpret_cast<device float *>(output) +
                                  Index(part) * rows * columns + Index(row) * columns + column,
                              dextents<int, 2>{columns, M}, array<int, 2>{1, columns});
    total.store(destination.template slice<N, M>(0, 0));
  } else {
    auto converted =
        multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), bfloat>();
    mpp_for_each(total, valid_mask,
                 [&](ushort i) __attribute__((always_inline)) { converted[i] = bfloat(total[i]); });
    auto destination = tensor(output + Index(row) * columns + column, dextents<int, 2>{columns, M},
                              array<int, 2>{1, columns});
    converted.store(destination.template slice<N, M>(0, 0));
  }
}

#define PIE_MPP_POINT(name, m, n, s, packed, relaxed, parts, paired, local, grid, index)           \
  template [[host_name(name)]] [[kernel]] void                                                     \
  affine_qmm_mpp<m, n, s, packed, relaxed, parts, paired, local, grid, index>(                     \
      const device uchar *, const device bfloat *, const device bfloat *, const device bfloat *,   \
      device bfloat *, constant int &, constant int &, const device float *, constant int &,       \
      constant int &, uint3, uint);

// The routed twin of the in-place kernel: a row tile belongs to one expert, whose bank lies
// `expert * columns` rows into the planes, and a tile no expert claimed is left unwritten.
template <int M, int N, int S, int TILE>
[[kernel]] void
affine_qmm_mpp_routed(const device uchar *weights [[buffer(0)]],
                      const device bfloat *scales [[buffer(1)]],
                      const device bfloat *biases [[buffer(2)]],
                      const device bfloat *input [[buffer(3)]],
                      device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
                      constant int &columns [[buffer(6)]], const device float *sums [[buffer(7)]],
                      const device int *tile_expert [[buffer(8)]],
                      uint3 position [[threadgroup_position_in_grid]]) {
  const uint column_tile = position.x, row_tile = position.y;
  const int expert = tile_expert[row_tile * M / TILE];
  if (expert < 0)
    return;
  const int groups = width / 64;
  const uint row = row_tile * M;
  const ulong column = ulong(expert) * columns + column_tile * N;
  auto activations = tensor(const_cast<device bfloat *>(input) + ulong(row) * width,
                            dextents<int, 2>{width, M}, array<int, 2>{1, width});
  auto weight_view = [&](uint group) {
    return tensor<device uint4b_format, dextents<int, 2>, tensor_inline>(
        const_cast<device uchar *>(weights) + column * (width / 2) + ulong(group) * 32,
        dextents<int, 2>{64, N}, array<int, 2>{1, width});
  };
  constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<S>> multiply;
  auto a = activations.template slice<64, M>(0, 0);
  auto b = weight_view(0).template slice<64, N>(0, 0);
  auto total =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  const ulong valid_mask = mpp_valid_mask(total);
  mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) { total[i] = 0; });
  for (uint group = 0; group < uint(groups); ++group) {
    decltype(total) result;
    auto lhs = activations.template slice<64, M>(group * 64, 0);
    auto rhs = weight_view(group).template slice<64, N>(0, 0);
    multiply.run(lhs, rhs, result);
    mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
      auto at = total.get_multidimensional_index(i);
      const ulong parameter = (column + at[0]) * groups + group;
      const ulong sum = ulong(row_tile) * M * groups + group * M + at[1];
      total[i] += result[i] * float(scales[parameter]) + sums[sum] * float(biases[parameter]);
    });
  }
  auto converted =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), bfloat>();
  mpp_for_each(total, valid_mask,
               [&](ushort i) __attribute__((always_inline)) { converted[i] = bfloat(total[i]); });
  auto destination = tensor(output + ulong(row) * columns + column_tile * N,
                            dextents<int, 2>{columns, M}, array<int, 2>{1, columns});
  converted.store(destination.template slice<N, M>(0, 0));
}

#define PIE_MPP_ROUTED_POINT(name, m, n, s, tile)                                                  \
  template [[host_name(name)]] [[kernel]] void affine_qmm_mpp_routed<m, n, s, tile>(               \
      const device uchar *, const device bfloat *, const device bfloat *, const device bfloat *,   \
      device bfloat *, constant int &, constant int &, const device float *, const device int *,   \
      uint3);

// A sorted expert tile on Splash's staged expert tile (runtime/metal/kernels/shared/moe_gguf.metal,
// moe_expert_gguf_m32 over common/gguf_staged_tile.h's decode tile): two simdgroups each stream
// their own 32 columns of the tile's expert, lane l staging column l's 32 inputs a step as half
// (q * scale + bias, rounded once) into the simdgroup's two alternating stages, and the 8-, 16- or
// 32-row matmul that holds the tile's live rows accumulates the stage over the whole contraction.
// Grid (N / 64, tiles of 32 sorted rows); live rows lead a tile (perm >= 0) and only they are stored.
constant constexpr ushort kRoutedColumns = 32, kRoutedStep = 32, kRoutedTile = 32;

// 32 affine 4-bit codes (16 bytes, low nibble first) as half, q * scale + bias rounded once.
inline void staged_dequant32(uint4 codes, float scale, float bias, threadgroup half *dst) {
  const uint words[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
  for (ushort w = 0; w < 4; ++w) {
    half4 lo, hi;
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      lo[j] = half(fma(float((words[w] >> (4 * j)) & 15u), scale, bias));
      hi[j] = half(fma(float((words[w] >> (4 * j + 16)) & 15u), scale, bias));
    }
    *((threadgroup half4 *)(dst + 8 * w)) = lo;
    *((threadgroup half4 *)(dst + 8 * w + 4)) = hi;
  }
}

template <ushort R>
inline void routed_staged_rows(const device uchar *weights, const device bfloat *scales,
                               const device bfloat *biases, const device bfloat *input,
                               device bfloat *output, int width, int columns, uint tile,
                               uint expert_column, uint origin, uint live,
                               threadgroup half *stage, uint lane) {
  auto a = tensor(const_cast<device bfloat *>(input) + ulong(tile) * kRoutedTile * width,
                  dextents<int, 2>{width, R}, array<int, 2>{1, width});
  constexpr auto descriptor =
      matmul2d_descriptor(R, kRoutedColumns, kRoutedStep, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(
      stage, dextents<int, 2>{kRoutedStep, kRoutedColumns}, array<int, 2>{1, kRoutedStep});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(
      stage + kRoutedStep * kRoutedColumns, dextents<int, 2>{kRoutedStep, kRoutedColumns},
      array<int, 2>{1, kRoutedStep});
  auto b0 = bt0.slice<kRoutedStep, kRoutedColumns>(0, 0);
  auto b1 = bt1.slice<kRoutedStep, kRoutedColumns>(0, 0);
  auto a0 = a.template slice<kRoutedStep, R>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    acc[i] = 0.0f;
  const ulong column = ulong(expert_column) + origin + lane;
  const int groups = width / 64, steps = width / kRoutedStep;
  const device uint4 *codes = reinterpret_cast<const device uint4 *>(weights + column * (width / 2));
  const device bfloat *column_scales = scales + column * groups;
  const device bfloat *column_biases = biases + column * groups;
  uint4 packed = codes[0];
  float scale = float(column_scales[0]), bias = float(column_biases[0]);
  for (int step = 0; step < steps; ++step) {
    threadgroup half *buffer = stage + (step & 1) * (kRoutedStep * kRoutedColumns);
    staged_dequant32(packed, scale, bias, buffer + lane * kRoutedStep);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < steps) {
      packed = codes[step + 1];
      scale = float(column_scales[(step + 1) / 2]);
      bias = float(column_biases[(step + 1) / 2]);
    }
    auto a_slice = a.template slice<kRoutedStep, R>(step * kRoutedStep, 0);
    if (step & 1)
      operation.run(a_slice, b1, acc);
    else
      operation.run(a_slice, b0, acc);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i))
      continue;
    auto index = acc.get_multidimensional_index(i);
    if (uint(index[1]) < live)
      output[(ulong(tile) * kRoutedTile + index[1]) * columns + origin + index[0]] = bfloat(acc[i]);
  }
}

[[kernel]] void affine_qmm_routed_staged(const device uchar *weights [[buffer(0)]],
                                         const device bfloat *scales [[buffer(1)]],
                                         const device bfloat *biases [[buffer(2)]],
                                         const device bfloat *input [[buffer(3)]],
                                         device bfloat *output [[buffer(4)]],
                                         constant int &width [[buffer(5)]],
                                         constant int &columns [[buffer(6)]],
                                         const device int *tile_expert [[buffer(7)]],
                                         const device int *perm [[buffer(8)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint simd_lane [[thread_index_in_simdgroup]],
                                         uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[2 * 2 * kRoutedStep * kRoutedColumns];
  const uint tile = group.y;
  const int expert = tile_expert[tile];
  if (expert < 0)
    return;
  const uint live = simd_sum(perm[tile * kRoutedTile + simd_lane] >= 0 ? 1u : 0u);
  const uint origin = group.x * 2 * kRoutedColumns + simd_group * kRoutedColumns;
  const uint expert_column = uint(expert) * uint(columns);
  threadgroup half *mine = stage + simd_group * (2 * kRoutedStep * kRoutedColumns);
  if (live <= 8)
    routed_staged_rows<8>(weights, scales, biases, input, output, width, columns, tile,
                          expert_column, origin, live, mine, simd_lane);
  else if (live <= 16)
    routed_staged_rows<16>(weights, scales, biases, input, output, width, columns, tile,
                           expert_column, origin, live, mine, simd_lane);
  else
    routed_staged_rows<32>(weights, scales, biases, input, output, width, columns, tile,
                           expert_column, origin, live, mine, simd_lane);
}
