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
               constant int &rows [[buffer(8)]], uint3 position [[threadgroup_position_in_grid]],
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
  const int groups = width / 64;
  const uint row = row_tile * M, column = column_tile * N;
  const uint part = LOCAL ? thread_id / (S * 32) : position.z;
  if constexpr (PACKED) {
    scales = reinterpret_cast<const device bfloat *>(weights + Index(columns) * width / 2);
    biases = scales + Index(columns) * groups;
  }
  const Index weight_base = PACKED ? Index(column / 256) * groups * 8192 + (column % 256) * 32
                                   : Index(column) * (width / 2);
  auto activations = tensor(const_cast<device bfloat *>(input) + Index(row) * width,
                            dextents<int, 2>{width, M}, array<int, 2>{1, width});
  auto weight_view = [&](uint group) {
    Index offset = weight_base + Index(group) * (PACKED ? 8192 : 32);
    return tensor<device uint4b_format, dextents<int, 2>, tensor_inline>(
        const_cast<device uchar *>(weights) + offset, dextents<int, 2>{64, N},
        array<int, 2>{1, PACKED ? 64 : width});
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
      Index parameter = PACKED ? (Index(column / 256) * groups + group) * 256 + column % 256 + at[0]
                               : Index(column + at[0]) * groups + group;
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
      uint3, uint);
