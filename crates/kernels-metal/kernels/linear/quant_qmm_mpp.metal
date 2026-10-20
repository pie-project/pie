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

// The 64-wide group sums of `M` activation rows over groups [first, stop), into
// `sums[(group - first) * M + row]`, by every thread of the threadgroup: eight lanes take a
// group of a row, eight inputs each, and fold it with three shuffles.
template <int M, int THREADS, class Index>
__attribute__((always_inline)) void
mpp_group_sums(const device bfloat *input, int width, uint first, uint stop,
               threadgroup float *sums, uint thread_id) {
  static_assert(THREADS % 32 == 0, "whole simdgroups fold the sums");
  const uint items = (stop - first) * M * 8;
  for (uint item = thread_id; item < items; item += THREADS) {
    const uint pair = item / 8, lane = item % 8;
    const uint row = pair % M, group = first + pair / M;
    const device bfloat4 *at = reinterpret_cast<const device bfloat4 *>(
        input + Index(row) * width + group * 64 + lane * 8);
    const float4 a = float4(at[0]), b = float4(at[1]);
    float sum = (a.x + a.y) + (a.z + a.w) + (b.x + b.y) + (b.z + b.w);
    sum += simd_shuffle_xor(sum, ushort(1));
    sum += simd_shuffle_xor(sum, ushort(2));
    sum += simd_shuffle_xor(sum, ushort(4));
    if (lane == 0)
      sums[pair] = sum;
  }
}

// The groups whose sums a threadgroup holds at once: 8 KB of them.
template <int M> constexpr uint mpp_sum_chunk() { return (2048 / M) & ~1u; }

// `SUMS` folds the activations' group sums here, where the zero points need them, rather
// than reading them from a pass of their own; `BIAS` adds the linear's bias to the
// landed row.
template <int M, int N, int S, bool PACKED, bool RELAXED, int PARTS, bool PAIRED, bool LOCAL,
          int ROW_GROUP, class Index, bool SUMS, bool BIAS>
[[kernel]] void
affine_qmm_mpp(const device uchar *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
               const device bfloat *biases [[buffer(2)]], const device bfloat *input [[buffer(3)]],
               device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
               constant int &columns [[buffer(6)]], const device float *sums [[buffer(7)]],
               constant int &rows [[buffer(8)]], constant int &stride [[buffer(9)]],
               const device bfloat *bias [[buffer(10)]],
               uint3 position [[threadgroup_position_in_grid]],
               uint thread_id [[thread_index_in_threadgroup]]) {
  static_assert(!SUMS || !LOCAL, "the local split's parts fold no sums of their own");
  static_assert(!BIAS || PARTS == 1 || LOCAL, "a partial lands no bias");
  constexpr uint CHUNK = mpp_sum_chunk<M>();
  threadgroup float held[SUMS ? CHUNK * M : 1];
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
  uint chunk = 0;
  auto accumulate = [&](uint group, const thread decltype(total) &result) {
    mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
      auto at = total.get_multidimensional_index(i);
      Index parameter = PACKED ? (Index(column / 256) * groups_total + group) * 256 + column % 256 + at[0]
                               : Index(column + at[0]) * groups_total + group;
      float sum;
      if constexpr (SUMS)
        sum = held[(group - chunk) * M + at[1]];
      else
        sum = sums[Index(row_tile) * M * groups + group * M + at[1]];
      total[i] += result[i] * float(scales[parameter]) + sum * float(biases[parameter]);
    });
  };
  uint group = part * (groups / PARTS);
  const uint end = (part + 1) * (groups / PARTS);
  for (chunk = group; chunk < end; chunk += SUMS ? CHUNK : end) {
    const uint stop = SUMS ? min(chunk + CHUNK, end) : end;
    if constexpr (SUMS) {
      if (chunk != part * (groups / PARTS))
        threadgroup_barrier(mem_flags::mem_threadgroup);
      mpp_group_sums<M, S * 32, Index>(input + Index(row) * width, width, chunk, stop, held,
                                       thread_id);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if constexpr (PAIRED) {
      for (; group + 1 < stop; group += 2) {
        decltype(total) first, second;
        dot(group, first);
        dot(group + 1, second);
        accumulate(group, first);
        accumulate(group + 1, second);
      }
    }
    for (; group < stop; ++group) {
      decltype(total) result;
      dot(group, result);
      accumulate(group, result);
    }
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
      if constexpr (BIAS)
        sum += float(bias[column + i % N]);
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
    mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
      float value = total[i];
      if constexpr (BIAS)
        value += float(bias[column + total.get_multidimensional_index(i)[0]]);
      converted[i] = bfloat(value);
    });
    auto destination = tensor(output + Index(row) * columns + column, dextents<int, 2>{columns, M},
                              array<int, 2>{1, columns});
    converted.store(destination.template slice<N, M>(0, 0));
  }
}

#define PIE_MPP_POINT(name, m, n, s, packed, relaxed, parts, paired, local, grid, index, sums,     \
                      bias)                                                                        \
  template [[host_name(name)]] [[kernel]] void                                                     \
  affine_qmm_mpp<m, n, s, packed, relaxed, parts, paired, local, grid, index, sums, bias>(         \
      const device uchar *, const device bfloat *, const device bfloat *, const device bfloat *,   \
      device bfloat *, constant int &, constant int &, const device float *, constant int &,       \
      constant int &, const device bfloat *, uint3, uint);

// The routed twin of the in-place kernel: a row tile belongs to one expert, whose bank lies
// `expert * columns` rows into the planes, and a tile no expert claimed is left unwritten.
template <int M, int N, int S, int TILE, bool SUMS>
[[kernel]] void
affine_qmm_mpp_routed(const device uchar *weights [[buffer(0)]],
                      const device bfloat *scales [[buffer(1)]],
                      const device bfloat *biases [[buffer(2)]],
                      const device bfloat *input [[buffer(3)]],
                      device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
                      constant int &columns [[buffer(6)]], const device float *sums [[buffer(7)]],
                      const device int *tile_expert [[buffer(8)]],
                      uint3 position [[threadgroup_position_in_grid]],
                      uint thread_id [[thread_index_in_threadgroup]]) {
  const uint column_tile = position.x, row_tile = position.y;
  const int expert = tile_expert[row_tile * M / TILE];
  if (expert < 0)
    return;
  constexpr uint CHUNK = mpp_sum_chunk<M>();
  threadgroup float held[SUMS ? CHUNK * M : 1];
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
  for (uint chunk = 0; chunk < uint(groups); chunk += SUMS ? CHUNK : uint(groups)) {
    const uint stop = SUMS ? min(chunk + CHUNK, uint(groups)) : uint(groups);
    if constexpr (SUMS) {
      if (chunk != 0)
        threadgroup_barrier(mem_flags::mem_threadgroup);
      mpp_group_sums<M, S * 32, ulong>(input + ulong(row) * width, width, chunk, stop, held,
                                       thread_id);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint group = chunk; group < stop; ++group) {
      decltype(total) result;
      auto lhs = activations.template slice<64, M>(group * 64, 0);
      auto rhs = weight_view(group).template slice<64, N>(0, 0);
      multiply.run(lhs, rhs, result);
      mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
        auto at = total.get_multidimensional_index(i);
        const ulong parameter = (column + at[0]) * groups + group;
        float sum;
        if constexpr (SUMS)
          sum = held[(group - chunk) * M + at[1]];
        else
          sum = sums[ulong(row_tile) * M * groups + group * M + at[1]];
        total[i] += result[i] * float(scales[parameter]) + sum * float(biases[parameter]);
      });
    }
  }
  auto converted =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), bfloat>();
  mpp_for_each(total, valid_mask,
               [&](ushort i) __attribute__((always_inline)) { converted[i] = bfloat(total[i]); });
  auto destination = tensor(output + ulong(row) * columns + column_tile * N,
                            dextents<int, 2>{columns, M}, array<int, 2>{1, columns});
  converted.store(destination.template slice<N, M>(0, 0));
}

#define PIE_MPP_ROUTED_POINT(name, m, n, s, tile, sums)                                            \
  template [[host_name(name)]] [[kernel]] void affine_qmm_mpp_routed<m, n, s, tile, sums>(         \
      const device uchar *, const device bfloat *, const device bfloat *, const device bfloat *,   \
      device bfloat *, constant int &, constant int &, const device float *, const device int *,   \
      uint3, uint);

// The routed MXFP4 matmul on the MPP matmul, as mlx-lm's `gather_qmm_rhs_nax` runs it: each
// 64-wide slice of the expert's codes is decoded into threadgroup bfloat (an E2M1 value times
// its power-of-two block scale is exact in bfloat) and multiplied there, two slices in flight
// so one barrier separates a slice's decode from its multiply.
constant float kMppMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float mpp_mxfp4_scale(uchar code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

template <int M, int N, int S, int TILE, bool BIAS>
[[kernel]] void
mxfp4_qmm_mpp_routed(const device uchar *codes [[buffer(0)]],
                     const device uchar *exponents [[buffer(1)]],
                     const device bfloat *input [[buffer(2)]], device bfloat *output [[buffer(3)]],
                     constant int &width [[buffer(4)]], constant int &columns [[buffer(5)]],
                     const device bfloat *bias [[buffer(6)]],
                     const device int *tile_expert [[buffer(7)]],
                     uint3 position [[threadgroup_position_in_grid]],
                     uint thread_id [[thread_index_in_threadgroup]]) {
  constexpr int BK = 64, LD = BK + 8, THREADS = 32 * S;
  constexpr int BYTES = N * BK / 2, PER_THREAD = BYTES / THREADS;
  static_assert(PER_THREAD % 4 == 0, "a thread decodes whole words");
  const uint column_tile = position.x, row_tile = position.y;
  const int expert = tile_expert[row_tile * M / TILE];
  if (expert < 0)
    return;
  const uint row = row_tile * M;
  const ulong column = ulong(expert) * columns + column_tile * N;
  const int blocks = width / 32;
  threadgroup bfloat staged[2][N * LD];

  // This thread's run of PER_THREAD bytes: one weight row, BK/2 bytes per row.
  const uint first = thread_id * PER_THREAD;
  const uint n = first / (BK / 2), byte = first % (BK / 2);
  const device uchar *code_row = codes + (column + n) * (width / 2) + byte;
  const device uchar *exponent_row = exponents + (column + n) * blocks;
  auto decode = [&](uint k0, threadgroup bfloat *into) {
    const device uchar *from = code_row + k0 / 2;
    threadgroup bfloat *to = into + n * LD + byte * 2;
#pragma unroll
    for (int w = 0; w < PER_THREAD; w += 4) {
      const float scale = mpp_mxfp4_scale(exponent_row[k0 / 32 + (byte + w) / 16]);
      const uchar4 four = *reinterpret_cast<const device uchar4 *>(from + w);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        to[2 * (w + j)] = bfloat(scale * kMppMxfp4Lut[four[j] & 0xf]);
        to[2 * (w + j) + 1] = bfloat(scale * kMppMxfp4Lut[four[j] >> 4]);
      }
    }
  };

  auto activations = tensor(const_cast<device bfloat *>(input) + ulong(row) * width,
                            dextents<int, 2>{width, M}, array<int, 2>{1, width});
  auto weight_view = [&](uint buffer) {
    return tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>(
        staged[buffer], dextents<int, 2>{BK, N}, array<int, 2>{1, LD});
  };
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<S>> multiply;
  auto a = activations.template slice<BK, M>(0, 0);
  auto b = weight_view(0).template slice<BK, N>(0, 0);
  auto total =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  const ulong valid_mask = mpp_valid_mask(total);
  mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) { total[i] = 0; });
  uint buffer = 0;
  for (int k0 = 0; k0 < width; k0 += BK, buffer ^= 1) {
    decode(k0, staged[buffer]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto lhs = activations.template slice<BK, M>(k0, 0);
    auto rhs = weight_view(buffer).template slice<BK, N>(0, 0);
    multiply.run(lhs, rhs, total);
  }
  auto converted =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), bfloat>();
  mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) {
    float value = total[i];
    if constexpr (BIAS)
      value += float(bias[column + total.get_multidimensional_index(i)[0]]);
    converted[i] = bfloat(value);
  });
  auto destination = tensor(output + ulong(row) * columns + column_tile * N,
                            dextents<int, 2>{columns, M}, array<int, 2>{1, columns});
  converted.store(destination.template slice<N, M>(0, 0));
}

#define PIE_MPP_MXFP4_POINT(name, m, n, s, tile, bias)                                             \
  template [[host_name(name)]] [[kernel]] void mxfp4_qmm_mpp_routed<m, n, s, tile, bias>(          \
      const device uchar *, const device uchar *, const device bfloat *, device bfloat *,          \
      constant int &, constant int &, const device bfloat *, const device int *, uint3, uint);
