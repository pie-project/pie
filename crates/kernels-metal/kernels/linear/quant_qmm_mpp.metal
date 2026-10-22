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

#include "../prim/mpp.h"

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

// The routed MXFP4 matmul on the MPP matmul, as mlx-lm's `gather_qmm_rhs_nax` runs it: each
// 64-wide slice of the expert's codes is decoded into threadgroup bfloat (an E2M1 value times
// its power-of-two block scale is exact in bfloat) and multiplied there, two slices in flight
// so one barrier separates a slice's decode from its multiply.
inline float mpp_mxfp4_scale(uchar code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

// The multiply of a routed MXFP4 tile's first `R` rows: every thread decodes its run of each
// slice into `staged`, and the matmul reads only the rows an expert holds.
template <int R, int N, int S, int BK, int LD, bool BIAS, class Decode>
inline void mxfp4_mpp_rows(const device bfloat *input, device bfloat *output, int width,
                           int columns, const device bfloat *bias, ulong column, uint row,
                           uint column_tile, threadgroup bfloat (&staged)[2][N * LD],
                           thread Decode &decode) {
  auto activations = tensor(const_cast<device bfloat *>(input) + ulong(row) * width,
                            dextents<int, 2>{width, R}, array<int, 2>{1, width});
  auto weight_view = [&](uint buffer) {
    return tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>(
        staged[buffer], dextents<int, 2>{BK, N}, array<int, 2>{1, LD});
  };
  constexpr auto descriptor = matmul2d_descriptor(
      R, N, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<S>> multiply;
  auto a = activations.template slice<BK, R>(0, 0);
  auto b = weight_view(0).template slice<BK, N>(0, 0);
  auto total =
      multiply.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  const ulong valid_mask = mpp_valid_mask(total);
  mpp_for_each(total, valid_mask, [&](ushort i) __attribute__((always_inline)) { total[i] = 0; });
  uint buffer = 0;
  auto codes = decode.load(0);
  for (int k0 = 0; k0 < width; k0 += BK, buffer ^= 1) {
    decode.store(codes, staged[buffer]);
    if (k0 + BK < width)
      codes = decode.load(k0 + BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto lhs = activations.template slice<BK, R>(k0, 0);
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
                            dextents<int, 2>{columns, R}, array<int, 2>{1, columns});
  converted.store(destination.template slice<N, R>(0, 0));
}

template <int M, int N, int S, int TILE, bool BIAS>
[[kernel]] void
mxfp4_qmm_mpp_routed(const device uchar *codes [[buffer(0)]],
                     const device uchar *exponents [[buffer(1)]],
                     const device bfloat *input [[buffer(2)]], device bfloat *output [[buffer(3)]],
                     constant int &width [[buffer(4)]], constant int &columns [[buffer(5)]],
                     const device bfloat *bias [[buffer(6)]],
                     const device int *tile_expert [[buffer(7)]],
                     const device int *perm [[buffer(8)]],
                     uint3 position [[threadgroup_position_in_grid]],
                     uint thread_id [[thread_index_in_threadgroup]]) {
  constexpr int BK = 64, LD = BK + 8, THREADS = 32 * S;
  constexpr int BYTES = N * BK / 2, PER_THREAD = BYTES / THREADS;
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
  // A run starts on a 16-byte boundary inside one 32-wide block, so it reads one exponent and
  // one aligned word quad; an E2M1 code's low three bits placed at a half's bits 9..11 read
  // as the code's value over 2^14, exactly (as mlx's `fp4_e2m1` decodes it).
  static_assert(PER_THREAD == 16, "a thread decodes one aligned quad of words");
  struct Slice {
    uint4 words;
    float scale;
  };
  struct Decoder {
    const device uchar *code_row;
    const device uchar *exponent_row;
    uint offset;
    Slice load(uint k0) const {
      return {*reinterpret_cast<const device uint4 *>(code_row + k0 / 2),
              mpp_mxfp4_scale(exponent_row[k0 / 32])};
    }
    void store(Slice slice, threadgroup bfloat *into) const {
      threadgroup bfloat4 *to = reinterpret_cast<threadgroup bfloat4 *>(into + offset);
#pragma unroll
      for (int w = 0; w < 4; ++w) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint word = slice.words[w] >> (16 * h);
          const uint low = ((word & 0x7u) << 9) | ((word & 0x8u) << 12) |
                           ((word & 0x70u) << 21) | ((word & 0x80u) << 24);
          const uint high = (((word >> 8) & 0x7u) << 9) | (((word >> 8) & 0x8u) << 12) |
                            (((word >> 8) & 0x70u) << 21) | (((word >> 8) & 0x80u) << 24);
          const float2 a = float2(as_type<half2>(low) * half(16384.0h)) * slice.scale;
          const float2 b = float2(as_type<half2>(high) * half(16384.0h)) * slice.scale;
          to[2 * w + h] = bfloat4(bfloat(a.x), bfloat(a.y), bfloat(b.x), bfloat(b.y));
        }
      }
    }
  } decode{code_row, exponent_row + byte / 16, n * LD + byte * 2};
  // An expert's rows fill its tiles from the front, so a tile whose later rows no route
  // claimed multiplies only its leading 16 or 32 rows.
  if constexpr (M == 64) {
    if (perm[row + 16] < 0) {
      mxfp4_mpp_rows<16, N, S, BK, LD, BIAS>(input, output, width, columns, bias, column, row,
                                             column_tile, staged, decode);
      return;
    }
    if (perm[row + 32] < 0) {
      mxfp4_mpp_rows<32, N, S, BK, LD, BIAS>(input, output, width, columns, bias, column, row,
                                             column_tile, staged, decode);
      return;
    }
  }
  mxfp4_mpp_rows<M, N, S, BK, LD, BIAS>(input, output, width, columns, bias, column, row,
                                        column_tile, staged, decode);
}

#define PIE_MPP_MXFP4_POINT(name, m, n, s, tile, bias)                                             \
  template [[host_name(name)]] [[kernel]] void mxfp4_qmm_mpp_routed<m, n, s, tile, bias>(          \
      const device uchar *, const device uchar *, const device bfloat *, device bfloat *,          \
      constant int &, constant int &, const device bfloat *, const device int *, const device int *, \
      uint3, uint);
