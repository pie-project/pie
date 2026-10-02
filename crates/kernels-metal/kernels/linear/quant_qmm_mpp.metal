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

// --- PQ2_0 positional-2-bit -> affine MPP packed front-end -----------------
//
// PQ2_0 reconstructs a weight as `w = (code - 1) * d`, code in {0,1,2,3}, with
// one fp16 scale `d` per 128-weight block. That is algebraically the affine MPP
// accumulate `total = sum(code*x)*scale + sum(x)*bias` with code=code (0..3,
// still fits uint4b), scale=d, bias=-d — identical in form to PTQ1_0 (code just
// ranges 0..3 instead of 0..2). So we DECODE each 34-byte PQ2_0 block into the
// exact affine `qmm_mpp_pack_bank` layout (uint4b codes + per-64-group bf16
// scale/bias) and the existing `affine_qmm_mpp` PACKED kernel runs unchanged.
//
// PQ2_0 is SIMPLER than PTQ1_0: no base-3 {16,8}+qh staging. Block = 34 bytes:
// the fp16 `d` LEADS at bytes 0-1 (not a trailing one), then `qs[32]` at byte
// offset 2; element `e` lives in byte `2 + e/4` at bit offset `(e%4)*2`,
// `code = (qs[2+e/4] >> ((e%4)*2)) & 3`, natural element order 0..127.
// (Authoritative: `quant_pq2_0.metal` `pq2_0_block_dot`.) Everything else —
// grid, packed layout, 64-group->block mapping, nibble/lane order, factors — is
// identical to `ptq1_0_qmm_mpp_pack_bank` below.

#define PQ2_0_BLOCK_BYTES 34
#define PQ2_0_QS_OFFSET 2

// The positional 2-bit code (0..3) for natural block element `e` of a 34-byte
// block, mirroring `quant_pq2_0.metal` `pq2_0_block_dot` / the host oracle
// `checkpoint::codec::pq2_0::decode_block`: element `e` -> byte `2 + e/4`, bits
// `(e%4)*2`. No pow3, no staging.
inline int pq2_0_block_code(const device uint8_t *blk, int e) {
  return (int(blk[PQ2_0_QS_OFFSET + e / 4]) >> ((e % 4) * 2)) & 0x3;
}

// Decode a native PQ2_0 bank (`columns` rows x `width` contraction, each row
// `ceil(width/128)*34` bytes) into the affine MPP packed layout. One thread per
// output uint (8 uint4b codes = 8 lanes of a 64-group), matching the grid shape
// of `qmm_mpp_pack_bank` / `ptq1_0_qmm_mpp_pack_bank`.
kernel void pq2_0_qmm_mpp_pack_bank(const device uint8_t *codes [[buffer(0)]],
                                    device uint *packed [[buffer(1)]],
                                    constant uint &width [[buffer(2)]],
                                    constant uint &columns [[buffer(3)]],
                                    uint word [[thread_position_in_grid]]) {
  ulong words = ulong(columns) * width / 8;
  if (word >= words)
    return;
  uint groups = width / 64;
  uint column = word / (width / 8);
  uint group = (word % (width / 8)) / 8;
  uint local = word % 8; // which of the group's 8 uints this thread fills
  ulong parameter = (ulong(column / 256) * groups + group) * 256 + column % 256;

  uint row_bytes = (width / 128) * PQ2_0_BLOCK_BYTES;
  uint block_half = group % 2; // 0 -> block elements [0,64), 1 -> [64,128)
  const device uint8_t *blk =
      codes + ulong(column) * row_bytes + ulong(group / 2) * PQ2_0_BLOCK_BYTES;

  uint value = 0;
  for (uint i = 0; i < 8; i++) {
    uint lane = local * 8 + i;
    int code = pq2_0_block_code(blk, int(block_half * 64 + lane));
    value |= (uint(code) & 0xf) << (i * 4);
  }
  packed[parameter * 8 + local] = value;

  if (local == 0) {
    ushort dbits = ushort(uint(blk[0]) | (uint(blk[1]) << 8));
    float d = float(as_type<half>(dbits));
    device bfloat *factors = reinterpret_cast<device bfloat *>(packed + words);
    factors[parameter] = bfloat(d);
    factors[ulong(columns) * groups + parameter] = bfloat(-d);
  }
}

// --- PTQ1_0 ternary -> affine MPP packed front-end -------------------------
//
// PTQ1_0 reconstructs a weight as `w = (trit - 1) * d`, trit in {0,1,2}, with
// one fp16 scale `d` per 128-weight block. That is algebraically the affine MPP
// accumulate `total = sum(code*x)*scale + sum(x)*bias` with code=trit (0,1,2),
// scale=d, bias=-d. So we DECODE each 28-byte PTQ1_0 block into the exact affine
// `qmm_mpp_pack_bank` layout (uint4b codes + per-64-group bf16 scale/bias) and
// the existing `affine_qmm_mpp` PACKED kernel runs unchanged and bit-exactly.
//
// A 128-weight block spans two consecutive 64-groups (g, g+1); both get the same
// d (scale) and -d (bias), so the sum over the two 64-groups equals the sum over
// the 128 block. The block's natural element order 0..127 (the fork's {16,8}+qh
// staging) is mirrored here exactly; 64-group g owns block g/2, half g%2, i.e.
// block elements [64*(g%2) .. 64*(g%2)+64), and lane l of the group is block
// element 64*(g%2)+l -> uint l/8, nibble (l%8)*4 (the affine lane order).

#define PTQ1_0_BLOCK_BYTES 28

constant int ptq1_0_pow3[5] = {1, 3, 9, 27, 81};

// One base-3 trit (0,1,2) for natural block element `e` of a 28-byte block,
// mirroring `quant_ptq1_0.metal` / `checkpoint::codec::ptq1_0::decode_block`:
// `q = (byte * 3^n) & 0xff; trit = (q*3) >> 8`, with the {16,8}+qh staging.
inline int ptq1_0_block_trit(const device uint8_t *blk, int e) {
  int byte_idx, n;
  if (e < 80) { // qs chunk A: bytes 0..15, 5 trit-slots, natural elements 0..79
    n = e / 16;
    byte_idx = e % 16;
  } else if (e < 120) { // qs chunk B: bytes 16..23, elements 80..119
    int e2 = e - 80;
    n = e2 / 8;
    byte_idx = 16 + (e2 % 8);
  } else { // qh tail: bytes 24..25, 4 trit-slots, elements 120..127
    int e3 = e - 120;
    n = e3 / 2;
    byte_idx = 24 + (e3 % 2);
  }
  int q = (int(blk[byte_idx]) * ptq1_0_pow3[n]) & 0xff;
  return (q * 3) >> 8;
}

// Decode a native PTQ1_0 bank (`columns` rows x `width` contraction, each row
// `ceil(width/128)*28` bytes) into the affine MPP packed layout. One thread per
// output uint (8 uint4b codes = 8 lanes of a 64-group), matching the grid shape
// of `qmm_mpp_pack_bank`.
kernel void ptq1_0_qmm_mpp_pack_bank(const device uint8_t *codes [[buffer(0)]],
                                     device uint *packed [[buffer(1)]],
                                     constant uint &width [[buffer(2)]],
                                     constant uint &columns [[buffer(3)]],
                                     uint word [[thread_position_in_grid]]) {
  ulong words = ulong(columns) * width / 8;
  if (word >= words)
    return;
  uint groups = width / 64;
  uint column = word / (width / 8);
  uint group = (word % (width / 8)) / 8;
  uint local = word % 8; // which of the group's 8 uints this thread fills
  ulong parameter = (ulong(column / 256) * groups + group) * 256 + column % 256;

  uint row_bytes = (width / 128) * PTQ1_0_BLOCK_BYTES;
  uint block_half = group % 2; // 0 -> block elements [0,64), 1 -> [64,128)
  const device uint8_t *blk =
      codes + ulong(column) * row_bytes + ulong(group / 2) * PTQ1_0_BLOCK_BYTES;

  uint value = 0;
  for (uint i = 0; i < 8; i++) {
    uint lane = local * 8 + i;
    int trit = ptq1_0_block_trit(blk, int(block_half * 64 + lane));
    value |= (uint(trit) & 0xf) << (i * 4);
  }
  packed[parameter * 8 + local] = value;

  if (local == 0) {
    ushort dbits = ushort(uint(blk[26]) | (uint(blk[27]) << 8));
    float d = float(as_type<half>(dbits));
    device bfloat *factors = reinterpret_cast<device bfloat *>(packed + words);
    factors[parameter] = bfloat(d);
    factors[ulong(columns) * groups + parameter] = bfloat(-d);
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
