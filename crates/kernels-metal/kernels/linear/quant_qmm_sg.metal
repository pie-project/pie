#include <metal_stdlib>
#include "../common/register_mma.metal"
using namespace metal;

// Up to eight rows of a 4-bit, 64-group affine matmul on the 8x8 simdgroup
// matrix MMA, its operands built in registers: the tile a GPU without neural
// accelerators (Apple9: M3, M4) runs a verify or draft fire on, as Splash's
// register decode tile does there. The MMA runs on the FP32 pipe on those
// GPUs, so everything else stays off it:
// - a code enters the MMA as the exact bf16 128 + code, two at once by OR-ing
//   0x4300 into a word's halves;
// - one chain per 64 codes (one scale, one bias), closed in fp32 as
//   s * chain + (b - 128 s) * sum, where sum is the row's sum of those inputs;
// - the rows' inputs are laid out once per matmul (`qmm_sg_table`) in the
//   order the lanes take them, with their sums, so a lane reads its 16 inputs
//   of a 64-code group as one load and no threadgroup stages them;
// - a lane reads its two words of a 64-code group (one scale, one bias) as
//   one 8-byte load per fragment, the next two groups' while these two's MMAs run.
// A threadgroup is 4 simdgroups of 16 columns; a lane holds column fm of each
// 8-column fragment and reads words 2c and 2c + 1 (c = fn / 2) of its 64-code
// group, whose pair p (codes p and p + 4 of word 2c + j) is its operand of
// MMA step (j, p): the step's eight k are the four lanes' pairs, all of one
// group. `SPLIT` is the grid's y extent: partitions of the contraction,
// written to `partials` and added, in partition order, by the last partition
// of each column tile to arrive.

namespace qmm_sg {

constant constexpr int kSpan = 256;

} // namespace qmm_sg

// The rows' inputs as the tile's lanes take them: for each 64-code group h and B-operand lane
// slot (fm, c), the 16 inputs of rows 2c and 2c + 1 at MMA steps (j, p) = 0..1 x 0..3, which sit
// at k = 64h + 8(2c' + j) + p + 4i for the A lane c' = fm / 2 that pairs with B row fm (i = fm % 2);
// then each row's sum of each 32 inputs, [group][row], at `table` + 8 * width; then the split
// tile's arrival counters, one per 64 columns, which it zeroes.
[[kernel]] void qmm_sg_table(const device bfloat *input [[buffer(0)]], device bfloat *table [[buffer(1)]],
                             constant int &width [[buffer(2)]], constant int &rows [[buffer(3)]],
                             constant int &counters [[buffer(4)]],
                             uint2 at [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  // A simdgroup per 64-code group h (grid.y), a lane per slot (grid.x).
  const uint slot = at.x, h = at.y;
  // The split tile's arrival counters, past the sums: zero before the tile counts on them.
  device uint *count = reinterpret_cast<device uint *>(table + 8 * ulong(width)) + width / 4;
  for (uint at_ = h * 32 + slot; at_ < uint(counters); at_ += grid.x * grid.y)
    count[at_] = 0;
  const uint fm = slot >> 2, c = slot & 3;
  const uint cp = fm >> 1, i = fm & 1;
  vec<bfloat, 16> v;
  float sum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint k = 64 * h + 8 * (2 * cp + j) + p + 4 * i;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int row = 2 * c + r;
        const bfloat x = row < rows ? input[ulong(row) * width + k] : bfloat(0.0f);
        v[8 * j + 2 * p + r] = x;
        sum[r] += float(x);
      }
    }
  reinterpret_cast<device vec<bfloat, 16> *>(table)[ulong(h) * 32 + slot] = v;
  // A lane's 16 inputs are 8 of each of its two rows, all in the 32-input group fm / 4; the
  // four lanes of that group and rows (fm bits 0 and 1) add theirs.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    sum[r] += simd_shuffle_xor(sum[r], 4u);
    sum[r] += simd_shuffle_xor(sum[r], 8u);
  }
  if ((fm & 3) == 0) {
    device float *sums = reinterpret_cast<device float *>(table + 8 * ulong(width));
#pragma unroll
    for (int r = 0; r < 2; ++r)
      sums[(2 * ulong(h) + (fm >> 2)) * 8 + 2 * c + r] = sum[r];
  }
}

template <int SPLIT>
[[kernel]] void
affine_qmm_sg(const device uint *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
              const device bfloat *biases [[buffer(2)]], const device bfloat *table [[buffer(3)]],
              device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
              constant int &columns [[buffer(6)]], device float *partials [[buffer(7)]],
              constant int &rows [[buffer(8)]], constant int &stride [[buffer(9)]],
              uint2 position [[threadgroup_position_in_grid]],
              uint lane [[thread_index_in_simdgroup]],
              uint simd [[simdgroup_index_in_threadgroup]]) {
  using namespace qmm_sg;
  constexpr int NF = 2; // 8-column fragments a simdgroup holds
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint c = fn >> 1;
  const uint column = position.x * 32 * NF + simd * 8 * NF + fm;
  const ulong words = ulong(stride) / 8, groups = ulong(stride) / 64;
  const device uint2 *w0 = reinterpret_cast<const device uint2 *>(weights + ulong(column) * words) + c;
  const ulong apart = 8 * words / 2; // uint2s between the two fragments
  const device bfloat *s0 = scales + ulong(column) * groups, *b0 = biases + ulong(column) * groups;
  const device vec<bfloat, 16> *xt =
      reinterpret_cast<const device vec<bfloat, 16> *>(table) + fm * 4 + c;
  const device float *sums = reinterpret_cast<const device float *>(table + 8 * ulong(width));
  // Partition y takes 256-input units [y * units / SPLIT, (y + 1) * units / SPLIT), as Splash's
  // tile cuts them: a contraction need not hold SPLIT equal parts (down's 68 units take 8).
  const int units = width / kSpan;
  const int first = int(position.y) * units / SPLIT * 4, last = (int(position.y) + 1) * units / SPLIT * 4;
  // Two groups a step: their words, scales and biases load while the step before runs.
  constexpr int U = 2;
  float2 acc[NF];
  uint2 q[U][NF];
  float qs[U][NF], qb[U][NF];
#pragma unroll
  for (int f = 0; f < NF; ++f)
    acc[f] = float2(0.0f);
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      q[u][f] = w0[f * apart + ulong(first + u) * 4];
      qs[u][f] = float(s0[8 * f * groups + first + u]);
      qb[u][f] = float(b0[8 * f * groups + first + u]);
    }
  for (int h0 = first; h0 < last; h0 += U) {
    uint2 cw[U][NF];
    float2 ck[U][NF];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        cw[u][f] = q[u][f];
        ck[u][f] = float2(qs[u][f], fma(-128.0f, qs[u][f], qb[u][f]));
      }
    if (h0 + U < last) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int f = 0; f < NF; ++f) {
          q[u][f] = w0[f * apart + ulong(h0 + U + u) * 4];
          qs[u][f] = float(s0[8 * f * groups + h0 + U + u]);
          qb[u][f] = float(b0[8 * f * groups + h0 + U + u]);
        }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int h = h0 + u;
      const vec<bfloat, 16> x = xt[ulong(h) * 32];
      // One chain per 64 codes (one scale and bias).
      float2 d[NF];
#pragma unroll
      for (int f = 0; f < NF; ++f)
        d[f] = float2(0.0f);
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (uint p = 0; p < 4; ++p) {
          const bfloat2 b = bfloat2(x[8 * j + 2 * p], x[8 * j + 2 * p + 1]);
#pragma unroll
          for (int f = 0; f < NF; ++f)
            mma(d[f], as_type<bfloat2>(((cw[u][f][j] >> (4 * p)) & 0x000F000Fu) | 0x43004300u), b);
        }
      const float2 sm = *reinterpret_cast<const device float2 *>(sums + 2 * ulong(h) * 8 + fn) +
                        *reinterpret_cast<const device float2 *>(sums + (2 * ulong(h) + 1) * 8 + fn);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        acc[f] = fma(d[f], ck[u][f].x, acc[f]);
        acc[f] = fma(sm, ck[u][f].y, acc[f]);
      }
    }
  }
  if constexpr (SPLIT > 1) {
    // Each partition publishes its sums; the last of a column tile's to arrive adds them in
    // partition order, as the split-k reduce did, and writes the output.
    device coherent(device) float *plane = partials;
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        plane[ulong(position.y) * rows * columns + o + 8 * f] = acc[f][i];
    }
    threadgroup uint arrival;
    device atomic_uint *count = reinterpret_cast<device atomic_uint *>(
                                    const_cast<device bfloat *>(table) + 8 * ulong(width)) +
                                width / 4 + position.x;
    const uint me = simd * 32 + lane;
    threadgroup_barrier(mem_flags::mem_device);
    if (me == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival != SPLIT - 1)
      return;
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        float total = 0.0f;
        for (int s = 0; s < SPLIT; ++s)
          total += s == int(position.y) ? acc[f][i] : plane[ulong(s) * rows * columns + o + 8 * f];
        output[o + 8 * f] = bfloat(total);
      }
    }
    if (me == 0)
      atomic_store_explicit(count, 0u, memory_order_relaxed);
  } else {
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        output[o + 8 * f] = bfloat(acc[f][i]);
    }
  }
}

#define PIE_SG_POINT(name, split)                                                                  \
  template [[host_name(name)]] [[kernel]] void affine_qmm_sg<split>(                               \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);

// Up to three matmuls of one table in one launch (a layer's projections of its normed rows: a
// GDN in-projection and its gates; a query, key and value), each segment s the columns of its own
// bank from column `offsets[s]` (o0, o1, o2; the bank's row stride `strides[s]`, d0, d1, d2) into
// its own output, rows of `columns[s]` (c0, c1, c2), in `splits[s]` partitions (p0, p1, p2): each
// column summed as `affine_qmm_sg` at that split sums it. The grid is one row of threadgroups:
// segment s takes columns[s] / 64 x splits[s] of them, partition-major, after the segments before
// it; a split segment's partials follow those of the split segments before it in `partials`, and
// its tiles' arrival counters theirs in the table.
template <int S>
[[kernel]] void
affine_qmm_sg_segments(const device uint *w0 [[buffer(0)]], const device bfloat *s0 [[buffer(1)]],
                       const device bfloat *b0 [[buffer(2)]], device bfloat *y0 [[buffer(3)]],
                       const device uint *w1 [[buffer(4)]], const device bfloat *s1 [[buffer(5)]],
                       const device bfloat *b1 [[buffer(6)]], device bfloat *y1 [[buffer(7)]],
                       const device uint *w2 [[buffer(8)]], const device bfloat *s2 [[buffer(9)]],
                       const device bfloat *b2 [[buffer(10)]], device bfloat *y2 [[buffer(11)]],
                       const device bfloat *table [[buffer(12)]], device float *partials [[buffer(13)]],
                       constant int &width [[buffer(14)]], constant int &rows [[buffer(15)]],
                       constant int &c0 [[buffer(16)]], constant int &c1 [[buffer(17)]],
                       constant int &c2 [[buffer(18)]], constant int &p0 [[buffer(19)]],
                       constant int &p1 [[buffer(20)]], constant int &p2 [[buffer(21)]],
                       constant int &d0 [[buffer(22)]], constant int &d1 [[buffer(23)]],
                       constant int &d2 [[buffer(24)]], constant int &o0 [[buffer(25)]],
                       constant int &o1 [[buffer(26)]], constant int &o2 [[buffer(27)]],
                       uint position [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
  using namespace qmm_sg;
  constexpr int NF = 2; // 8-column fragments a simdgroup holds
  const int seg_columns[3] = {c0, c1, c2}, splits[3] = {p0, p1, p2}, strides[3] = {d0, d1, d2};
  const int offsets[3] = {o0, o1, o2};
  // The segment this threadgroup falls in, and the threadgroups, counters and partials before it.
  int s = 0;
  uint before = 0, tiles_before = 0;
  ulong floats_before = 0;
#pragma unroll
  for (int t = 0; t < S - 1; ++t) {
    const uint tiles = uint(seg_columns[t]) / 64, tgs = tiles * uint(splits[t]);
    if (s == t && position >= before + tgs) {
      s = t + 1;
      before += tgs;
      if (splits[t] > 1) {
        tiles_before += tiles;
        floats_before += ulong(splits[t]) * ulong(rows) * ulong(seg_columns[t]);
      }
    }
  }
  const device uint *weights = s == 0 ? w0 : s == 1 ? w1 : w2;
  const device bfloat *scales = s == 0 ? s0 : s == 1 ? s1 : s2;
  const device bfloat *biases = s == 0 ? b0 : s == 1 ? b1 : b2;
  device bfloat *output = s == 0 ? y0 : s == 1 ? y1 : y2;
  const int columns = seg_columns[s], split = splits[s], stride = strides[s];
  const uint tiles = uint(columns) / 64, local = position - before;
  const uint tile = local % tiles, part = local / tiles;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint c = fn >> 1;
  const uint column = tile * 32 * NF + simd * 8 * NF + fm;
  const uint wcolumn = uint(offsets[s]) + column;
  const ulong words = ulong(stride) / 8, groups = ulong(stride) / 64;
  const device uint2 *wp = reinterpret_cast<const device uint2 *>(weights + ulong(wcolumn) * words) + c;
  const ulong apart = 8 * words / 2; // uint2s between the two fragments
  const device bfloat *sp = scales + ulong(wcolumn) * groups, *bp = biases + ulong(wcolumn) * groups;
  const device vec<bfloat, 16> *xt =
      reinterpret_cast<const device vec<bfloat, 16> *>(table) + fm * 4 + c;
  const device float *sums = reinterpret_cast<const device float *>(table + 8 * ulong(width));
  // Partition `part` takes 256-input units as `affine_qmm_sg`'s partition y does.
  const int units = width / kSpan;
  const int first = int(part) * units / split * 4, last = (int(part) + 1) * units / split * 4;
  constexpr int U = 2;
  float2 acc[NF];
  uint2 q[U][NF];
  float qs[U][NF], qb[U][NF];
#pragma unroll
  for (int f = 0; f < NF; ++f)
    acc[f] = float2(0.0f);
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      q[u][f] = wp[f * apart + ulong(first + u) * 4];
      qs[u][f] = float(sp[8 * f * groups + first + u]);
      qb[u][f] = float(bp[8 * f * groups + first + u]);
    }
  for (int h0 = first; h0 < last; h0 += U) {
    uint2 cw[U][NF];
    float2 ck[U][NF];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        cw[u][f] = q[u][f];
        ck[u][f] = float2(qs[u][f], fma(-128.0f, qs[u][f], qb[u][f]));
      }
    if (h0 + U < last) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int f = 0; f < NF; ++f) {
          q[u][f] = wp[f * apart + ulong(h0 + U + u) * 4];
          qs[u][f] = float(sp[8 * f * groups + h0 + U + u]);
          qb[u][f] = float(bp[8 * f * groups + h0 + U + u]);
        }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int h = h0 + u;
      const vec<bfloat, 16> x = xt[ulong(h) * 32];
      float2 d[NF];
#pragma unroll
      for (int f = 0; f < NF; ++f)
        d[f] = float2(0.0f);
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (uint p = 0; p < 4; ++p) {
          const bfloat2 b = bfloat2(x[8 * j + 2 * p], x[8 * j + 2 * p + 1]);
#pragma unroll
          for (int f = 0; f < NF; ++f)
            mma(d[f], as_type<bfloat2>(((cw[u][f][j] >> (4 * p)) & 0x000F000Fu) | 0x43004300u), b);
        }
      const float2 sm = *reinterpret_cast<const device float2 *>(sums + 2 * ulong(h) * 8 + fn) +
                        *reinterpret_cast<const device float2 *>(sums + (2 * ulong(h) + 1) * 8 + fn);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        acc[f] = fma(d[f], ck[u][f].x, acc[f]);
        acc[f] = fma(sm, ck[u][f].y, acc[f]);
      }
    }
  }
  threadgroup uint arrival;
  if (split > 1) {
    device coherent(device) float *plane = partials + floats_before;
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        plane[ulong(part) * rows * columns + o + 8 * f] = acc[f][i];
    }
    device atomic_uint *count = reinterpret_cast<device atomic_uint *>(
                                    const_cast<device bfloat *>(table) + 8 * ulong(width)) +
                                width / 4 + tiles_before + tile;
    const uint me = simd * 32 + lane;
    threadgroup_barrier(mem_flags::mem_device);
    if (me == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival != uint(split) - 1)
      return;
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        float total = 0.0f;
        for (int t = 0; t < split; ++t)
          total += t == int(part) ? acc[f][i] : plane[ulong(t) * rows * columns + o + 8 * f];
        output[o + 8 * f] = bfloat(total);
      }
    }
    if (me == 0)
      atomic_store_explicit(count, 0u, memory_order_relaxed);
  } else {
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        output[o + 8 * f] = bfloat(acc[f][i]);
    }
  }
}

#define PIE_SG_SEGMENTS(name, s)                                                                   \
  template [[host_name(name)]] [[kernel]] void affine_qmm_sg_segments<s>(                          \
      const device uint *, const device bfloat *, const device bfloat *, device bfloat *,          \
      const device uint *, const device bfloat *, const device bfloat *, device bfloat *,          \
      const device uint *, const device bfloat *, const device bfloat *, device bfloat *,          \
      const device bfloat *, device float *, constant int &, constant int &, constant int &,       \
      constant int &, constant int &, constant int &, constant int &, constant int &,              \
      constant int &, constant int &, constant int &, constant int &, constant int &,              \
      constant int &, uint, uint, uint);

// diag: affine_qmm_sg with its prefetch depth U, fragments a simdgroup NF, simdgroups a
// threadgroup SIMDS and, VS, its U groups' scales and biases as one load each.
template <int SPLIT, int U, int NF, int SIMDS, int VS>
[[kernel]] void
affine_qmm_sgx(const device uint *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
               const device bfloat *biases [[buffer(2)]], const device bfloat *table [[buffer(3)]],
               device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
               constant int &columns [[buffer(6)]], device float *partials [[buffer(7)]],
               constant int &rows [[buffer(8)]], constant int &stride [[buffer(9)]],
               uint2 position [[threadgroup_position_in_grid]],
               uint lane [[thread_index_in_simdgroup]],
               uint simd [[simdgroup_index_in_threadgroup]]) {
  using namespace qmm_sg;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint c = fn >> 1;
  const uint column = position.x * 8 * NF * SIMDS + simd * 8 * NF + fm;
  const ulong words = ulong(stride) / 8, groups = ulong(stride) / 64;
  const device uint2 *w0 = reinterpret_cast<const device uint2 *>(weights + ulong(column) * words) + c;
  const ulong apart = 8 * words / 2;
  const device bfloat *s0 = scales + ulong(column) * groups, *b0 = biases + ulong(column) * groups;
  const device vec<bfloat, 16> *xt =
      reinterpret_cast<const device vec<bfloat, 16> *>(table) + fm * 4 + c;
  const device float *sums = reinterpret_cast<const device float *>(table + 8 * ulong(width));
  const int units = width / kSpan;
  const int first = int(position.y) * units / SPLIT * 4, last = (int(position.y) + 1) * units / SPLIT * 4;
  float2 acc[NF];
  uint2 q[U][NF];
  float qs[U][NF], qb[U][NF];
#pragma unroll
  for (int f = 0; f < NF; ++f)
    acc[f] = float2(0.0f);
  auto load = [&](int at) {
#pragma unroll
    for (int f = 0; f < NF; ++f) {
#pragma unroll
      for (int u = 0; u < U; ++u)
        q[u][f] = w0[f * apart + ulong(at + u) * 4];
      if constexpr (VS && U == 2) {
        const bfloat2 s = *reinterpret_cast<const device bfloat2 *>(s0 + 8 * f * groups + at);
        const bfloat2 b = *reinterpret_cast<const device bfloat2 *>(b0 + 8 * f * groups + at);
        qs[0][f] = float(s[0]); qs[U - 1][f] = float(s[1]);
        qb[0][f] = float(b[0]); qb[U - 1][f] = float(b[1]);
      } else if constexpr (VS && U == 4) {
        const vec<bfloat, 4> s = *reinterpret_cast<const device vec<bfloat, 4> *>(s0 + 8 * f * groups + at);
        const vec<bfloat, 4> b = *reinterpret_cast<const device vec<bfloat, 4> *>(b0 + 8 * f * groups + at);
#pragma unroll
        for (int u = 0; u < U; ++u) {
          qs[u][f] = float(s[u & 3]);
          qb[u][f] = float(b[u & 3]);
        }
      } else {
#pragma unroll
        for (int u = 0; u < U; ++u) {
          qs[u][f] = float(s0[8 * f * groups + at + u]);
          qb[u][f] = float(b0[8 * f * groups + at + u]);
        }
      }
    }
  };
  load(first);
  for (int h0 = first; h0 < last; h0 += U) {
    uint2 cw[U][NF];
    float2 ck[U][NF];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        cw[u][f] = q[u][f];
        ck[u][f] = float2(qs[u][f], fma(-128.0f, qs[u][f], qb[u][f]));
      }
    if (h0 + U < last)
      load(h0 + U);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int h = h0 + u;
      const vec<bfloat, 16> x = xt[ulong(h) * 32];
      float2 d[NF];
#pragma unroll
      for (int f = 0; f < NF; ++f)
        d[f] = float2(0.0f);
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (uint p = 0; p < 4; ++p) {
          const bfloat2 b = bfloat2(x[8 * j + 2 * p], x[8 * j + 2 * p + 1]);
#pragma unroll
          for (int f = 0; f < NF; ++f) {
            if constexpr (VS == 3) { if (p & 1) continue; }
            const uint sh = VS == 2 ? 0 : 4 * p;
            const uint w = cw[u][f][j];
            if constexpr (VS == 4) {
              const ushort lo = ushort(extract_bits(w, 4 * p, 4)) | ushort(0x4300);
              const ushort hi = ushort(extract_bits(w, 16 + 4 * p, 4)) | ushort(0x4300);
              mma(d[f], as_type<bfloat2>(ushort2(lo, hi)), b);
            } else if constexpr (VS == 5) {
              const ushort lo = ushort((w >> (4 * p)) & 0xFu) | ushort(0x4300);
              const ushort hi = ushort((w >> (16 + 4 * p)) & 0xFu) | ushort(0x4300);
              mma(d[f], as_type<bfloat2>(ushort2(lo, hi)), b);
            } else if constexpr (VS == 6) {
              const ushort2 h = as_type<ushort2>(w);
              const ushort2 c2 = ((h >> ushort(4 * p)) & ushort(0xF)) | ushort(0x4300);
              mma(d[f], as_type<bfloat2>(c2), b);
            } else {
              mma(d[f], as_type<bfloat2>(((w >> sh) & 0x000F000Fu) | 0x43004300u), b);
            }
          }
        }
      const float2 sm = *reinterpret_cast<const device float2 *>(sums + 2 * ulong(h) * 8 + fn) +
                        *reinterpret_cast<const device float2 *>(sums + (2 * ulong(h) + 1) * 8 + fn);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        acc[f] = fma(d[f], ck[u][f].x, acc[f]);
        acc[f] = fma(sm, ck[u][f].y, acc[f]);
      }
    }
  }
  if constexpr (SPLIT > 1) {
    device coherent(device) float *plane = partials;
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        plane[ulong(position.y) * rows * columns + o + 8 * f] = acc[f][i];
    }
    threadgroup uint arrival;
    device atomic_uint *count = reinterpret_cast<device atomic_uint *>(
                                    const_cast<device bfloat *>(table) + 8 * ulong(width)) +
                                width / 4 + position.x;
    const uint me = simd * 32 + lane;
    threadgroup_barrier(mem_flags::mem_device);
    if (me == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival != SPLIT - 1)
      return;
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        float total = 0.0f;
        for (int s = 0; s < SPLIT; ++s)
          total += s == int(position.y) ? acc[f][i] : plane[ulong(s) * rows * columns + o + 8 * f];
        output[o + 8 * f] = bfloat(total);
      }
    }
    if (me == 0)
      atomic_store_explicit(count, 0u, memory_order_relaxed);
  } else {
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        output[o + 8 * f] = bfloat(acc[f][i]);
    }
  }
}

#define PIE_SGX(name, split, u, nf, simds, vs)                                                     \
  template [[host_name(name)]] [[kernel]] void affine_qmm_sgx<split, u, nf, simds, vs>(            \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);
