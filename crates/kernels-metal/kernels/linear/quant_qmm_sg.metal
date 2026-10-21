#include <metal_stdlib>
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
// - a lane reads its two words of a 64-code group as one 8-byte load per
//   fragment, and the scales and biases of `V` groups as one load each, the
//   next `U` groups' while these groups' MMAs run.
// A threadgroup is 4 simdgroups of 16 columns; a lane holds column fm of each
// 8-column fragment and reads words 2c and 2c + 1 (c = fn / 2) of its 64-code
// group, whose pair p (codes p and p + 4 of word 2c + j) is its operand of
// MMA step (j, p): the step's eight k are the four lanes' pairs, all of one
// group. A matmul's contraction may split in partitions, each written to
// `partials` and added, in partition order, by the last partition of each
// column tile to arrive.

namespace qmm_sg {

template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

// c += a x b on operands held in plain registers: lane (fm, fn) holds
// M[fm][fn] and M[fm][fn + 1] of each.
__attribute__((always_inline)) inline void mma(thread float2 &c, bfloat2 a, bfloat2 b) {
  simdgroup_matrix<bfloat, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

constant constexpr int kSpan = 256;
constant constexpr int NF = 2; // 8-column fragments a simdgroup holds

// A lane's place in the MMA fragments.
struct Lane {
  uint fm, fn, c;
};
__attribute__((always_inline)) inline Lane lane_of(uint lane) {
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  return {fm, fn, fn >> 1};
}

// The scales (or biases) of `V` groups of a column from group `h`, one load: V is 1, 2 or 4 and
// h a multiple of V.
template <int V>
__attribute__((always_inline)) inline vec<bfloat, V> factors(const device bfloat *row, int h) {
  if constexpr (V == 1)
    return vec<bfloat, 1>(row[h]);
  else
    return reinterpret_cast<const device vec<bfloat, V> *>(row)[h / V];
}

// One partition of a column tile's contraction, the 64-code groups [first, last), into `acc`:
// lane `l` of the simdgroup at column `column - fm`. `stride` is the bank's row stride, `width`
// the table's contraction. Each group's chain closes into `acc` in group order, `U` groups a
// step, their scales and biases `V` groups a load.
template <int U, int V>
__attribute__((always_inline)) inline void walk(const device uint *weights, const device bfloat *scales,
                                                const device bfloat *biases, const device bfloat *table,
                                                int width, int stride, uint column, Lane l, int first,
                                                int last, thread float2 (&acc)[NF]) {
  static_assert(U % V == 0, "a step loads whole factor vectors");
  constexpr int FV = U / V;
  const ulong words = ulong(stride) / 8, groups = ulong(stride) / 64;
  const device uint2 *w0 = reinterpret_cast<const device uint2 *>(weights + ulong(column) * words) + l.c;
  const ulong apart = 8 * words / 2; // uint2s between the two fragments
  const device bfloat *s0 = scales + ulong(column) * groups, *b0 = biases + ulong(column) * groups;
  const device vec<bfloat, 16> *xt =
      reinterpret_cast<const device vec<bfloat, 16> *>(table) + l.fm * 4 + l.c;
  const device float *sums = reinterpret_cast<const device float *>(table + 8 * ulong(width));
  uint2 q[U][NF];
  vec<bfloat, V> qs[NF][FV], qb[NF][FV];
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    acc[f] = float2(0.0f);
#pragma unroll
    for (int u = 0; u < U; ++u)
      q[u][f] = w0[f * apart + ulong(first + u) * 4];
#pragma unroll
    for (int v = 0; v < FV; ++v) {
      qs[f][v] = factors<V>(s0 + 8 * f * groups, first + v * V);
      qb[f][v] = factors<V>(b0 + 8 * f * groups, first + v * V);
    }
  }
  for (int h0 = first; h0 < last; h0 += U) {
    uint2 cw[U][NF];
    float2 ck[U][NF];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        cw[u][f] = q[u][f];
        const float s = float(qs[f][u / V][u % V]);
        ck[u][f] = float2(s, fma(-128.0f, s, float(qb[f][u / V][u % V])));
      }
    if (h0 + U < last) {
#pragma unroll
      for (int f = 0; f < NF; ++f) {
#pragma unroll
        for (int u = 0; u < U; ++u)
          q[u][f] = w0[f * apart + ulong(h0 + U + u) * 4];
#pragma unroll
        for (int v = 0; v < FV; ++v) {
          qs[f][v] = factors<V>(s0 + 8 * f * groups, h0 + U + v * V);
          qb[f][v] = factors<V>(b0 + 8 * f * groups, h0 + U + v * V);
        }
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
      const float2 sm = *reinterpret_cast<const device float2 *>(sums + 2 * ulong(h) * 8 + l.fn) +
                        *reinterpret_cast<const device float2 *>(sums + (2 * ulong(h) + 1) * 8 + l.fn);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        acc[f] = fma(d[f], ck[u][f].x, acc[f]);
        acc[f] = fma(sm, ck[u][f].y, acc[f]);
      }
    }
  }
}

// Partition `part` of `split` takes the 256-input units [part * units / split, (part + 1) *
// units / split), as 64-code groups, as Splash's tile cuts them: a contraction need not hold
// `split` equal parts (down's 68 units take 8).
__attribute__((always_inline)) inline int2 partition(int width, int split, int part) {
  const int units = width / kSpan;
  return int2(part * units / split * 4, (part + 1) * units / split * 4);
}

// Lands a column tile's `acc` in `output`, rows of `columns`: whole, or one of `split`
// partitions, each publishing its sums to `partials` ([split][rows][columns]) and the last of the
// tile's to arrive at `count` adding them in partition order and writing the output.
__attribute__((always_inline)) inline void land(thread float2 (&acc)[NF], device bfloat *output, int columns,
                                                int rows, uint column, Lane l, int split, uint part,
                                                device float *partials, device atomic_uint *count,
                                                uint me, threadgroup uint &arrival) {
  if (split > 1) {
    device coherent(device) float *plane = partials;
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = l.fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        plane[ulong(part) * rows * columns + o + 8 * f] = acc[f][i];
    }
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
      const uint row = l.fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        float total = 0.0f;
        for (int s = 0; s < split; ++s)
          total += s == int(part) ? acc[f][i] : plane[ulong(s) * rows * columns + o + 8 * f];
        output[o + 8 * f] = bfloat(total);
      }
    }
    if (me == 0)
      atomic_store_explicit(count, 0u, memory_order_relaxed);
  } else {
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const uint row = l.fn + i;
      if (int(row) >= rows)
        continue;
      const ulong o = ulong(row) * columns + column;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        output[o + 8 * f] = bfloat(acc[f][i]);
    }
  }
}

// The table's arrival counters, one per 64 columns, past its inputs and sums.
__attribute__((always_inline)) inline device atomic_uint *counters(const device bfloat *table, int width) {
  return reinterpret_cast<device atomic_uint *>(const_cast<device bfloat *>(table) + 8 * ulong(width)) +
         width / 4;
}

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

// One matmul: a threadgroup per 64-column tile (x) and partition (y) of `SPLIT`.
template <int SPLIT, int U, int V>
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
  const Lane l = lane_of(lane);
  const uint column = position.x * 32 * NF + simd * 8 * NF + l.fm;
  const int2 span = partition(width, SPLIT, int(position.y));
  float2 acc[NF];
  walk<U, V>(weights, scales, biases, table, width, stride, column, l, span.x, span.y, acc);
  threadgroup uint arrival;
  land(acc, output, columns, rows, column, l, SPLIT, position.y, partials,
       counters(table, width) + position.x, simd * 32 + lane, arrival);
}

#define PIE_SG_POINT(name, split, u, v)                                                            \
  template [[host_name(name)]] [[kernel]] void affine_qmm_sg<split, u, v>(                         \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint);

// Up to three matmuls of one table in one launch (a layer's projections of its normed rows: a
// GDN in-projection and its gates; a query, key and value), each segment s its own bank, output
// (rows of `columns[s]` = c0, c1, c2; its bank's row stride `strides[s]` = d0, d1, d2) and
// partitions `splits[s]` = p0, p1, p2, each
// summed as that matmul alone sums it. The grid is one row of threadgroups: segment s takes
// columns[s] / 64 x splits[s] of them, partition-major, after the segments before it; a split
// segment's partials follow those of the split segments before it in `partials`, and its tiles'
// arrival counters theirs in the table.
template <int S, int U, int V>
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
                       constant int &d2 [[buffer(24)]], uint position [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
  using namespace qmm_sg;
  const int columns[3] = {c0, c1, c2}, splits[3] = {p0, p1, p2}, strides[3] = {d0, d1, d2};
  // The segment this threadgroup falls in, and the threadgroups, counters and partials before it.
  int s = 0;
  uint first = 0, tiles_before = 0;
  ulong floats_before = 0;
#pragma unroll
  for (int t = 0; t < S - 1; ++t) {
    const uint tiles = uint(columns[t]) / 64, groups = tiles * uint(splits[t]);
    if (s == t && position >= first + groups) {
      s = t + 1;
      first += groups;
      if (splits[t] > 1) {
        tiles_before += tiles;
        floats_before += ulong(splits[t]) * ulong(rows) * ulong(columns[t]);
      }
    }
  }
  const device uint *weights = s == 0 ? w0 : s == 1 ? w1 : w2;
  const device bfloat *scales = s == 0 ? s0 : s == 1 ? s1 : s2;
  const device bfloat *biases = s == 0 ? b0 : s == 1 ? b1 : b2;
  device bfloat *output = s == 0 ? y0 : s == 1 ? y1 : y2;
  const int n = columns[s], split = splits[s];
  const uint tiles = uint(n) / 64, local = position - first;
  const uint tile = local % tiles, part = local / tiles;
  const Lane l = lane_of(lane);
  const uint column = tile * 32 * NF + simd * 8 * NF + l.fm;
  const int2 span = partition(width, split, int(part));
  float2 acc[NF];
  walk<U, V>(weights, scales, biases, table, width, strides[s], column, l, span.x, span.y, acc);
  threadgroup uint arrival;
  land(acc, output, n, rows, column, l, split, part, partials + floats_before,
       counters(table, width) + tiles_before + tile, simd * 32 + lane, arrival);
}

#define PIE_SG_SEGMENTS(name, s, u, v)                                                             \
  template [[host_name(name)]] [[kernel]] void affine_qmm_sg_segments<s, u, v>(                    \
      const device uint *, const device bfloat *, const device bfloat *, device bfloat *,          \
      const device uint *, const device bfloat *, const device bfloat *, device bfloat *,          \
      const device uint *, const device bfloat *, const device bfloat *, device bfloat *,          \
      const device bfloat *, device float *, constant int &, constant int &, constant int &,       \
      constant int &, constant int &, constant int &, constant int &, constant int &,              \
      constant int &, constant int &, constant int &, uint, uint, uint);
