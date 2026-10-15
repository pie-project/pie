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
