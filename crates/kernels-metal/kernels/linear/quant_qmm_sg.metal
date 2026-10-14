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
// - the threadgroup stages up to 1024 of the rows' inputs at a time,
//   transposed, with their sums, for its four simdgroups, while each lane's
//   next 256 codes load during this span's MMAs.
// A threadgroup is 4 simdgroups of 16 columns; a lane holds column fm of each
// 8-column fragment and reads word fn / 2 of its 32-code group, whose pair p
// (codes p and p + 4 of the word) is its operand of fragment p. The inputs
// follow that order. `SPLIT` is the grid's y extent: partitions of the
// contraction, written to `partials` and folded by the split-k reduce.

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
constant constexpr int kGroups = kSpan / 32;
// Spans staged at once: one barrier pair per 1024 inputs.
constant constexpr int kStage = 4;

} // namespace qmm_sg

template <int SPLIT>
[[kernel]] void
affine_qmm_sg(const device uint *weights [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
              const device bfloat *biases [[buffer(2)]], const device bfloat *input [[buffer(3)]],
              device bfloat *output [[buffer(4)]], constant int &width [[buffer(5)]],
              constant int &columns [[buffer(6)]], device float *partials [[buffer(7)]],
              constant int &rows [[buffer(8)]], constant int &stride [[buffer(9)]],
              uint2 position [[threadgroup_position_in_grid]],
              uint tid [[thread_index_in_threadgroup]],
              uint lane [[thread_index_in_simdgroup]],
              uint simd [[simdgroup_index_in_threadgroup]]) {
  using namespace qmm_sg;
  constexpr int NF = 2; // 8-column fragments a simdgroup holds
  // Up to kStage spans of the rows' inputs as [k][row], and their rows' sums of each 32
  // inputs as [group][row].
  threadgroup bfloat xt[kStage * kSpan * 8];
  threadgroup float xs[kStage * kGroups * 8];

  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint c = fn >> 1;
  const uint column = position.x * 32 * NF + simd * 8 * NF + fm;
  const ulong words = ulong(stride) / 8, groups = ulong(stride) / 64;
  const device uint *w0 = weights + ulong(column) * words + c;
  const device bfloat *s0 = scales + ulong(column) * groups, *b0 = biases + ulong(column) * groups;
  // The physical k, within a 32-code group, of this lane's input row fm of fragment p is
  // kp + p: logical k 2j and 2j + 1 of a fragment are codes 8j + p and 8j + p + 4.
  const uint kp = 8 * (fm >> 1) + 4 * (fm & 1);

  // Staging: thread t loads 16 inputs of row t / 16 at 16 (t % 16) of each span.
  const uint srow = tid / 16, sseg = tid % 16;
  const bool live = int(srow) < rows;
  const device bfloat *xin = input + ulong(min(int(srow), rows - 1)) * width + sseg * 16;

  const int spans = width / kSpan / SPLIT;
  const int first = int(position.y) * spans, last = first + spans;
  float2 acc[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f)
    acc[f] = float2(0.0f);
  uint q[NF][kGroups];
  float2 kc[NF][kGroups / 2];
  const auto fetch = [&](int span) __attribute__((always_inline)) {
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int g = 0; g < kGroups; ++g)
        q[f][g] = w0[8 * f * words + (span * kGroups + g) * 4];
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int g = 0; g < kGroups / 2; ++g) {
        const float a = float(s0[8 * f * groups + span * 4 + g]);
        kc[f][g] = float2(a, fma(-128.0f, a, float(b0[8 * f * groups + span * 4 + g])));
      }
  };
  fetch(first);
  for (int chunk = first; chunk < last; chunk += kStage) {
    const int n = min(kStage, last - chunk);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < n; ++j) {
      const vec<bfloat, 8> lo = *reinterpret_cast<const device vec<bfloat, 8> *>(xin + (chunk + j) * kSpan);
      const vec<bfloat, 8> hi = *reinterpret_cast<const device vec<bfloat, 8> *>(xin + (chunk + j) * kSpan + 8);
      threadgroup bfloat *into = xt + j * kSpan * 8;
      float sum = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const bfloat a = live ? lo[i] : bfloat(0.0f), b = live ? hi[i] : bfloat(0.0f);
        into[(sseg * 16 + i) * 8 + srow] = a;
        into[(sseg * 16 + 8 + i) * 8 + srow] = b;
        sum += float(a) + float(b);
      }
      sum += simd_shuffle_xor(sum, 1u);
      if ((sseg & 1) == 0)
        xs[(j * kGroups + (sseg >> 1)) * 8 + srow] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < n; ++j) {
      uint cq[NF][kGroups];
      float2 ck[NF][kGroups / 2];
#pragma unroll
      for (int f = 0; f < NF; ++f) {
#pragma unroll
        for (int g = 0; g < kGroups; ++g)
          cq[f][g] = q[f][g];
#pragma unroll
        for (int g = 0; g < kGroups / 2; ++g)
          ck[f][g] = kc[f][g];
      }
      if (chunk + j + 1 < last)
        fetch(chunk + j + 1);
      threadgroup const bfloat *from = xt + j * kSpan * 8;
      threadgroup const float *sums = xs + j * kGroups * 8;
      // One chain per 64 codes (one scale and bias), two 32-code groups.
#pragma unroll
      for (int h = 0; h < kGroups / 2; ++h) {
        float2 d[NF];
#pragma unroll
        for (int f = 0; f < NF; ++f)
          d[f] = float2(0.0f);
#pragma unroll
        for (int g = 2 * h; g < 2 * h + 2; ++g)
#pragma unroll
          for (uint p = 0; p < 4; ++p) {
            const bfloat2 x = *reinterpret_cast<threadgroup const bfloat2 *>(from + (g * 32 + kp + p) * 8 + fn);
#pragma unroll
            for (int f = 0; f < NF; ++f)
              mma(d[f], as_type<bfloat2>(((cq[f][g] >> (4 * p)) & 0x000F000Fu) | 0x43004300u), x);
          }
        const float2 sm = *reinterpret_cast<threadgroup const float2 *>(sums + 2 * h * 8 + fn) +
                          *reinterpret_cast<threadgroup const float2 *>(sums + (2 * h + 1) * 8 + fn);
#pragma unroll
        for (int f = 0; f < NF; ++f) {
          acc[f] = fma(d[f], ck[f][h].x, acc[f]);
          acc[f] = fma(sm, ck[f][h].y, acc[f]);
        }
      }
    }
  }
#pragma unroll
  for (uint i = 0; i < 2; ++i) {
    const uint row = fn + i;
    if (int(row) >= rows)
      continue;
    const ulong o = ulong(row) * columns + column;
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      if constexpr (SPLIT > 1)
        partials[ulong(position.y) * rows * columns + o + 8 * f] = acc[f][i];
      else
        output[o + 8 * f] = bfloat(acc[f][i]);
    }
  }
}

#define PIE_SG_POINT(name, split)                                                                  \
  template [[host_name(name)]] [[kernel]] void affine_qmm_sg<split>(                               \
      const device uint *, const device bfloat *, const device bfloat *, const device bfloat *,    \
      device bfloat *, constant int &, constant int &, device float *, constant int &,             \
      constant int &, uint2, uint, uint, uint);
