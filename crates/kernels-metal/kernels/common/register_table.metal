#include <metal_stdlib>
using namespace metal;

// The register tile's input table (`qmm_sg_table` in linear/quant_qmm_sg.metal)
// as a producer of the tile's rows lays it while it writes them: eight rows of
// `width` inputs, each at its lane slot; then each row's sum of each 32 inputs,
// [group][row] f32, at `table` + 8 * width; then the split tile's arrival
// counters, zeroed. Bit for bit the table `qmm_sg_table` lays over the rows.

namespace register_table {

// Where input `k` of row `row` sits: for 64-input group h, input k = 64h +
// 8(2cp + j) + p + 4i is step (j, p) of B lane (fm, c) = (2cp + i, row / 2),
// its row's half of the pair.
inline ulong slot(uint k, uint row) {
  const uint o = k & 63;
  const uint p = o & 3, i = (o >> 2) & 1, j = (o >> 3) & 1, cp = o >> 4;
  return (ulong(k >> 6) * 32 + (2 * cp + i) * 4 + (row >> 1)) * 16 + 8 * j + 2 * p + (row & 1);
}

// The sum of the 32 inputs of half `q` of group `h` of a row read from `src`
// (indexed by input), added in the order `qmm_sg_table` adds them: each lane
// its 8 in step order, then the lanes of the two i, then of the two cp.
template <typename P>
inline float half_sum(P src, uint h, uint q) {
#pragma clang fp reassociate(off)
  float part[2][2];
  for (uint cl = 0; cl < 2; ++cl)
    for (uint i = 0; i < 2; ++i) {
      const uint cp = 2 * q + cl;
      float sum = 0.0f;
      for (uint j = 0; j < 2; ++j)
        for (uint p = 0; p < 4; ++p)
          sum += float(src[64 * h + 8 * (2 * cp + j) + p + 4 * i]);
      part[cl][i] = sum;
    }
  return (part[0][0] + part[0][1]) + (part[1][0] + part[1][1]);
}

inline device float *sums(device bfloat *table, uint width) {
  return reinterpret_cast<device float *>(table + 8 * ulong(width));
}

// Thread `at` of `threads` lays row `row`'s sums from `src`.
template <typename P>
inline void lay_sums(P src, device bfloat *table, uint width, uint row, uint at, uint threads) {
  device float *s = sums(table, width);
  for (uint n = at; n < width / 32; n += threads)
    s[ulong(n) * 8 + row] = half_sum(src, n >> 1, n & 1);
}

// Thread `at` of `threads` lays rows `rows`..7 as zeros and zeroes the
// `counters` arrival counters.
inline void lay_rest(device bfloat *table, uint width, uint rows, uint counters, uint at,
                     uint threads) {
  device float *s = sums(table, width);
  for (uint row = rows; row < 8; ++row) {
    for (uint k = at; k < width; k += threads)
      table[slot(k, row)] = bfloat(0.0f);
    for (uint n = at; n < width / 32; n += threads)
      s[ulong(n) * 8 + row] = 0.0f;
  }
  device uint *count = reinterpret_cast<device uint *>(s) + width / 4;
  for (uint n = at; n < counters; n += threads)
    count[n] = 0;
}

} // namespace register_table
