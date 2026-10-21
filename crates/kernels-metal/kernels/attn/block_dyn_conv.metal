

#include <metal_stdlib>
using namespace metal;

// A thread per channel and row: row `pos.y` of the lane whose span holds it, each of its taps
// reaching back no further than the lane's first row. A row past every lane is left alone.
template <typename T>
[[kernel]] void block_dyn_conv(
    const device T* x              [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device T* coeff          [[buffer(2)]],
    const device T* base           [[buffer(3)]],
    device T* y                    [[buffer(4)]],
    const constant int& channels   [[buffer(5)]],
    const constant int& side       [[buffer(6)]],
    const constant int& taps       [[buffer(7)]],
    const constant int& group      [[buffer(8)]],
    const constant int& lanes      [[buffer(9)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int row = int(pos.y);
  if (c >= channels || row < indptr[0] || row >= indptr[lanes]) {
    return;
  }
  // The lane whose span [indptr[r], indptr[r + 1]) holds the row.
  int lo = 0, hi = lanes;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (indptr[mid] <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const int begin = indptr[lo];
  const int t = row - begin;
  const int groups = channels / group;
  const int g = c / group;
  const size_t chans = size_t(channels);
  const size_t pitch = size_t(2 * taps) * size_t(groups);

  float acc = 0.0f;
  for (int k = 0; k < taps; ++k) {
    const int src = t - k;
    if (src < 0) {
      break;
    }
    const int at = side * taps + k;
    const float coef = float(base[size_t(at) * chans + size_t(c)])
                     + float(coeff[size_t(row) * pitch + size_t(at) * size_t(groups) + size_t(g)]);
    acc += coef * float(x[size_t(begin + src) * chans + size_t(c)]);
  }
  y[size_t(row) * chans + size_t(c)] = T(acc);
}

#define instantiate_block_dyn_conv(name, itype)                         \
  template [[host_name("block_dyn_conv_" #name)]]                       \
  [[kernel]] void block_dyn_conv<itype>(                                \
      const device itype*, const device int*, const device itype*,      \
      const device itype*, device itype*, const constant int&,          \
      const constant int&, const constant int&, const constant int&,    \
      const constant int&, uint2);

instantiate_block_dyn_conv(bfloat16, bfloat)
