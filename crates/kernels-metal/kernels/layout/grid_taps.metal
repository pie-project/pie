#include <metal_stdlib>
using namespace metal;

// Position-table taps for each patch: bilinear (kind 0) over its image's
// grid stretched onto a side x side table, or axes (kind 1), one unit tap
// on the column into the first `side` rows and one on the row into the
// next `side`.
static inline void axis_taps(int index, int size, int side, thread int* taps, thread float* w) {
  const float src = float(index) * float(side - 1) / float(max(size - 1, 1));
  const float fl = floor(src);
  for (int t = 0; t < 2; ++t) {
    taps[t] = clamp(int(fl) + t, 0, side - 1);
    w[t] = max(1.0f - fabs(src - fl - float(t)), 0.0f);
  }
}

[[kernel]] void grid_taps(
    const device int* positions  [[buffer(0)]],
    const device int* grids      [[buffer(1)]],
    const device int* segments   [[buffer(2)]],
    device int* ids              [[buffer(3)]],
    device float* weights        [[buffer(4)]],
    const constant int& kind     [[buffer(5)]],
    const constant int& side     [[buffer(6)]],
    const constant int& images   [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  const int n = int(tid);
  int lo = 0, hi = images;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (segments[mid] <= n) lo = mid; else hi = mid;
  }
  const int row = positions[n * 3 + 1];
  const int col = positions[n * 3 + 2];
  if (kind == 1) {
    ids[n * 2] = min(col, side - 1);
    ids[n * 2 + 1] = side + min(row, side - 1);
    weights[n * 2] = 1.0f;
    weights[n * 2 + 1] = 1.0f;
    return;
  }
  const int gh = grids[lo * 3 + 1];
  const int gw = grids[lo * 3 + 2];
  int h_taps[2], w_taps[2];
  float h_w[2], w_w[2];
  axis_taps(row, gh, side, h_taps, h_w);
  axis_taps(col, gw, side, w_taps, w_w);
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      ids[n * 4 + a * 2 + b] = h_taps[a] * side + w_taps[b];
      weights[n * 4 + a * 2 + b] = h_w[a] * w_w[b];
    }
  }
}
