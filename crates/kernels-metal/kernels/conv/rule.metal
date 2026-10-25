#include "grid.h"


inline int conv_axis(int n, int k, int s, int front, int back, thread bool& ok) {
    const int span = n + front + back - k;
    if (span < 0) { ok = false; return 0; }
    return span / (s > 0 ? s : 1) + 1;
}

[[kernel]] void grid_rule(
    const device int* grid            [[buffer(0)]],
    device int* out                   [[buffer(1)]],
    const constant int& kind          [[buffer(2)]],
    const constant int& a0            [[buffer(3)]],
    const constant int& a1            [[buffer(4)]],
    const constant int& a2            [[buffer(5)]],
    const constant int& b0            [[buffer(6)]],
    const constant int& b1            [[buffer(7)]],
    const constant int& b2            [[buffer(8)]],
    const constant int& c0            [[buffer(9)]],
    const constant int& c1            [[buffer(10)]],
    const constant int& c2            [[buffer(11)]],
    const constant int& d0            [[buffer(12)]],
    const constant int& d1            [[buffer(13)]],
    const constant int& d2            [[buffer(14)]],
    const constant int& flag          [[buffer(15)]],
    const constant int& clips         [[buffer(16)]],
    uint gid                          [[thread_position_in_grid]]) {
  if (gid != 0) return;
  struct { int kind, a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2, flag, clips; } g =
      {kind, a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2, flag, clips};
  long off = 0;
  for (int l = 0; l < g.clips; ++l) {
    const int t = grid[4 * l], h = grid[4 * l + 1], w = grid[4 * l + 2];
    int ot = 0, oh = 0, ow = 0;
    bool ok = true;
    switch (g.kind) {
      case 0: {
        const int back_t = g.flag ? 0 : g.d0;
        ot = conv_axis(t, g.a0, g.b0, g.c0, back_t, ok);
        oh = conv_axis(h, g.a1, g.b1, g.c1, g.d1, ok);
        ow = conv_axis(w, g.a2, g.b2, g.c2, g.d2, ok);
        break;
      }
      case 1:
        ot = (g.flag != 0 && t > 0) ? 1 + (t - 1) * g.a0 : t * g.a0;
        oh = h * g.a1;
        ow = w * g.a2;
        break;
      case 2:

        ot = t * g.a0 - g.flag;
        oh = h * g.a1;
        ow = w * g.a2;
        ok = ot > 0;
        break;
      case 3:
        ok = g.a0 > 0 && g.a1 > 0 && g.a2 > 0
            && t % g.a0 == 0 && h % g.a1 == 0 && w % g.a2 == 0;
        if (ok) { ot = t / g.a0; oh = h / g.a1; ow = w / g.a2; }
        break;
      default:
        ok = g.a0 > 0 && g.a1 > 0 && g.a2 > 0
            && h % g.a1 == 0 && w % g.a2 == 0;
        if (ok) { ot = (t + g.a0 - 1) / g.a0; oh = h / g.a1; ow = w / g.a2; }
        break;
    }
    if (!ok) { ot = 0; oh = 0; ow = 0; }
    out[4 * l] = ot;
    out[4 * l + 1] = oh;
    out[4 * l + 2] = ow;
    out[4 * l + 3] = int(off);
    off += long(ot) * long(oh) * long(ow);
  }
}

// Each token row's cell: a row of a clip of `tgrid` its patch's (t, h, w)
// there, any other row positions[row] in column `axis`; zeros elsewhere.
[[kernel]] void cells(
    const device int* tgrid           [[buffer(0)]],
    const device int* positions       [[buffer(1)]],
    device float* y                   [[buffer(2)]],
    const constant int& clips         [[buffer(3)]],
    const constant int& axis          [[buffer(4)]],
    const constant int& width         [[buffer(5)]],
    const constant int& rows          [[buffer(6)]],
    uint gid                          [[thread_position_in_grid]]) {
  const int row = int(gid);
  if (row >= rows) return;
  device float* o = y + size_t(width) * size_t(row);
  for (int c = 0; c < width; ++c) o[c] = 0.0f;
  SpatialClip g;
  if (clip_of(tgrid, clips, row, g) < 0) {
    o[axis] = float(positions[row]);
    return;
  }
  const SpatialVoxel v = unravel(g, row - g.off);
  o[0] = float(v.t);
  o[1] = float(v.h);
  o[2] = float(v.w);
}

// Each row's count of the rows `csr` packs in its lane's group.
[[kernel]] void group_rows(
    const device int* csr             [[buffer(0)]],
    const device int* groups          [[buffer(1)]],
    const device int* lanes           [[buffer(2)]],
    device float* y                   [[buffer(3)]],
    const constant int& lane_count    [[buffer(4)]],
    const constant int& group_count   [[buffer(5)]],
    const constant int& rows          [[buffer(6)]],
    uint gid                          [[thread_position_in_grid]]) {
  const int row = int(gid);
  if (row >= rows) return;
  const int l = lanes[row];
  const int g = (l >= 0 && l < lane_count) ? groups[l] : -1;
  y[row] = (g >= 0 && g < group_count) ? float(csr[g + 1] - csr[g]) : 0.0f;
}
