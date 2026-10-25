#include <metal_simdgroup>
#include "grid.h"

struct ConvGeom {
    int c_in;
    int c_out;
    int kt, kh, kw;
    int st, sh, sw;

    int pt;
    int ph, pw;

    int causal;

    int replicate;
    int clips;
    int rows_out;

    int has_cache;
};

inline int conv_tap_row(
    thread ConvGeom& g,
    SpatialClip in,
    SpatialVoxel o,
    int cache_base,
    int it, int ih, int iw,
    thread bool& from_cache) {
  from_cache = false;
  const int hi = o.h * g.sh - g.ph + ih;
  const int wi = o.w * g.sw - g.pw + iw;
  if (hi < 0 || hi >= in.h || wi < 0 || wi >= in.w) return -1;
  int ti = o.t * g.st - g.pt + it;
  if (ti < 0) {
    if (g.causal != 0 && g.has_cache != 0) {
      from_cache = true;
      return cache_base + ((ti + g.pt) * in.h + hi) * in.w + wi;
    }
    if (g.replicate == 0) return -1;
    ti = 0;
  }
  if (ti >= in.t) {

    if (g.causal != 0 || g.replicate == 0) return -1;
    ti = in.t - 1;
  }
  return ravel(in, ti, hi, wi);
}

template <typename T>
[[kernel]] void conv3d(
    const device T* x                 [[buffer(0)]],
    const device int* grid            [[buffer(1)]],
    const device T* w                 [[buffer(2)]],
    const device float* bias          [[buffer(3)]],
    const device T* cache             [[buffer(4)]],
    device T* o                       [[buffer(5)]],
    const device int* o_grid          [[buffer(6)]],
    const constant int& c_in          [[buffer(7)]],
    const constant int& c_out         [[buffer(8)]],
    const constant int& kt            [[buffer(9)]],
    const constant int& kh            [[buffer(10)]],
    const constant int& kw            [[buffer(11)]],
    const constant int& st            [[buffer(12)]],
    const constant int& sh            [[buffer(13)]],
    const constant int& sw            [[buffer(14)]],
    const constant int& pt            [[buffer(15)]],
    const constant int& ph            [[buffer(16)]],
    const constant int& pw            [[buffer(17)]],
    const constant int& causal        [[buffer(18)]],
    const constant int& replicate     [[buffer(19)]],
    const constant int& clips         [[buffer(20)]],
    const constant int& rows_out      [[buffer(21)]],
    const constant int& has_cache     [[buffer(22)]],
    const constant uint& has_bias     [[buffer(23)]],
    const constant int& x_base        [[buffer(24)]],
    const constant int& o_base        [[buffer(25)]],
    uint2 tgid                        [[threadgroup_position_in_grid]],
    uint simd_gid                     [[simdgroup_index_in_threadgroup]],
    uint simd_lid                     [[thread_index_in_simdgroup]],
    uint simds                        [[simdgroups_per_threadgroup]]) {
  ConvGeom g;
  g.c_in = c_in; g.c_out = c_out;
  g.kt = kt; g.kh = kh; g.kw = kw;
  g.st = st; g.sh = sh; g.sw = sw;
  g.pt = pt; g.ph = ph; g.pw = pw;
  g.causal = causal; g.replicate = replicate;
  g.clips = clips; g.rows_out = rows_out; g.has_cache = has_cache;
  const int row = int(tgid.y);
  const int n = int(tgid.x * simds + simd_gid);
  if (row >= g.rows_out || n >= g.c_out) return;

  // `x_base` / `o_base`: `x`'s and `o`'s row 0 in the grids' fire-wide rows.
  SpatialClip og;
  const int fire_row = row + o_base;
  const int l = clip_of(o_grid, g.clips, fire_row, og);
  device T* out = o + size_t(row) * size_t(g.c_out) + size_t(n);
  if (l < 0) {
    if (simd_lid == 0) *out = static_cast<T>(0.0f);
    return;
  }
  SpatialClip in = clip_at(grid, l);
  in.off -= x_base;
  const SpatialVoxel v = unravel(og, fire_row - og.off);

  int planes = 0;
  for (int j = 0; j < l; ++j) planes += clip_plane(clip_at(grid, j));
  const int cache_base = planes * g.pt;

  const int taps = g.kt * g.kh * g.kw;
  const device T* wrow = w + size_t(n) * size_t(taps) * size_t(g.c_in);
  float acc = 0.0f;
  for (int tap = 0; tap < taps; ++tap) {
    const int plane = g.kh * g.kw;
    const int it = tap / plane;
    const int rest = tap - it * plane;
    const int ih = rest / g.kw;
    const int iw = rest - ih * g.kw;
    bool from_cache = false;
    const int src = conv_tap_row(g, in, v, cache_base, it, ih, iw, from_cache);
    if (src < 0) continue;
    const device T* xr =
        (from_cache ? cache : x) + size_t(src) * size_t(g.c_in);
    const device T* wt = wrow + size_t(tap) * size_t(g.c_in);
    for (int c = int(simd_lid); c < g.c_in; c += 32) {
      acc = fma(float(xr[c]), float(wt[c]), acc);
    }
  }
  acc = simd_sum(acc);
  if (simd_lid == 0) {
    *out = static_cast<T>(has_bias != 0 ? acc + bias[n] : acc);
  }
}

#define instantiate_conv3d(name, itype)                            \
  template [[host_name("conv3d_" #name)]]                          \
  [[kernel]] void conv3d<itype>(                                   \
      const device itype*, const device int*, const device itype*,         \
      const device float*, const device itype*, device itype*,             \
      const device int*,                                                   \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, const constant uint&,                           \
      const constant int&, const constant int&,                            \
      uint2, uint, uint, uint);

instantiate_conv3d(bfloat16, bfloat)

