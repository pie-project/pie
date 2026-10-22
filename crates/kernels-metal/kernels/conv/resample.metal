#include "grid.h"


template <typename T>
[[kernel]] void upsample_nearest(
    const device T* x                 [[buffer(0)]],
    const device int* grid            [[buffer(1)]],
    device T* y                       [[buffer(2)]],
    const device int* o_grid          [[buffer(3)]],
    const constant uint& channels     [[buffer(4)]],
    const constant uint& clips        [[buffer(5)]],
    const constant uint& ft           [[buffer(6)]],
    const constant uint& fh           [[buffer(7)]],
    const constant uint& fw           [[buffer(8)]],
    const constant uint& keep_first   [[buffer(9)]],
    uint2 tid                         [[thread_position_in_grid]]) {
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= channels) return;
  const size_t e = size_t(row) * size_t(channels) + size_t(col);
  SpatialClip og;
  const int l = clip_of(o_grid, int(clips), int(row), og);
  if (l < 0) {
    y[e] = static_cast<T>(0.0f);
    return;
  }
  const SpatialClip ig = clip_at(grid, l);
  const SpatialVoxel o = unravel(og, int(row) - og.off);
  const int ti = keep_first != 0
      ? (o.t == 0 ? 0 : (o.t - 1) / int(ft) + 1)
      : o.t / int(ft);
  const int src = ravel(ig, ti, o.h / int(fh), o.w / int(fw));
  y[e] = x[size_t(src) * size_t(channels) + size_t(col)];
}

#define instantiate_upsample(name, itype)                                  \
  template [[host_name("upsample_nearest_" #name)]]                \
  [[kernel]] void upsample_nearest<itype>(                         \
      const device itype*, const device int*, device itype*,               \
      const device int*, const constant uint&, const constant uint&,       \
      const constant uint&, const constant uint&, const constant uint&,    \
      const constant uint&, uint2);

instantiate_upsample(bfloat16, bfloat)
instantiate_upsample(float32, float)

template <typename T>
[[kernel]] void pixel_shuffle(
    const device T* x                 [[buffer(0)]],
    const device int* grid            [[buffer(1)]],
    device T* y                       [[buffer(2)]],
    const device int* o_grid          [[buffer(3)]],
    const constant uint& channels     [[buffer(4)]],
    const constant uint& clips        [[buffer(5)]],
    const constant uint& r1           [[buffer(6)]],
    const constant uint& r2           [[buffer(7)]],
    const constant uint& r3           [[buffer(8)]],
    const constant uint& trim_t       [[buffer(9)]],
    uint2 tid                         [[thread_position_in_grid]]) {
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= channels) return;
  const size_t e = size_t(row) * size_t(channels) + size_t(col);
  SpatialClip og;
  const int l = clip_of(o_grid, int(clips), int(row), og);
  if (l < 0) {
    y[e] = static_cast<T>(0.0f);
    return;
  }
  const SpatialClip ig = clip_at(grid, l);
  const SpatialVoxel o = unravel(og, int(row) - og.off);
  const int ot = o.t + int(trim_t);
  const int src = ravel(ig, ot / int(r1), o.h / int(r2), o.w / int(r3));
  const int block = ((ot % int(r1)) * int(r2) + (o.h % int(r2))) * int(r3) + (o.w % int(r3));
  const uint r = r1 * r2 * r3;
  const uint c_in = channels * r;
  y[e] = x[size_t(src) * size_t(c_in) + size_t(col) * size_t(r) + size_t(block)];
}

#define instantiate_shuffle(name, itype)                                   \
  template [[host_name("pixel_shuffle_" #name)]]                   \
  [[kernel]] void pixel_shuffle<itype>(                            \
      const device itype*, const device int*, device itype*,               \
      const device int*, const constant uint&, const constant uint&,       \
      const constant uint&, const constant uint&, const constant uint&,    \
      const constant uint&, uint2);

instantiate_shuffle(bfloat16, bfloat)
instantiate_shuffle(float32, float)

template <typename T>
[[kernel]] void pixel_unshuffle(
    const device T* x                 [[buffer(0)]],
    const device int* grid            [[buffer(1)]],
    device T* y                       [[buffer(2)]],
    const device int* o_grid          [[buffer(3)]],
    const constant uint& channels     [[buffer(4)]],
    const constant uint& clips        [[buffer(5)]],
    const constant uint& r1           [[buffer(6)]],
    const constant uint& r2           [[buffer(7)]],
    const constant uint& r3           [[buffer(8)]],
    uint2 tid                         [[thread_position_in_grid]]) {
  const uint r = r1 * r2 * r3;
  const uint c_out = channels * r;
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= c_out) return;
  const size_t e = size_t(row) * size_t(c_out) + size_t(col);
  SpatialClip og;
  const int l = clip_of(o_grid, int(clips), int(row), og);
  if (l < 0) {
    y[e] = static_cast<T>(0.0f);
    return;
  }
  const SpatialClip ig = clip_at(grid, l);
  const SpatialVoxel o = unravel(og, int(row) - og.off);
  const uint cin = col / r;
  const uint block = col - cin * r;
  const uint i1 = block / (r2 * r3);
  const uint i2 = (block / r3) % r2;
  const uint i3 = block % r3;
  const int src = ravel(ig, o.t * int(r1) + int(i1), o.h * int(r2) + int(i2),
                        o.w * int(r3) + int(i3));
  y[e] = x[size_t(src) * size_t(channels) + size_t(cin)];
}

#define instantiate_unshuffle(name, itype)                                 \
  template [[host_name("pixel_unshuffle_" #name)]]                 \
  [[kernel]] void pixel_unshuffle<itype>(                          \
      const device itype*, const device int*, device itype*,               \
      const device int*, const constant uint&, const constant uint&,       \
      const constant uint&, const constant uint&, const constant uint&,    \
      uint2);

instantiate_unshuffle(bfloat16, bfloat)
instantiate_unshuffle(float32, float)

template <typename T>
[[kernel]] void avg_down(
    const device T* x                 [[buffer(0)]],
    const device int* grid            [[buffer(1)]],
    device T* y                       [[buffer(2)]],
    const device int* o_grid          [[buffer(3)]],
    const constant uint& channels     [[buffer(4)]],
    const constant uint& clips        [[buffer(5)]],
    const constant uint& r1           [[buffer(6)]],
    const constant uint& r2           [[buffer(7)]],
    const constant uint& r3           [[buffer(8)]],
    const constant uint& group        [[buffer(9)]],
    uint2 tid                         [[thread_position_in_grid]]) {
  const uint r = r1 * r2 * r3;
  const uint c_out = channels * r / group;
  const uint n = tid.x;
  const uint row = tid.y;
  if (n >= c_out) return;
  const size_t e = size_t(row) * size_t(c_out) + size_t(n);
  SpatialClip og;
  const int l = clip_of(o_grid, int(clips), int(row), og);
  if (l < 0) {
    y[e] = static_cast<T>(0.0f);
    return;
  }
  const SpatialClip ig = clip_at(grid, l);
  const SpatialVoxel o = unravel(og, int(row) - og.off);
  const int pad_t = (int(r1) - ig.t % int(r1)) % int(r1);
  float acc = 0.0f;
  for (uint j = 0; j < group; ++j) {
    const uint q = n * group + j;
    const uint cin = q / r;
    const uint block = q - cin * r;
    const uint i1 = block / (r2 * r3);
    const uint i2 = (block / r3) % r2;
    const uint i3 = block % r3;
    const int ti = o.t * int(r1) + int(i1) - pad_t;
    if (ti < 0) continue;
    const int src = ravel(ig, ti, o.h * int(r2) + int(i2), o.w * int(r3) + int(i3));
    acc += float(x[size_t(src) * size_t(channels) + size_t(cin)]);
  }
  y[e] = static_cast<T>(acc / float(group));
}

#define instantiate_avg_down(name, itype)                                  \
  template [[host_name("avg_down_" #name)]]                        \
  [[kernel]] void avg_down<itype>(                                 \
      const device itype*, const device int*, device itype*,               \
      const device int*, const constant uint&, const constant uint&,       \
      const constant uint&, const constant uint&, const constant uint&,    \
      const constant uint&, uint2);

instantiate_avg_down(bfloat16, bfloat)

template <typename T>
[[kernel]] void cache_gather(
    const device T* slab              [[buffer(0)]],
    const device int* slot_ids        [[buffer(1)]],
    const device int* grid            [[buffer(2)]],
    device T* cache                   [[buffer(3)]],
    const constant uint& clips        [[buffer(4)]],
    const constant uint& frames       [[buffer(5)]],
    const constant uint& channels     [[buffer(6)]],
    const constant uint& slot_stride  [[buffer(7)]],
    uint2 tid                         [[thread_position_in_grid]]) {
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= channels) return;
  const size_t at_out = size_t(row) * size_t(channels) + size_t(col);

  int base = 0;
  int found = -1;
  int at = 0;
  for (uint l = 0; l < clips; ++l) {
    const SpatialClip c = clip_at(grid, int(l));
    const int rows = int(frames) * clip_plane(c);
    if (int(row) < at + rows) {
      base = at;
      found = int(l);
      break;
    }
    at += rows;
  }
  if (found < 0) {
    cache[at_out] = static_cast<T>(0.0f);
    return;
  }
  const size_t local = size_t(row - uint(base)) * size_t(channels) + size_t(col);
  if (local >= size_t(slot_stride)) {
    cache[at_out] = static_cast<T>(0.0f);
    return;
  }
  cache[at_out] = slab[size_t(slot_ids[found]) * size_t(slot_stride) + local];
}

template <typename T>
[[kernel]] void cache_store(
    const device T* x                 [[buffer(0)]],
    const device T* cache             [[buffer(1)]],
    const device int* slot_ids        [[buffer(2)]],
    const device int* grid            [[buffer(3)]],
    device T* slab                    [[buffer(4)]],
    const constant uint& clips        [[buffer(5)]],
    const constant uint& frames       [[buffer(6)]],
    const constant uint& channels     [[buffer(7)]],
    const constant uint& slot_stride  [[buffer(8)]],
    uint2 tid                         [[thread_position_in_grid]]) {
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= channels) return;

  int base = 0;
  int found = -1;
  SpatialClip g;
  int at = 0;
  for (uint l = 0; l < clips; ++l) {
    const SpatialClip c = clip_at(grid, int(l));
    const int rows = int(frames) * clip_plane(c);
    if (int(row) < at + rows) {
      base = at;
      g = c;
      found = int(l);
      break;
    }
    at += rows;
  }
  if (found < 0) return;
  const int local_row = int(row) - base;
  const size_t local = size_t(local_row) * size_t(channels) + size_t(col);
  if (local >= size_t(slot_stride)) return;
  const int plane = clip_plane(g);
  const int f = local_row / plane;
  const int hw = local_row - f * plane;

  const int src_t = g.t - int(frames) + f;
  T value;
  if (src_t >= 0) {
    value = x[(size_t(g.off) + size_t(src_t) * size_t(plane) + size_t(hw)) * size_t(channels)
              + size_t(col)];
  } else {
    value = cache[(size_t(base) + size_t(f + g.t) * size_t(plane) + size_t(hw))
                      * size_t(channels)
                  + size_t(col)];
  }
  slab[size_t(slot_ids[found]) * size_t(slot_stride) + local] = value;
}

#define instantiate_cache_gather(name, itype)                              \
  template [[host_name("cache_gather_" #name)]]                    \
  [[kernel]] void cache_gather<itype>(                             \
      const device itype*, const device int*, const device int*,           \
      device itype*, const constant uint&, const constant uint&,           \
      const constant uint&, const constant uint&, uint2);

instantiate_cache_gather(bfloat16, bfloat)

#define instantiate_cache_store(name, itype)                               \
  template [[host_name("cache_store_" #name)]]                     \
  [[kernel]] void cache_store<itype>(                              \
      const device itype*, const device itype*, const device int*,         \
      const device int*, device itype*, const constant uint&,              \
      const constant uint&, const constant uint&, const constant uint&,    \
      uint2);

instantiate_cache_store(bfloat16, bfloat)
