#include <metal_stdlib>
using namespace metal;

#ifndef PIE_RS_SNAPSHOTS
#define PIE_RS_SNAPSHOTS 4
#endif


inline float causal_conv1d_silu(float z) {
  return z / (1.0f + metal::exp(-z));
}

template <typename T>
[[kernel]] void causal_conv1d(
    const device T* x              [[buffer(0)]],
    const device T* weight         [[buffer(1)]],
    const device float* conv_state [[buffer(2)]],
    device float* new_conv_state   [[buffer(3)]],
    const device uint* slots       [[buffer(4)]],
    device T* y                    [[buffer(5)]],
    const constant int& channels   [[buffer(6)]],
    const constant int& conv_width [[buffer(7)]],
    const constant int& dilation   [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int r = int(pos.y);
  const size_t chans = size_t(channels);
  const size_t taps = size_t(conv_width);
  const size_t hist = size_t((conv_width - 1) * dilation + 1);
  const size_t col = size_t(c);
  const size_t slab = size_t(slots[r]) * hist * chans;
  const size_t tap0 = size_t(c) * taps;

  const float fresh = float(x[size_t(r) * chans + col]);

  float acc = 0.0f;
  for (size_t k = 0; k + 1 < taps; ++k) {
    acc += conv_state[slab + (k * size_t(dilation) + 1) * chans + col] *
           float(weight[tap0 + k]);
  }
  acc += fresh * float(weight[tap0 + taps - 1]);
  y[size_t(r) * chans + col] = T(causal_conv1d_silu(acc));

  for (size_t k = 0; k + 1 < hist; ++k) {
    new_conv_state[slab + k * chans + col] =
        conv_state[slab + (k + 1) * chans + col];
  }
  new_conv_state[slab + (hist - 1) * chans + col] = fresh;
}

template <typename T>
[[kernel]] void causal_conv1d_chunked(
    const device T* x              [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device T* weight         [[buffer(2)]],
    const device float* conv_state [[buffer(3)]],
    device float* new_conv_state   [[buffer(4)]],
    const device uint* slots       [[buffer(5)]],
    device T* y                    [[buffer(6)]],
    const constant int& channels   [[buffer(7)]],
    const constant int& conv_width [[buffer(8)]],
    const constant int& dilation   [[buffer(9)]],
    const constant int& run_rows   [[buffer(10)]],
    const device int* snaps        [[buffer(11)]],
    const constant int& snap_lane0 [[buffer(12)]],
    uint3 pos [[thread_position_in_grid]]) {
  // One thread a channel, a lane and a run of at least `run_rows` rows. Only
  // a run that starts before the history ends reads the carried state, and
  // runs are at least that long, so only a lane's first run reads it; that
  // run writes the new state after, as the bank may be the one it read.
  const int c = int(pos.x);
  const int r = int(pos.y);
  const int begin = indptr[r];
  const int end = indptr[r + 1];

  if (end <= begin) {
    return;
  }
  const int span = end - begin;
  const int width = conv_width;
  const int dil = dilation;
  const int hist = (width - 1) * dil + 1;

  const size_t chans = size_t(channels);
  const size_t taps = size_t(conv_width);
  const size_t col = size_t(c);

  const size_t slab = size_t(slots[begin]) * size_t(hist) * chans;
  const size_t tap0 = size_t(c) * taps;

  const int run = max(run_rows, hist);
  const int first = int(pos.z) * run;
  if (first >= span) {
    return;
  }
  const int last = min(span, first + run);
  for (int t = first; t < last; ++t) {
    float acc = 0.0f;
    for (int k = 0; k < width; ++k) {
      const int src = t - (width - 1 - k) * dil;
      const float tap = (src < 0)
          ? conv_state[slab + size_t(hist + src) * chans + col]
          : float(x[size_t(begin + src) * chans + col]);
      acc += tap * float(weight[tap0 + size_t(k)]);
    }
    y[size_t(begin + t) * chans + col] = T(causal_conv1d_silu(acc));
  }
  if (first != 0) {
    return;
  }

  // The state as of each snapshot row lands in a slot of its own, read
  // before the lane's own slot takes its state past the last row.
  if (snap_lane0 >= 0) {
    const device int* snap = snaps + size_t(snap_lane0 + r) * 2 * PIE_RS_SNAPSHOTS;
    for (int i = 0; i < PIE_RS_SNAPSHOTS && snap[i] > 0; ++i) {
      const size_t into = size_t(snap[PIE_RS_SNAPSHOTS + i]) * size_t(hist) * chans;
      for (int s = 0; s < hist; ++s) {
        const int src = snap[i] - hist + s;
        new_conv_state[into + size_t(s) * chans + col] = (src < 0)
            ? conv_state[slab + size_t(hist + src) * chans + col]
            : float(x[size_t(begin + src) * chans + col]);
      }
    }
  }

  for (int s = 0; s < hist; ++s) {
    const int src = span - hist + s;
    new_conv_state[slab + size_t(s) * chans + col] = (src < 0)
        ? conv_state[slab + size_t(hist + src) * chans + col]
        : float(x[size_t(begin + src) * chans + col]);
  }
}

// `causal_conv1d_chunked` with four adjacent channels a thread, each its
// own arithmetic: the loads are vectors and the taps stay in registers.
// Holds widths of at most `PIE_CONV_X4_TAPS`.
#define PIE_CONV_X4_TAPS 8

// Four channels' tap at row `src` of a lane, the carried window's before it.
template <typename T>
inline float4 causal_conv1d_x4_tap(
    const device T* x, const device float* conv_state, size_t slab, size_t chans, size_t col,
    int begin, int hist, int src) {
  return (src < 0)
      ? *reinterpret_cast<const device float4*>(conv_state + slab + size_t(hist + src) * chans + col)
      : float4(*reinterpret_cast<const device vec<T, 4>*>(x + size_t(begin + src) * chans + col));
}

template <typename T>
[[kernel]] void causal_conv1d_chunked_x4(
    const device T* x              [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device T* weight         [[buffer(2)]],
    const device float* conv_state [[buffer(3)]],
    device float* new_conv_state   [[buffer(4)]],
    const device uint* slots       [[buffer(5)]],
    device T* y                    [[buffer(6)]],
    const constant int& channels   [[buffer(7)]],
    const constant int& conv_width [[buffer(8)]],
    const constant int& dilation   [[buffer(9)]],
    const constant int& run_rows   [[buffer(10)]],
    const device int* snaps        [[buffer(11)]],
    const constant int& snap_lane0 [[buffer(12)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x) * 4;
  const int r = int(pos.y);
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  if (end <= begin || c >= channels) {
    return;
  }
  const int span = end - begin;
  const int width = conv_width;
  const int dil = dilation;
  const int hist = (width - 1) * dil + 1;
  const size_t chans = size_t(channels);
  const size_t col = size_t(c);
  const size_t slab = size_t(slots[begin]) * size_t(hist) * chans;
  const int run = max(run_rows, hist);
  const int first = int(pos.z) * run;
  if (first >= span) {
    return;
  }
  const int last = min(span, first + run);

  float4 w[PIE_CONV_X4_TAPS];
  for (int k = 0; k < width; ++k) {
    w[k] = float4(float(weight[(col + 0) * size_t(width) + size_t(k)]),
                  float(weight[(col + 1) * size_t(width) + size_t(k)]),
                  float(weight[(col + 2) * size_t(width) + size_t(k)]),
                  float(weight[(col + 3) * size_t(width) + size_t(k)]));
  }
  for (int t = first; t < last; ++t) {
    float4 acc = 0.0f;
    for (int k = 0; k < width; ++k) {
      const float4 tap = causal_conv1d_x4_tap<T>(x, conv_state, slab, chans, col, begin, hist, t - (width - 1 - k) * dil);
      acc.x += tap.x * w[k].x;
      acc.y += tap.y * w[k].y;
      acc.z += tap.z * w[k].z;
      acc.w += tap.w * w[k].w;
    }
    *reinterpret_cast<device vec<T, 4>*>(y + size_t(begin + t) * chans + col) =
        vec<T, 4>(T(causal_conv1d_silu(acc.x)), T(causal_conv1d_silu(acc.y)),
                  T(causal_conv1d_silu(acc.z)), T(causal_conv1d_silu(acc.w)));
  }
  if (first != 0) {
    return;
  }
  if (snap_lane0 >= 0) {
    const device int* snap = snaps + size_t(snap_lane0 + r) * 2 * PIE_RS_SNAPSHOTS;
    for (int i = 0; i < PIE_RS_SNAPSHOTS && snap[i] > 0; ++i) {
      const size_t into = size_t(snap[PIE_RS_SNAPSHOTS + i]) * size_t(hist) * chans;
      for (int s = 0; s < hist; ++s) {
        *reinterpret_cast<device float4*>(new_conv_state + into + size_t(s) * chans + col) =
            causal_conv1d_x4_tap<T>(x, conv_state, slab, chans, col, begin, hist, snap[i] - hist + s);
      }
    }
  }
  for (int s = 0; s < hist; ++s) {
    *reinterpret_cast<device float4*>(new_conv_state + slab + size_t(s) * chans + col) =
        causal_conv1d_x4_tap<T>(x, conv_state, slab, chans, col, begin, hist, span - hist + s);
  }
}

#define instantiate_causal_conv1d_chunked_x4(name, itype)               \
  template [[host_name("causal_conv1d_chunked_x4_" #name)]]             \
  [[kernel]] void causal_conv1d_chunked_x4<itype>(                      \
      const device itype*, const device int*, const device itype*,      \
      const device float*, device float*, const device uint*,           \
      device itype*, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const device int*,      \
      const constant int&, uint3);

instantiate_causal_conv1d_chunked_x4(bfloat16, bfloat)

#define instantiate_causal_conv1d(name, itype)                          \
  template [[host_name("causal_conv1d_" #name)]]                        \
  [[kernel]] void causal_conv1d<itype>(                                 \
      const device itype*, const device itype*, const device float*,    \
      device float*, const device uint*, device itype*,                 \
      const constant int&, const constant int&, const constant int&,    \
      uint2);

#define instantiate_causal_conv1d_chunked(name, itype)                  \
  template [[host_name("causal_conv1d_chunked_" #name)]]                \
  [[kernel]] void causal_conv1d_chunked<itype>(                         \
      const device itype*, const device int*, const device itype*,      \
      const device float*, device float*, const device uint*,           \
      device itype*, const constant int&, const constant int&,          \
      const constant int&, const constant int&, const device int*,      \
      const constant int&, uint3);

instantiate_causal_conv1d(bfloat16, bfloat)

instantiate_causal_conv1d_chunked(bfloat16, bfloat)

template <typename T>
[[kernel]] void causal_conv1d_committed(
    const device T* x              [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device int* replay       [[buffer(2)]],
    const device int* commit       [[buffer(3)]],
    const device int* slots        [[buffer(4)]],
    const constant int& lane0      [[buffer(5)]],
    const device T* weight         [[buffer(6)]],
    const device float* conv_state [[buffer(7)]],
    device float* new_conv_state   [[buffer(8)]],
    device T* y                    [[buffer(9)]],
    const constant int& channels   [[buffer(10)]],
    const constant int& conv_width [[buffer(11)]],
    const constant int& dilation   [[buffer(12)]],
    const device int* snaps        [[buffer(13)]],
    const constant int& snap_lane0 [[buffer(14)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int r = int(pos.y);
  int begin = indptr[r];
  for (int j = 0; j < r; ++j) {
    begin += replay[lane0 + j];
  }
  const int span = (indptr[r + 1] - indptr[r]) + replay[lane0 + r];
  if (span <= 0) {
    return;
  }
  const int width = conv_width;
  const int dil = dilation;
  const int hist = (width - 1) * dil + 1;

  const size_t chans = size_t(channels);
  const size_t taps = size_t(conv_width);
  const size_t col = size_t(c);

  const int slot = slots[lane0 + r];
  if (slot < 0) {
    return;
  }
  const size_t slab = size_t(slot) * size_t(hist) * chans;
  const size_t tap0 = size_t(c) * taps;

  for (int t = 0; t < span; ++t) {
    float acc = 0.0f;
    for (int k = 0; k < width; ++k) {
      const int src = t - (width - 1 - k) * dil;
      const float tap = (src < 0)
          ? conv_state[slab + size_t(hist + src) * chans + col]
          : float(x[size_t(begin + src) * chans + col]);
      acc += tap * float(weight[tap0 + size_t(k)]);
    }
    y[size_t(begin + t) * chans + col] = T(causal_conv1d_silu(acc));
  }

  // A lane that snapshots folds its own rows, none replayed.
  if (snap_lane0 >= 0) {
    const device int* snap = snaps + size_t(snap_lane0 + r) * 2 * PIE_RS_SNAPSHOTS;
    for (int i = 0; i < PIE_RS_SNAPSHOTS && snap[i] > 0; ++i) {
      const size_t into = size_t(snap[PIE_RS_SNAPSHOTS + i]) * size_t(hist) * chans;
      for (int s = 0; s < hist; ++s) {
        const int src = snap[i] - hist + s;
        new_conv_state[into + size_t(s) * chans + col] = (src < 0)
            ? conv_state[slab + size_t(hist + src) * chans + col]
            : float(x[size_t(begin + src) * chans + col]);
      }
    }
  }

  int keep = commit[lane0 + r];
  if (keep > span) {
    keep = span;
  }
  if (keep <= 0) {
    return;
  }

  float next[64];
  for (int s = 0; s < hist && s < 64; ++s) {
    const int src = keep - hist + s;
    next[s] = (src < 0)
        ? conv_state[slab + size_t(hist + src) * chans + col]
        : float(x[size_t(begin + src) * chans + col]);
  }
  for (int s = 0; s < hist && s < 64; ++s) {
    new_conv_state[slab + size_t(s) * chans + col] = next[s];
  }
}

#define instantiate_causal_conv1d_committed(name, itype)                \
  template [[host_name("causal_conv1d_committed_" #name)]]              \
  [[kernel]] void causal_conv1d_committed<itype>(                       \
      const device itype*, const device int*, const device int*,        \
      const device int*, const device int*, const constant int&,        \
      const device itype*, const device float*, device float*,          \
      device itype*, const constant int&, const constant int&,          \
      const constant int&, const device int*, const constant int&,      \
      uint2);

instantiate_causal_conv1d_committed(bfloat16, bfloat)

// The history the verify conv commits, at most: a thread of it holds the
// rows it writes back until the threadgroup has read them.
constant constexpr int kConvVerifyHist = 16;

// `causal_conv1d_committed` with the rows of a lane spread over the
// threadgroup's `rows` threads of a channel: a verify window's dozen rows,
// one thread a channel walking them in turn, left the conv waiting on each
// row's loads. Every row's output lands in the extended plane `y` the scan
// reads, and a row of the window's own in `land` too, at its own row. The
// window as of `commit` rows is read before any of it is written back: the
// bank is read and written in place.
template <typename T>
[[kernel]] void causal_conv1d_verify(
    const device T* x              [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device int* replay       [[buffer(2)]],
    const device int* commit       [[buffer(3)]],
    const device int* slots        [[buffer(4)]],
    const constant int& lane0      [[buffer(5)]],
    const device T* weight         [[buffer(6)]],
    const device float* conv_state [[buffer(7)]],
    device float* new_conv_state   [[buffer(8)]],
    device T* y                    [[buffer(9)]],
    device T* land                 [[buffer(10)]],
    const constant int& channels   [[buffer(11)]],
    const constant int& conv_width [[buffer(12)]],
    const constant int& dilation   [[buffer(13)]],
    const device int* snaps        [[buffer(14)]],
    const constant int& snap_lane0 [[buffer(15)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 at [[thread_position_in_threadgroup]],
    uint3 shape [[threads_per_threadgroup]]) {
  const int c = int(pos.x);
  const int first = int(at.y);
  const int rows = int(shape.y);
  const int r = int(pos.z);
  const bool live = c < channels;
  int begin = indptr[r];
  for (int j = 0; j < r; ++j) {
    begin += replay[lane0 + j];
  }
  const int rep = replay[lane0 + r];
  const int span = (indptr[r + 1] - indptr[r]) + rep;
  const int slot = slots[lane0 + r];
  // Uniform over the threadgroup, so its barrier below is too.
  if (span <= 0 || slot < 0) {
    return;
  }
  const int width = conv_width;
  const int dil = dilation;
  const int hist = (width - 1) * dil + 1;
  const size_t chans = size_t(channels);
  const size_t col = size_t(c);
  const size_t slab = size_t(slot) * size_t(hist) * chans;
  const size_t tap0 = size_t(c) * size_t(width);
  const int own = indptr[r] - rep;
  // The window's row `src`, the bank's rows before the lane's first.
  auto tap = [&](int src) -> float {
    return (src < 0) ? conv_state[slab + size_t(hist + src) * chans + col]
                     : float(x[size_t(begin + src) * chans + col]);
  };

  for (int t = first; live && t < span; t += rows) {
    float acc = 0.0f;
    for (int k = 0; k < width; ++k) {
      acc += tap(t - (width - 1 - k) * dil) * float(weight[tap0 + size_t(k)]);
    }
    const T out = T(causal_conv1d_silu(acc));
    y[size_t(begin + t) * chans + col] = out;
    if (t >= rep) {
      land[size_t(own + t) * chans + col] = out;
    }
  }

  // A lane that snapshots folds its own rows, none replayed; snapshots land
  // in slots of their own, read from the bank before it is written.
  if (live && snap_lane0 >= 0) {
    const device int* snap = snaps + size_t(snap_lane0 + r) * 2 * PIE_RS_SNAPSHOTS;
    for (int i = 0; i < PIE_RS_SNAPSHOTS && snap[i] > 0; ++i) {
      const size_t into = size_t(snap[PIE_RS_SNAPSHOTS + i]) * size_t(hist) * chans;
      for (int s = first; s < hist; s += rows) {
        new_conv_state[into + size_t(s) * chans + col] = tap(snap[i] - hist + s);
      }
    }
  }

  const int keep = min(commit[lane0 + r], span);
  float next[kConvVerifyHist];
  int held = 0;
  for (int s = first; live && keep > 0 && s < hist; s += rows) {
    next[held++] = tap(keep - hist + s);
  }
  threadgroup_barrier(mem_flags::mem_device);
  held = 0;
  for (int s = first; live && keep > 0 && s < hist; s += rows) {
    new_conv_state[slab + size_t(s) * chans + col] = next[held++];
  }
}

#define instantiate_causal_conv1d_verify(name, itype)                   \
  template [[host_name("causal_conv1d_verify_" #name)]]                 \
  [[kernel]] void causal_conv1d_verify<itype>(                          \
      const device itype*, const device int*, const device int*,        \
      const device int*, const device int*, const constant int&,        \
      const device itype*, const device float*, device float*,          \
      device itype*, device itype*, const constant int&,                \
      const constant int&, const constant int&, const device int*,      \
      const constant int&, uint3, uint3, uint3);

instantiate_causal_conv1d_verify(bfloat16, bfloat)
