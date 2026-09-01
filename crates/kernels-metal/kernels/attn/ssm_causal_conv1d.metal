#include <metal_stdlib>
using namespace metal;

// The depthwise causal convolution, DILATED. `dilation` spreads the taps: tap
// `j` reads `dilation * (conv_width - 1 - j)` positions back, so the history a
// lane has to keep is
//
//     hist = (conv_width - 1) * dilation + 1
//
// rows of `channels` and not `conv_width` of them — which is exactly the
// rectangle `model::qwen_4`'s `caches()` declares for the PLE's own conv state
// (`[(conv_kernel - 1) * dilation + 1, streams * hidden]`), and which collapses
// back to `conv_width` at `dilation == 1`. Every index below is the CUDA
// sibling's (`kernels-cuda/kernels/attn/ssm.cuh`,
// `ssm_causal_conv1d_{update,chunked}_batched`) with its `dil` spelled out, so
// the undilated arm is BIT-IDENTICAL to what this file served before: at
// `dil == 1`, `hist == taps`, `(k * dil + 1) == (k + 1)`, and
// `(hist + src) == (width + src)`.
//
// **THE STAGED-GEOMETRY `win` SEAT IS NOT PORTED**, for `ple.metal`'s reason:
// the null path is the semantics and this plane has no recording to replay.

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

  // The window shifts by ONE position — not by one tap — because the history
  // is stored densely: a dilated read strides over it, a step does not.
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
    uint2 pos [[thread_position_in_grid]]) {
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

  for (int s = 0; s < hist; ++s) {
    const int src = span - hist + s;
    new_conv_state[slab + size_t(s) * chans + col] = (src < 0)
        ? conv_state[slab + size_t(hist + src) * chans + col]
        : float(x[size_t(begin + src) * chans + col]);
  }
}

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
      const constant int&, uint2);

instantiate_causal_conv1d(bfloat16, bfloat)

instantiate_causal_conv1d_chunked(bfloat16, bfloat)
