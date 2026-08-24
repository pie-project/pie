// Raw-Metal short causal depthwise convolution over the recurrent conv slab.
//
// Two entrypoints, one per point: `causal_conv1d` is the decode step (one
// input row per request) and `causal_conv1d_chunked` is the prefill window
// (a CSR range of rows per request). They are separate kernels with the same
// arithmetic written twice, because the step's window is entirely in the slab
// and the chunk's is mostly in `x` -- folding them would put a branch on the
// token axis inside the innermost loop of both.
//
// ── THE SLAB IS `[K, C]`, AND `C` IS THE FAST AXIS ──────────────────────────
//
// A slot's conv state is `conv_k * conv_dim` floats -- K ROWS of C channels,
// oldest row first, `state[k * C + c]`. `driver-metal`'s
// `layout::recurrent::Shape::conv_bytes_per_slot` is that product and
// `gdn_core.metal` indexes `conv_state[(slot * Kc + j) * CDIM + c]`, so the
// rectangle stated here is the one the pool allocates.
//
// K rows is the RECTANGLE and not the live window. Between fires only `K - 1`
// of the rows carry taps the next token convolves over: the step reads rows
// `1 .. K-1` and the incoming column, shifts every row down one, and lands the
// new column at row `K - 1`. Row 0 is where the shift's tail goes -- it is
// read once, by the token that arrives before it is overwritten, and never
// again. A declaration that stated `K - 1` rows would be stating the live
// window, which is a different number from the rectangle the kernel indexes,
// and the two differ by exactly the row this paragraph exists to keep.
//
// The arithmetic is `causal_conv1d_update_batched` and
// `causal_conv1d_prefill_batched` from `kernels-cuda/kernels/ssm/`, which is
// where the numeric contract was measured:
//
//     y[t, c] = silu( sum_{k=0..K-1} W[c, k] * x[t - K + 1 + k, c] )
//
// with `x[t < 0, c]` read from the slab at row `K + t`. `silu(z) = z * sigmoid(z)`
// spelled `z / (1 + exp(-z))`, the way both cuda kernels and `gdn_core.metal`
// spell it. There is no bias: `ssm.causal_conv1d` declares none, and the cuda
// claim body passes a null one.
//
// ── WHY THE SHIFTED WINDOW LANDS IN A SECOND PLANE ─────────────────────────
//
// `conv_state` is read-only here and `new_conv_state` is written, which is
// this plane's ping-pong: `driver-metal`'s pool allocates two conv planes per
// layer and `Pool::carry_forward` makes the written one the read one once the
// fire retires. The cuda kernels shift in place because one block owns a
// channel for the whole launch; a Metal dispatch has no such promise across
// threadgroups, and a channel whose taps are being read by the token after it
// cannot be the channel a shift is landing on.
//
// A request the fire does not name keeps whatever its `new_conv_state` rows
// held, which the previous carry-forward made equal to its `conv_state` rows --
// so the copy back is an identity for it. That invariant is why the chunked
// kernel may return early on an empty window rather than copying it forward
// by hand.
//
// Launch (both): dispatchThreads grid=(C, R, 1), tg=(min(C, 256), 1, 1). One
// thread owns one (channel, request) column for the whole kernel, so the
// shift it writes is over rows it alone read.

#include <metal_stdlib>
using namespace metal;

// The one spelling of SiLU both kernels below share, matching
// `pie::ssm::silu_f` in `kernels-cuda/kernels/ssm/causal_conv1d.cuh`.
inline float causal_conv1d_silu(float z) {
  return z / (1.0f + metal::exp(-z));
}

// Decode step: one input row per request, the other `K - 1` taps off the slab.
template <typename T>
[[kernel]] void causal_conv1d(
    const device T* x              [[buffer(0)]],  // [R, C]
    const device T* weight         [[buffer(1)]],  // [C, K]
    const device float* conv_state [[buffer(2)]],  // [slots, K, C] read
    device float* new_conv_state   [[buffer(3)]],  // [slots, K, C] written
    const device uint* slots       [[buffer(4)]],  // [R], one seat per row
    device T* y                    [[buffer(5)]],  // [R, C]
    const constant int& channels   [[buffer(6)]],
    const constant int& conv_width [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int r = int(pos.y);
  const size_t chans = size_t(channels);
  const size_t taps = size_t(conv_width);
  const size_t col = size_t(c);
  const size_t slab = size_t(slots[r]) * taps * chans;
  const size_t tap0 = size_t(c) * taps;

  const float fresh = float(x[size_t(r) * chans + col]);

  // Rows `1 .. K-1` of the slab, then the arriving column at tap `K - 1`.
  float acc = 0.0f;
  for (size_t k = 0; k + 1 < taps; ++k) {
    acc += conv_state[slab + (k + 1) * chans + col] * float(weight[tap0 + k]);
  }
  acc += fresh * float(weight[tap0 + taps - 1]);
  y[size_t(r) * chans + col] = T(causal_conv1d_silu(acc));

  // Shift every row down one and land the arriving column at row `K - 1`.
  for (size_t k = 0; k + 1 < taps; ++k) {
    new_conv_state[slab + k * chans + col] =
        conv_state[slab + (k + 1) * chans + col];
  }
  new_conv_state[slab + (taps - 1) * chans + col] = fresh;
}

// Prefill window: `indptr[r] .. indptr[r + 1]` rows of `x` for request `r`,
// with the taps before the window read off the slab and the trailing `K` rows
// of the window persisted back into it.
template <typename T>
[[kernel]] void causal_conv1d_chunked(
    const device T* x              [[buffer(0)]],  // [N, C]
    const device int* indptr       [[buffer(1)]],  // [R + 1]
    const device T* weight         [[buffer(2)]],  // [C, K]
    const device float* conv_state [[buffer(3)]],  // [slots, K, C] read
    device float* new_conv_state   [[buffer(4)]],  // [slots, K, C] written
    const device uint* slots       [[buffer(5)]],  // [N], one seat per token
    device T* y                    [[buffer(6)]],  // [N, C]
    const constant int& channels   [[buffer(7)]],
    const constant int& conv_width [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int r = int(pos.y);
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  // An empty window leaves BOTH planes alone: the carry-forward that follows
  // the fire copies `new_conv_state` over `conv_state` for every slot, and the
  // previous one already made the two equal here.
  if (end <= begin) {
    return;
  }
  const int span = end - begin;
  const int width = conv_width;

  const size_t chans = size_t(channels);
  const size_t taps = size_t(conv_width);
  const size_t col = size_t(c);
  // Every token of a request sits in the same seat, so the window's first row
  // names it.
  const size_t slab = size_t(slots[begin]) * taps * chans;
  const size_t tap0 = size_t(c) * taps;

  for (int t = 0; t < span; ++t) {
    float acc = 0.0f;
    for (int k = 0; k < width; ++k) {
      const int src = t - (width - 1) + k;
      const float tap = (src < 0)
          ? conv_state[slab + size_t(width + src) * chans + col]
          : float(x[size_t(begin + src) * chans + col]);
      acc += tap * float(weight[tap0 + size_t(k)]);
    }
    y[size_t(begin + t) * chans + col] = T(causal_conv1d_silu(acc));
  }

  // The trailing `K` rows of `[slab window | x window]`, oldest first, which
  // is where a follow-up step resumes from.
  for (int s = 0; s < width; ++s) {
    const int src = span - width + s;
    new_conv_state[slab + size_t(s) * chans + col] = (src < 0)
        ? conv_state[slab + size_t(width + src) * chans + col]
        : float(x[size_t(begin + src) * chans + col]);
  }
}

#define instantiate_causal_conv1d(name, itype)                          \
  template [[host_name("causal_conv1d_" #name)]]                        \
  [[kernel]] void causal_conv1d<itype>(                                 \
      const device itype*, const device itype*, const device float*,    \
      device float*, const device uint*, device itype*,                 \
      const constant int&, const constant int&, uint2);

#define instantiate_causal_conv1d_chunked(name, itype)                  \
  template [[host_name("causal_conv1d_chunked_" #name)]]                \
  [[kernel]] void causal_conv1d_chunked<itype>(                         \
      const device itype*, const device int*, const device itype*,      \
      const device float*, device float*, const device uint*,           \
      device itype*, const constant int&, const constant int&, uint2);

instantiate_causal_conv1d(bfloat16, bfloat)

instantiate_causal_conv1d_chunked(bfloat16, bfloat)
