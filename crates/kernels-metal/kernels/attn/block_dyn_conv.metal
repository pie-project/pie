// block_dyn_conv.metal — DFlash2's two-tap grouped dynamic convolution.
//
// Within one request's span of rows, every row mixes itself with the row
// before it, and the mixing coefficients are the row's OWN: a learned
// per-channel base plus a per-group correction the sublayer's input projected
// (`kernel_projection`, both sides at once). The reference
// (`mlx_dspark.dflash_model.DFlashGroupedConv._convolve`):
//
//     coeff[i, t, c] = base[side, t, c] + delta[i, t, g(c)]
//     y[i, c]        = Σ_t coeff[i, t, c] · x[i − t, c],   x[i − t] = 0 for i < t
//
// applied to the block rows alone — position 0 (the anchor) has no in-block
// predecessor, position 1 reads the anchor — which is exactly the zero fill
// at each request's first row here. One thread per (channel, request), the
// span walked in order; a draft block is eight rows, so the walk is short,
// and any longer span is merely correct.
//
// `coeff` rows are `[2 · taps · groups]` laid `(side, tap, group)`; `base` is
// `[2 · taps, channels]` at row `side · taps + tap`. Accumulated in f32 and
// rounded once, where the reference accumulates in bf16 — an ulp-class
// parting, the same one every other op in this plane takes.

#include <metal_stdlib>
using namespace metal;

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
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int r = int(pos.y);
  if (c >= channels) {
    return;
  }
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  if (end <= begin) {
    return;
  }
  const int span = end - begin;
  const int groups = channels / group;
  const int g = c / group;
  const size_t chans = size_t(channels);
  const size_t pitch = size_t(2 * taps) * size_t(groups);

  for (int t = 0; t < span; ++t) {
    const size_t row = size_t(begin + t);
    float acc = 0.0f;
    for (int k = 0; k < taps; ++k) {
      const int src = t - k;
      if (src < 0) {
        break;
      }
      const int at = side * taps + k;
      const float coef = float(base[size_t(at) * chans + size_t(c)])
                       + float(coeff[row * pitch + size_t(at) * size_t(groups) + size_t(g)]);
      acc += coef * float(x[size_t(begin + src) * chans + size_t(c)]);
    }
    y[row * chans + size_t(c)] = T(acc);
  }
}

#define instantiate_block_dyn_conv(name, itype)                         \
  template [[host_name("block_dyn_conv_" #name)]]                       \
  [[kernel]] void block_dyn_conv<itype>(                                \
      const device itype*, const device int*, const device itype*,      \
      const device itype*, device itype*, const constant int&,          \
      const constant int&, const constant int&, const constant int&,    \
      uint2);

instantiate_block_dyn_conv(bfloat16, bfloat)
